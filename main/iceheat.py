#%% Ice Heat

# 1D Heat Equation solver for block of ice floating on seawater using a
# Crank-Nicolson method
import numpy as np
#import matplotlib.pyplot as plt
from scipy import optimize
from scipy import sparse
#from scipy.sparse import linalg, diags

#%% Constants in SI Units
alpha = 0.6; # albedo
eps_ice = 0.96; # emissivity
sigma = 5.67*10**(-8); # Stefan-Boltzmann constant, W/(m^2K^4) 
lw_in = 200; # incoming longwave, W/m^2
rho_a = 1.225; # density of air, kg/m^3
rho_w  = 1020; # density of SEA water, kg/m^3
rho_i = 916.7; # density of ice, kg/m^3
V_a = 5; # velocity of air, m/s
u_star_top = 0.2; # friction velocity of atmosphere, m/s
V_w = 1; # velocity of water, m/s
u_star_bottom = 0.1; #friction velocity of water, m/s
T_w = 272.15; # temperature of bulk water, K
c_pa = 1004; # specific heat of air, J/(kgK)
c_pw = 3985; # specific heat of SEA water, J/(kgK)
c_pi = 2027; # specific heat of ice, J/(kgK)
c_h = 0.01737; # bulk transfer coefficient for latent heat
c_e = 0.01737; # bulk transer coefficient for sensible heat
Lv = 2500000; # latent heat of vaporization, J/kg
Lf = 334000; # latent heat of fusion, J/kg
Ls = Lv + Lf; # latent heat of sublimation, J/kg
p_a = 101325; # pressure of air,  Pa
R_d = 287.0; # dry air constant, J/(kgK)
R_v = 461.5; # water vapor gas constant, J/(kgK)
eps_R = R_d/R_v; # ratio of gas constants
kappa_ice = 2.25; # thermal conductivity of ice, W/(mK)
q_a = .0018; # specific humidity of air, kg_w/kg_a
alpha_ice = (kappa_ice)/(c_pi*rho_i); # thermal diffusivity of ice, m^2/s

#%% Physical functions

# Calculate q from T
def ice_q(T):
    e = 611*np.exp((Ls/R_v)*((1.0/273.0)-(1.0/T)))
    q = (eps_R*e)/(p_a+e*(eps_R-1))
    return q

#Function to calculate incoming solar value, starting at midnight
def sw_net(t): #t is in seconds, so dt*i would be evaluated
    t_hours = (t/3600.0)%24
    if (t_hours) < 7 or (t_hours) > 22:
        #print(f"hour = {(i*dt/(3600.0))%24}, dark")
        sw_in = 0.0
    else:
        sw_in = 500*np.sin((np.pi/15.0)*(t_hours-7.0))
        #print(f"hour = {(i*dt/(3600.0))%24}, light")
        #print(f"sw_in = {sw_in}")
    shortwave_net = (1-alpha)*sw_in; # net shortwave, W/m^2 
    return shortwave_net
# the initial value will thus be sw_net(0.0), evaluated below

# Define function to calculate temperature based on time of day
def air_temp(t): #t is in seconds, so dt*i would be evaluated
    t_hours = (t/3600.0)%24
    temp = 7.0*np.sin((np.pi/12.0)*(t_hours-13.0))+268
    return temp
# the initial value for temperature would be air_temp(0.0)

#%% Numerical Parameters

# Space mesh
L = 2.0; # depth of sea ice
n = 400; # number of nodes
dx = L/n; # length between nodes
x = np.linspace(0.0,L,n+1);

# Time parameters
dt = 0.5; # time between iterations, in seconds
nt = 1500000; # amount of iterations
t_days = (dt*nt)/86400.0

# Calculate r, want ~0.25, must be < 0.5
r = ((alpha_ice)*(dt))/(2*dx*dx); # stability condition
print("The value of r is ", r)

#ND Parameters
diff_time_scale = (float(L**2))/(alpha_ice) #in seconds

#Create the matrices in the C_N scheme
#these do not change during the iterations

#%% CN Scheme set up

#create sprase matrices in "lil" format for easy manipulating
A = sparse.diags([-r, 1+2*r, -r], [-1, 0, 1], shape = (n+1,n+1),format='lil')
B = sparse.diags([r, 1-2*r, r], [-1, 0, 1], shape = (n+1,n+1),format='lil')

#now we pad the matrices with one in the TL and BR corners
A[0,[0,1]] = [1.0,0.0]
A[1,0] = -r
A[n,[n-1,n]] = [0.0,1.0]
A[n-1,n] = -r

B[0,[0,1]] = [1.0,0.0]
B[1,0] = r
B[n,[n-1,n]] = [0.0,1.0]
B[n-1,n] = r

#now convert to different format that is computation-friendly
A = A.tocsc()
B = B.tocsc()

#%% Initial and Boundary Conditions

#some inital profile
u = np.full(n+1, 273.15)

#force as column vector
u.shape = (len(u),1)

#set initial BC as well
T_it = float(u[0])
T_bt = float(u[-1])

#calculate initial RHS
rhs = B.dot(u)

#set initial conditions to the matrix as the first row
u_soln = u
minute_list = [0]

#%% Initial Fluxes - this section returns values of coefficients and fluxes

# First, a sanity check, to see if the individual fluxes makes sense
# We will calculate each part separately, using T_it and t=0
rad_net = sw_net(0.0)+lw_in*(1-eps_ice)-eps_ice*sigma*(T_it**4)
H_t = rho_a*u_star_top*c_pa*c_h*(air_temp(0.0) - T_it)
Ls_t = rho_a*u_star_top*Ls*c_e*(q_a - ice_q(float(u[0]))) #q_i was defined above for T_it
G_t = (1.0/dx)*kappa_ice*(float(u[1])-T_it)
print("\n-------Initial Fluxes-------")
print("\nradiation_net =",rad_net,"\nH_t =",H_t,"\nLs_t =",Ls_t,"\nG_t=",G_t)
print(f"sum of these terms={rad_net+H_t+Ls_t+G_t}")

# Now we look at the starting polynomial coefficients
a0 = sw_net(0.0)+lw_in*(1-eps_ice)+(rho_a*u_star_top)*((c_pa*c_h*air_temp(0.0)) \
            +(Ls*c_h*(q_a-ice_q(float(u[0])))))+(kappa_ice*(float(u[1]))/dx);
a1 = (-1.0*rho_a*c_pa*u_star_top*c_h)+((-1.0*kappa_ice)/dx);
a2 = 0;
a3 = 0;
a4 = (-1.0*sigma*eps_ice);

# calculate the initial root
def top_ice_flux_init(x):
    return a0 + a1*x + a4*(x**4)
root = optimize.newton(top_ice_flux_init, float(u[0]))
print("\n-------Initial Values-------")
print("\na0=",a0,"\na1=",a1,"\na2=a3=0","\na4=",a4)
print("The initial root is ",root)
print("\n----------------------------")

#%% Prepare plots and empty lists

#Create an empty list for outputs and plots
top_ice_temp_list = ["T_s"]
air_temp_list = ["T_a"]
sw_net_list = ["SW_net"]
lw_in_list = ["LW_in"]
lw_out_list = ["LW_out"]
rad_net_list = ["R_net"]
H_t_list = ["H_t"]
Ls_t_list = ["Ls_t"]
G_t_list = ["G_t"]
T5_list = ["T_5mm"]
H_b_list = ["H_b"]
G_b_list = ["G_b"]
Lf_b_list = ["Lf_b"]
top_flux_sum_list = ["top_flux_sum"]
bottom_flux_sum_list = ["bottom_flux_sum"]
mass_loss_bottom_list = ["mass_loss_bottom"]
mass_loss_top_list = ["mass_loss_top"]
thickness_loss_top_list = ["thickness_loss_top"]
thickness_loss_bottom_list = ["thickness_loss_bottom"]


#%% Start Iteration

for i in range(0,nt):
    
    # run through the CN scheme for interior points
    u = sparse.linalg.spsolve(A,rhs)
    
    # force to be column vector
    u.shape = (len(u),1)
    
    # append this array to solution file every 120 seconds
    if (i*dt)%120 == 0 and (i != 0):
        u_soln = np.append(u_soln, u, axis=1)
        minute_list.append((i*dt/60.0))
    
    # update values of the top EB polynomial (only a0 changes)    
    a0 = sw_net(i*dt)+lw_in*(1-eps_ice)+(rho_a*u_star_top)*((c_pa*c_h*air_temp(i*dt)) \
        +(Ls*c_h*(q_a-ice_q(float(u[0])))))+(kappa_ice*(float(u[1]))/dx);
    
    def top_ice_flux(x):
        return a0 + a1*x + a4*(x**4)
    
    # find the nearest root, this uses the secant method
    root = optimize.newton(top_ice_flux, float(u[0]))
   
    # set this root as the new BC for Tsoln
    u[0]=root
    
    # update rhs with new interior nodes
    rhs = B.dot(u)

    # print progress
    print(f"%={(i/nt)*100:.3f}, hr={(i*dt/3600)%24:.4f}, root={root:.4f}, value={top_ice_flux(root):.4f}")
    
    # Calculate individual terms on top from this iteration
    sw_net_value = sw_net(i*dt)
    lw_in_value = lw_in
    lw_out_value = -1.0*lw_in*eps_ice-(eps_ice*sigma*(float(u[0])**4))
    rad_net = sw_net_value+lw_in_value+lw_out_value
    H_t = rho_a*u_star_top*c_pa*c_h*(air_temp(i*dt) - float(u[0]))
    Ls_t = rho_a*u_star_top*Ls*c_e*(q_a - ice_q(float(u[0])))
    G_t = (1.0/dx)*kappa_ice*(float(u[1])-float(u[0]))
    top_flux_sum = rad_net + H_t + Ls_t + G_t

    # Calculate individual terms on bottom from this iteration
    H_b = rho_w*c_pw*u_star_bottom*c_h*(T_w-float(u[-1]))
    G_b = (1.0/dx)*kappa_ice*(float(u[-2])-float(u[-1]))
    Lf_b = -1*H_b - G_b
    bottom_flux_sum = H_b + G_b + Lf_b

    #Calculate mass loss rate
    mass_loss_top = Ls_t/Ls #sublimation on top
    mass_loss_bottom = Lf_b/Lf #melting on bottom
    
    # Now add the values to their respective lists
    sw_net_list.append(sw_net_value)
    lw_in_list.append(lw_in_value)
    lw_out_list.append(lw_out_value)
    rad_net_list.append(rad_net)
    H_t_list.append(H_t)
    Ls_t_list.append(Ls_t)
    G_t_list.append(G_t)
    T5_list.append(float(u[1]))
    air_temp_list.append(air_temp(i*dt))
    top_ice_temp_list.append(root)
    Lf_b_list.append(Lf_b)
    H_b_list.append(H_b)
    G_b_list.append(G_b)
    top_flux_sum_list.append(top_flux_sum)
    bottom_flux_sum_list.append(bottom_flux_sum)
    mass_loss_top_list.append(mass_loss_top)
    mass_loss_bottom_list.append(mass_loss_bottom)
    thickness_loss_top_list.append(mass_loss_top/rho_i)
    thickness_loss_bottom_list.append(mass_loss_bottom/rho_i)

#%% writing the solution to a file

# change minute list to 2Darray, then concatenate
minute_list = np.array(minute_list)
minute_list.shape = (len(minute_list),1)
u_soln = u_soln.transpose()
u_soln = np.concatenate((minute_list, u_soln), axis=1)

# now write the heat solution matrix to a file

np.savetxt(f"solutions/ice_solver_{n+1}nodes_{nt}tsteps.txt",
           u_soln,fmt='%.10f',delimiter=' ')

# create dimensional time array (in seconds, can convert later)
time_list = [nt * dt for nt in range(1,nt+1)]
time_list.insert(0,"t_dim")




# combine all other 1D temporal values in lists from above
master_fluxes_array = np.column_stack((time_list, sw_net_list, lw_in_list,
                                       lw_out_list, rad_net_list, H_t_list,
                                       Ls_t_list, G_t_list, T5_list,
                                       air_temp_list, top_ice_temp_list,
                                       Lf_b_list, H_b_list, G_b_list,
                                       top_flux_sum_list,
                                       bottom_flux_sum_list,
                                       mass_loss_bottom_list,
                                       mass_loss_top_list,
                                       thickness_loss_top_list,
                                       thickness_loss_bottom_list))
# save the matrix of components as CSV
np.savetxt(f"solutions/fluxes1D_{n+1}nodes_{nt}tsteps.csv",
           master_fluxes_array,
           fmt='%s',delimiter=',')
