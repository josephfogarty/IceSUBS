#%% Ice Heat

# 1D Heat Equation solver for block of ice floating on seawater using a
# forward time centered space (FTCS) finite difference method
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import sparse
from scipy.sparse import linalg, diags

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

#%% Constants in SI Units
alpha = 0.3; # albedo
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
    e = 611*np.exp((Ls/R_v)*((1/273)-(1/T)))
    return (eps_R*e)/(p_a+e*(eps_R-1))


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

# Define function to calculate temperature based on time
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
nt = 1000000; # amount of iterations
t_days = (dt*nt)/86400.0

# Calculate r, want ~0.25, must be < 0.5
r = ((alpha_ice)*(dt))/(2*dx*dx); # stability condition
print("The value of r is ", r)

#ND Parameters
diff_time_scale = (float(L**2))/(alpha_ice) #in seconds

#Create the matrices in the C_N scheme
#these do not change during the iterations

#%% CN Scheme set up

#create sprase matrices
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

##Plot this to see what it looks like
#p = np.poly1d([a4, a3, a2, a1, a0])
#x_test = np.linspace(0,1000,1001)
#y_test = p(x_test)
#plt.plot(x_test, y_test)
#plt.title("Initial Curve of Top of Ice Polynomial")
#plt.xlabel("Temperature (K)")
#plt.ylabel("Flux (W m**-2)")
#plt.axhline("0",linewidth=1, color="k")
#plt.show()

#Let's calculate the initial root
def top_ice_flux_init(x):
    return a0 + a1*x + a4*(x**4)
root = optimize.newton(top_ice_flux_init, float(u[0]))
print("\n-------Initial Values-------")
print("\na0=",a0,"\na1=",a1,"\na2=a3=0","\na4=",a4)
print("The initial root is ",root)
print("\n----------------------------")

#%% Prepare plots anmd empty lists

#Create an empty list for outputs and plots
top_ice_temp_list = []
air_temp_list = []
sw_net_list = []
lw_in_list = []
lw_out_list = []
rad_net_list = []
H_t_list = []
Ls_t_list = []
G_t_list = []
T5_list = []
H_b_list = []
G_b_list = []
Lf_b_list = []
top_flux_sum_list = []
bottom_flux_sum_list = []
mass_loss_bottom_list = []
mass_loss_top_list = []
thickness_loss_top_list = []
thickness_loss_bottom_list = []

#%% Start Iteration
for i in range(0,nt):
    
    # run through the CN scheme for interior points
    u = sparse.linalg.spsolve(A,rhs)
    
    # force to be column vector
    u.shape = (len(u),1)
    
    # append this array to solution file
    if (i*dt)%120 == 0: #every 60 seconds
        u_soln = np.append(u_soln, u, axis=1)    
    
    # update values of the top EB polynomial (only a0 and a1 change)    
    a0 = sw_net(i*dt)+lw_in*(1-eps_ice)+(rho_a*u_star_top)*((c_pa*c_h*air_temp(i*dt)) \
                +(Ls*c_h*(q_a-ice_q(float(u[0])))))+(kappa_ice*(float(u[1]))/dx);
    def top_ice_flux(x):
        return a0 + a1*x + a4*(x**4)
    
    # find the nearest root, this uses the secant method
    root = optimize.newton(top_ice_flux, float(u[0]))
   
    # set this root as the new BC for Tsoln
    u[0]=root
    
    #update rhs with new interior nodes
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

# write the solution matrix to a file
u_soln = u_soln.transpose()
np.savetxt(f"solutions/ice_solver_{n+1}nodes.txt",u_soln, fmt = '%.10f',delimiter=' ')

#%% Some temporal output

#colors
bluecol = 'tab:blue'
redcol = 'tab:red'

#Create Time Array
time_list = dt*(np.array(list(range(1,nt+1)))) #in seconds, can convert later
time_hours = time_list/3600.0

#Bottom Fluxes
title_bottom = f"Time Evolution of Bottom Fluxes after {t_days:.2f} days"
plt.plot(time_hours,Lf_b_list,label="Latent Heat (f) Flux")
plt.plot(time_hours,G_b_list,label="Conductive Heat Flux")
plt.plot(time_hours,H_b_list,label="Sensible Heat Flux")
plt.plot(time_hours,bottom_flux_sum_list,label="Bottom Flux Sum")
plt.title(title_bottom)
plt.xlabel("Time (hr)")
plt.ylabel('Flux (W m**-2)')
plt.grid()
plt.legend(prop={'size':20})
#plt.tight_layout()
plt.savefig("figures/bottom_fluxes_temporal.png")
plt.close()

#Top fluxes
title_top=f"Time Evolution of Top Fluxes after {t_days:.2f} days"
plt.plot(time_hours,rad_net_list,label="Radiation Net Flux")
plt.plot(time_hours,H_t_list,label="Sensible Heat Flux")
plt.plot(time_hours,Ls_t_list,label="Latent Heat (s) Flux")
plt.plot(time_hours,G_t_list,label="Conductive Heat Flux")
plt.plot(time_hours,top_flux_sum_list,label="Top Flux Sum")
plt.title(title_top)
plt.ylabel('Flux (W m**-2)')
plt.xlabel("Time (hr)")
plt.legend()
plt.xlim(left=0)
plt.grid()
plt.tight_layout()
plt.savefig("figures/top_fluxes_temporal.png")
plt.close()

#Radiation budget
title_top_rad=f"Time Evolution of Top Radiative Fluxes after {t_days:.2f} days"
plt.plot(time_hours,lw_in_list,label="Longwave In Flux")
plt.plot(time_hours,lw_out_list,label="Longwave Out Flux")
plt.plot(time_hours,sw_net_list,label="Shortwave Net Flux")
plt.plot(time_hours,rad_net_list,label="Total Net Radiative Flux")
plt.title(title_top_rad)
plt.ylabel('Flux (W m**-2)')
plt.xlabel("Time (hr)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("figures/radiative_fluxes_temporal.png")
plt.close()

#Plot time evoluation of mass loss (top and bottom)
title_mass=f"Mass Loss Rate after {t_days:.2f} days"
plt.title(title_mass)
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Mass Loss Rate (Top) (kg s**-1 m**-2)",color=bluecol)
ax1.plot(time_hours,mass_loss_top_list,color=bluecol)
ax2 = ax1.twinx()
ax2.set_ylabel("Mass Loss Rate (Bottom) (kg s**-1 m**-2)",color=redcol)
ax2.plot(time_hours,mass_loss_bottom_list,color=redcol)
ax2.tick_params(axis='y')
plt.tight_layout()
plt.grid()
plt.savefig("figures/mass_loss_temporal.png")
print(f"Mass change on top: {sum(mass_loss_top_list)*dt:.3f}")
print(f"Mass change on bottom: {sum(mass_loss_bottom_list)*dt:.3f}")
plt.close()

#Thickness loss (top and bottom)
title_mass2=f"Thickness Loss Rate after {t_days:.2f} days"
plt.title(title_mass2)
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Thickness Loss Rate (Top) (kg s**-1 m**-2)",color=bluecol)
ax1.plot(time_hours,thickness_loss_top_list,color=bluecol)
ax2 = ax1.twinx()
ax2.set_ylabel("Thickness Loss Rate (Bottom) (kg s**-1 m**-2)",color=redcol)
ax2.plot(time_hours,thickness_loss_bottom_list,color=redcol)
ax2.tick_params(axis='y')
plt.tight_layout()
plt.grid()
plt.savefig("figures/thickness_loss_temporal.png")
print(f"Thickness change on top: {(sum(thickness_loss_top_list)*dt):.6f}")
print(f"Thickness change on bottom: {(sum(thickness_loss_bottom_list)*dt):.6f}")
plt.close()

#Surface temp and air temp comparison
title_T_it=f"Surface and Air Temperature Evolution after {t_days:.2f} days"
plt.plot(time_hours,top_ice_temp_list,label="Top of Ice Surface Temperature")
plt.plot(time_hours,air_temp_list,label="Air Temperature")
plt.title(title_T_it)
plt.xlabel("Time (hr)")
plt.ylabel('Temperature (K)')
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("figures/surface_and _air_temp_temporal.png")
plt.close()