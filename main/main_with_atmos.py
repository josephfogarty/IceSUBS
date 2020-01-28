"""
1D Heat Equation solver for block of ice floating on seawater using the
Crank-Nicolson method
"""
import numpy as np
from scipy import optimize
from scipy import sparse
#from scipy.sparse import linalg, diags
from constants import cnst
import funcs as fn

# label some constants for ease of reading below
lw_in = cnst.lw_in
eps_ice = cnst.eps_ice
sigma = cnst.sigma
rho_a = cnst.rho_a
rho_i = cnst.rho_i
rho_w = cnst.rho_w
kappa_ice = cnst.kappa_ice
q_a = cnst.q_a
c_e = cnst.c_e
c_h = cnst.c_h
u_star_top = cnst.u_star_top
c_pa = cnst.c_pa
Ls = cnst.Ls


#%% CN Scheme set up

# Calculate r, want ~0.25, must be < 0.5
r = ((cnst.alpha_ice)*(cnst.dt))/(2*cnst.dx*cnst.dx); # stability condition
print("The value of r is ", r)

#create sprase matrices in "lil" format for easy manipulating
A = sparse.diags([-r, 1+2*r, -r], [-1, 0, 1], shape = (cnst.n+1,cnst.n+1),format='lil')
B = sparse.diags([r, 1-2*r, r], [-1, 0, 1], shape = (cnst.n+1,cnst.n+1),format='lil')

#now we pad the matrices with one in the TL and BR corners
A[0,[0,1]] = [1.0,0.0]
A[1,0] = -r
A[cnst.n,[cnst.n-1,cnst.n]] = [0.0,1.0]
A[cnst.n-1,cnst.n] = -r

B[0,[0,1]] = [1.0,0.0]
B[1,0] = r
B[cnst.n,[cnst.n-1,cnst.n]] = [0.0,1.0]
B[cnst.n-1,cnst.n] = r

#now convert to different format that is computation-friendly
A = A.tocsc()
B = B.tocsc()

#%% Initial and Boundary Conditions

#some inital profile for ice block
u_ice = np.full(cnst.n+1, 273.15)

#force as column vector
u_ice.shape = (len(u_ice),1)

#set initial BC as well
T_it = float(u_ice[0])
T_bt = float(u_ice[-1])

#calculate initial RHS
rhs = B.dot(u_ice)

#set initial conditions to the matrix as the first row
u_ice_soln = u_ice
# create a list for how many times the solution was written
write_list = [0]

#%% Initial Fluxes - this section returns values of coefficients and fluxes

# First, a sanity check, to see if the individual fluxes makes sense
# We will calculate each part separately, using T_it and t=0
rad_net = fn.sw_net(0.0)+lw_in*(1.0-eps_ice)-eps_ice*sigma*(T_it**4)

H_t = rho_a*u_star_top*c_pa*c_h*(fn.air_temp(0.0) - T_it)

Ls_t = rho_a*u_star_top*Ls*c_e*(q_a - fn.ice_q(float(u_ice[0])))

G_t = (1.0/cnst.dx)*kappa_ice*(float(u_ice[1])-T_it)

print("\n-------Initial Fluxes-------",
      "\nradiation_net =",rad_net,"\nH_t =",H_t,"\nLs_t =",Ls_t,"\nG_t=",G_t,
      f"sum of these terms={rad_net+H_t+Ls_t+G_t}")

# Now we look at the starting polynomial coefficients
a0 = fn.sw_net(0.0)+lw_in*(1-cnst.eps_ice)+(rho_a*u_star_top)*((c_pa*c_h*fn.air_temp(0.0)) \
            +(Ls*c_h*(q_a-fn.ice_q(float(u_ice[0])))))+(kappa_ice*(float(u_ice[1]))/cnst.dx);
a1 = (-1.0*rho_a*c_pa*u_star_top*c_h)+((-1.0*kappa_ice)/cnst.dx);
a2 = 0;
a3 = 0;
a4 = (-1.0*sigma*eps_ice);

# calculate the initial root
def top_ice_flux_init(x):
    return a0 + a1*x + a4*(x**4)
root = optimize.newton(top_ice_flux_init, float(u_ice[0]))
print("\n------Initial Values------","\na0=",a0,"\na1=",a1,"\na2=a3=0",
      "\na4=",a4,"\nThe initial root is ",root,"\n--------------------------")

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

for i in range(0,cnst.nt):
    
    # tstep in seconds
    ts = i*cnst.dt
    
    # run through the CN scheme for interior points
    u = sparse.linalg.spsolve(A,rhs)
    
    # force to be column vector
    u.shape = (len(u),1)
    
    # append this array to solution file every pcount
    if (ts)%(cnst.p_count) == 0 and (i != 0):
        u_ice_soln = np.append(u_ice_soln, u, axis=1)
        write_list.append(ts)
    
    # update values of the top EB polynomial (only a0 changes)
    a0 = fn.sw_net(ts)+lw_in*(1-eps_ice)+(rho_a*u_star_top)*((c_pa*c_h*fn.air_temp(ts)) \
        +(Ls*c_h*(q_a-fn.ice_q(float(u[0])))))+(kappa_ice*(float(u[1]))/cnst.dx);
    
    def top_ice_flux(x):
        return a0 + a1*x + a4*(x**4)
    
    # find the nearest root, this uses the secant method
    root = optimize.newton(top_ice_flux, float(u[0]))
   
    # set this root as the new BC for Tsoln
    u[0]=root
    
    # update rhs with new interior nodes
    rhs = B.dot(u)

    # print progress
    print(f"%={(i/cnst.nt)*100:.3f}, hr={(ts/3600)%24:.4f}, root={root:.4f}, value={top_ice_flux(root):.4f}")
    
    # Calculate individual terms on top from this iteration
    sw_net_value = fn.sw_net(ts)
    lw_in_value = cnst.lw_in
    lw_out_value = -1.0*cnst.lw_in*cnst.eps_ice-(cnst.eps_ice*cnst.sigma*(float(u[0])**4))
    rad_net = sw_net_value+lw_in_value+lw_out_value
    H_t = cnst.rho_a*cnst.u_star_top*cnst.c_pa*cnst.c_h*(fn.air_temp(i*cnst.dt) - float(u[0]))
    Ls_t = cnst.rho_a*cnst.u_star_top*cnst.Ls*cnst.c_e*(cnst.q_a - fn.ice_q(float(u[0])))
    G_t = (1.0/cnst.dx)*cnst.kappa_ice*(float(u[1])-float(u[0]))
    top_flux_sum = rad_net + H_t + Ls_t + G_t

    # Calculate individual terms on bottom from this iteration
    H_b = cnst.rho_w*cnst.c_pw*cnst.u_star_bottom*cnst.c_h*(cnst.T_w-float(u[-1]))
    G_b = (1.0/cnst.dx)*cnst.kappa_ice*(float(u[-2])-float(u[-1]))
    Lf_b = -1.0*H_b - G_b
    bottom_flux_sum = H_b + G_b + Lf_b

    #Calculate mass loss rate
    mass_loss_top = Ls_t/cnst.Ls #sublimation on top
    mass_loss_bottom = Lf_b/cnst.Lf #melting on bottom
    
    # Now add the values to their respective lists
    sw_net_list.append(sw_net_value)
    lw_in_list.append(lw_in_value)
    lw_out_list.append(lw_out_value)
    rad_net_list.append(rad_net)
    H_t_list.append(H_t)
    Ls_t_list.append(Ls_t)
    G_t_list.append(G_t)
    T5_list.append(float(u[1]))
    air_temp_list.append(fn.air_temp(i*cnst.dt))
    top_ice_temp_list.append(root)
    Lf_b_list.append(Lf_b)
    H_b_list.append(H_b)
    G_b_list.append(G_b)
    top_flux_sum_list.append(top_flux_sum)
    bottom_flux_sum_list.append(bottom_flux_sum)
    mass_loss_top_list.append(mass_loss_top)
    mass_loss_bottom_list.append(mass_loss_bottom)
    thickness_loss_top_list.append(mass_loss_top/cnst.rho_i)
    thickness_loss_bottom_list.append(mass_loss_bottom/cnst.rho_i)

#%% writing the solution to a file

# change minute list to 2Darray, then concatenate
write_list2 = np.array(np.arange(0,cnst.total_t,cnst.p_count))
write_list2.shape = (len(write_list2),1)
u_ice_soln = u_ice_soln.transpose()
u_ice_soln = np.concatenate((write_list2, u_ice_soln), axis=1)

# now write the heat solution matrix to a file
np.savetxt(f"solutions/ice_solver_{cnst.n+1}nodes_{cnst.nt}tsteps.txt",
           u_ice_soln,fmt='%.10f',delimiter=' ')

# create dimensional time array (in seconds, can convert later)
# create time array because it is hard to do in constants file
t_list = [cnst.nt*cnst.dt for cnst.nt in range(0,cnst.nt)]
t_list.insert(0,"t_dim")

# combine all other 1D temporal values in lists from above
master_fluxes_array = np.column_stack((t_list, sw_net_list, lw_in_list,
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
np.savetxt(f"solutions/fluxes1D_{cnst.n+1}nodes_{cnst.nt}tsteps.csv",
           master_fluxes_array,
           fmt='%s',delimiter=',')
