#%% Ice Heat

# 1D Heat Equation solver for block of ice floating on seawater using a
# forward time centered space (FTCS) finite difference method
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import imageio
import os #shutil

# Constants in SI Units
alpha = 0.3; # albedo
eps_ice = 0.96; # emissivity
#sw_in = 0; # initial incoming shortwave at 0000 local time, W/m^2
#sw_net = (1-alpha)*sw_in; # net shortwave, W/m^2
sigma = 5.67*10**(-8); # Stefan-Boltzmann constant, W/(m^2K^4) 
lw_in = 200; # incoming longwave, W/m^2
rho_a = 1.225; # density of air, kg/m^3
rho_w  = 1020; # density of SEA water, kg/m^3
rho_i = 916.7; # density of ice, kg/m^3
V_a = 5; # velocity of air, m/s
u_star_top = 0.2; # friction velocity of atmosphere, m/s
V_w = 1; # velocity of water, m/s
u_star_bottom = 0.1; #friction velocity of water, m/s
#T_a = 278.15; # constant temperature of bulk air, K
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

#%% Initial and Boundary Conditions

# Space mesh
L = 2.0; # depth of sea ice
n = 400; # number of nodes
dx = L/n; # length between nodes
x = np.linspace(0.0,L,n+1);

#ND Parameters
diff_time_scale = (float(L**2))/(alpha_ice) #in seconds

# Create two boundary conditions (top and bottom of ice)
T_it = 272.65
T_ib = 273.15

#Create Solution Vector and initial profile
Tsoln = np.zeros(n+1)
Tsoln_pr = np.full(n+1, 272.65)
init = np.full(n+1, 272.65)

# Now we have a initial linear distribution of temperature in the sea ice
plt.plot(x,Tsoln_pr,"g-",label="Initial Profile")
plt.title("Initial Distribution of Temperature in Sea Ice")
plt.close()

# Time parameters
dt = 0.5; # time between iterations, in seconds
nt = 1000000; # amount of iterations
t_days = (dt*nt)/86400.0

# Calculate r, want ~0.25, must be < 0.5
r = ((alpha_ice)*(dt))/(dx*dx); # stability condition
print("The value of r is ", r)

#%% Initial Polynomial Values

# Calculate initial q from T (3 methods)

def ice_q(T):
    e = 611*np.exp((Ls/R_v)*((1/273)-(1/T)))
    return (eps_R*e)/(p_a+e*(eps_R-1))
q_i = ice_q(T_it)
#Alt method:
#def ice_q(T):
#    A_ice = 3.41*(10**12) #Pascal
#    B_ice = 6130 #Kelvin
#    e = A_ice*np.exp((-1.0*B_ice)/T)
#    return (eps_R*e)/(p_a+e*(eps_R-1))
#q_i = ice_q(T_it)
#Alt method2:
#def vap_pres_from_temp(T):
#    T_t = 273.16 #K
#    tau = 1 - (T/T_t)
#    a1 = -22.4948
#    a2 = -0.227
#    a3 = 0.502
#    a4 = 0.562
#    rhs = (a1*tau+a2*tau**2+a3*tau**3+a4*tau**4)*(T_t/T)
#    print(rhs)
#    e = 611.657*np.exp(rhs)
#    return (eps_R*e)/(p_a+e*(eps_R-1))

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

#%% Polynomial for Top of Ice - this section should return values of coefficients

# This is also a sanity check where we apply initial conditions,
# starting at 0000 local time in Barrow, Alaska. Currently the two functions of
# time we have are sw_net and air_temp

# First, a sanity check, to see if this polynomial makes sense
# We will calculate each part separately, using T_it and t=0

rad_net = sw_net(0.0)+lw_in*(1-eps_ice)-eps_ice*sigma*(T_it**4)
H_t = rho_a*u_star_top*c_pa*c_h*(air_temp(0.0) - T_it)
Ls_t = rho_a*u_star_top*Ls*c_e*(q_a - q_i) #q_i was defined above for T_it
G_t = (1.0/dx)*kappa_ice*(Tsoln_pr[1]-T_it)
print("\n-------Initial Fluxes-------")
print("\nradiation_net =",rad_net,"\nH_t =",H_t,"\nLs_t =",Ls_t,"\nG_t=",G_t)
print(f"sum of these terms={rad_net+H_t+Ls_t+G_t}")

# Now we look at the starting polynomial coefficients
a0 = sw_net(0.0)+lw_in*(1-eps_ice)+(rho_a*u_star_top)*((c_pa*c_h*air_temp(0.0))+(Ls*c_h*(q_a-q_i)))+(kappa_ice*(Tsoln[1])/dx);
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
root = optimize.newton(top_ice_flux_init, Tsoln[0])
print("\n-------Initial Values-------")
print("\na0=",a0,"\na1=",a1,"\na2=a3=0","\na4=",a4)
print("The initial root is ",root)
print("\n----------------------------")

#%% Start Iteration and prepare plots

#first, clear the folder with the images to make room for new ones
folder = "figures/giffiles"
filelist = [f for f in os.listdir(folder)]
for f in filelist:
    os.remove(os.path.join(folder, f))
print("\nGif images cleared!\n")


for i in range(0,nt):
    # Run through the FTCS with these BC
    for j in range(1,n):
        Tsoln[j] = Tsoln_pr[j] + r*(Tsoln_pr[j+1]-2*Tsoln_pr[j]+Tsoln_pr[j-1])
    
    # time in seconds to hours on a 24-hour clock will be used for radiation function
    
    #Now update values of the top EB polynomial (only a0 and a1 change)    
    a0 = sw_net(i*dt)+lw_in*(1-eps_ice)+(rho_a*u_star_top)*((c_pa*c_h*air_temp(i*dt))+(Ls*c_h*(q_a-ice_q(Tsoln[0]))))+(kappa_ice*(Tsoln[1])/dx);
    #a1 stays like this (it is all constants)
    #a1 = (-1.0*rho_a*c_pa*u_star_top*c_h)+(-1.0*kappa_ice/dx);
    def top_ice_flux(x):
        return a0 + a1*x + a4*(x**4)
    #Now find the nearest root, this uses the secant method
    root = optimize.newton(top_ice_flux, Tsoln[0])
    #residual = 
    print(f"i={i}/{nt}, hr={(i*dt/3600)%24:.4f}, root={root:.4f}, value={top_ice_flux(root):.4f}")
    
    #Now set this root as the new BC for Tsoln
    Tsoln[0]=root
    top_ice_temp_list.append(root)

    #Make sure the bottom BC is still 0 degrees C
    Tsoln[-1]=273.15
    
    # Calculate individual terms on top from this iteration
    sw_net_value = sw_net(i*dt)
    lw_in_value = lw_in
    lw_out_value = -1.0*lw_in*eps_ice-(eps_ice*sigma*(Tsoln[0]**4))
    rad_net = sw_net_value+lw_in_value+lw_out_value
    H_t = rho_a*u_star_top*c_pa*c_h*(air_temp(i*dt) - Tsoln[0])
    Ls_t = rho_a*u_star_top*Ls*c_e*(q_a - ice_q(Tsoln[0]))
    G_t = (1.0/dx)*kappa_ice*(Tsoln[1]-Tsoln[0])
    top_flux_sum = rad_net + H_t + Ls_t + G_t

    # Calculate individual terms on bottom from this iteration
    H_b = rho_w*c_pw*u_star_bottom*c_h*(T_w-T_ib)
    G_b = (1.0/dx)*kappa_ice*(Tsoln[-2]-T_ib)
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
    T5_list.append(Tsoln[1])
    air_temp_list.append(air_temp(i*dt))
    
    Lf_b_list.append(Lf_b)
    H_b_list.append(H_b)
    G_b_list.append(G_b)
    top_flux_sum_list.append(top_flux_sum)
    bottom_flux_sum_list.append(bottom_flux_sum)
    
    mass_loss_top_list.append(mass_loss_top)
    mass_loss_bottom_list.append(mass_loss_bottom)
    thickness_loss_top_list.append(mass_loss_top/rho_i)
    thickness_loss_bottom_list.append(mass_loss_bottom/rho_i)

    # Let's make a movie!    
    if (i*dt) <= 172800: #for less than two days
        if (i*dt)%120 == 0: #take a snapshot every 120 seconds
            title = str(int((i*dt)//60))
            plt.close()
            plt.plot(x,Tsoln,"k",label = f"{(i*dt/3600.0)%24:.2f} hours")
            plt.legend(loc=4)
            title1=f"Distribution of Temperature in Sea Ice after {t_days:.2f} days"
            plt.title(title1)
            plt.xlabel("x (m)")
            plt.ylabel("Temperature (K)")
            plt.tight_layout()
            plt.savefig("figures/giffiles/plot"+title+".png")
            plt.close()
    if (i*dt) > 172800: #after one day, not as frequent
        if (i*dt)%300 == 0: #every five minutes
            title = str(int((i*dt)//60))
            plt.close()
            plt.plot(x,Tsoln,"k",label = f"{(i*dt/3600.0)%24:.2f} hours")
            plt.legend(loc=4)
            title1=f"Distribution of Temperature in Sea Ice after {t_days:.2f} days"
            plt.title(title1)
            plt.xlabel("x (m)")
            plt.ylabel("Temperature (K)")
            plt.tight_layout()
            plt.savefig("figures/giffiles/plot"+title+".png")
            plt.close()

    #update Tsoln before next time step
    Tsoln_pr = Tsoln

#%% Movie Time

png_dir = 'figures/giffiles/'
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('figures/icemovie.gif',images)

#%% Plotting Main Results
locs, labels = plt.yticks()
    
# Plot the figure after nt iterations with initial profile
plt.plot(x,init,"g",label="Initial Profile")
plt.plot(x,Tsoln,"k",label=f"After {t_days:.2f} days")
title1=f"Distribution of Temperature in Sea Ice after {t_days:.2f} days"
plt.title(title1)
plt.xlabel("x (m)")
#plt.yticks(locs, map(lambda x: "%.1f" % x, locs*1e0))
plt.ylabel("Temperature (K)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/ice_temp_distribution.png")
plt.close()

#%% Some more output

locs, labels = plt.yticks()

#Create Time Array
time_list = dt*(np.array(list(range(1,nt+1)))) #in seconds, can convert later
time_hours = time_list/3600.0

#Plot time evolution of bottom fluxes
title_bottom = f"Time Evolution of Bottom Fluxes after {t_days:.2f} days"
plt.plot(time_hours,Lf_b_list,label="Latent Heat (f) Flux")
plt.plot(time_hours,G_b_list,label="Conductive Heat Flux")
plt.plot(time_hours,H_b_list,label="Sensible Heat Flux")
plt.plot(time_hours,bottom_flux_sum_list,label="Bottom Flux Sum")
plt.title(title_bottom)
plt.xlabel("Time (hr)")
#plt.yticks(locs, map(lambda x: "%.1f" % x, locs*1e0))
plt.ylabel('Flux (W m**-2)')
plt.legend()
plt.tight_layout()
plt.savefig("figures/bottom_fluxes_temporal.png")
plt.close()

#Plot time evolution of top radiation
title_top_rad=f"Time Evolution of Top Radiative Fluxes after {t_days:.2f} days"
plt.plot(time_hours,lw_in_list,label="Longwave In Flux")
plt.plot(time_hours,lw_out_list,label="Longwave Out Flux")
plt.plot(time_hours,sw_net_list,label="Shortwave Net Flux")
plt.plot(time_hours,rad_net_list,label="Total Net Radiative Flux")
plt.title(title_top_rad)
plt.ylabel('Flux (W m**-2)')
#plt.yticks(locs, map(lambda x: "%.1f" % x, locs*1e0))
plt.xlabel("Time (hr)")
#plt.axhline("0",linewidth=1, color="k")
plt.legend()
plt.tight_layout()
plt.savefig("figures/radiative_fluxes_temporal.png")
plt.close()

#Plot time evolution of top fluxes
title_top=f"Time Evolution of Top Fluxes after {t_days:.2f} days"
plt.plot(time_hours,rad_net_list,label="Radiation Net Flux")
plt.plot(time_hours,H_t_list,label="Sensible Heat Flux")
plt.plot(time_hours,Ls_t_list,label="Latent Heat (s) Flux")
plt.plot(time_hours,G_t_list,label="Conductive Heat Flux")
plt.plot(time_hours,top_flux_sum_list,label="Top Flux Sum")
plt.title(title_top)
plt.ylabel('Flux (W m**-2)')
#plt.yticks(locs, map(lambda x: "%.1f" % x, locs*1e0))
plt.xlabel("Time (hr)")
#plt.axhline("0",linewidth=1, color="k")
plt.legend()
plt.tight_layout()
plt.savefig("figures/top_fluxes_temporal.png")
plt.close()

#Plot time evoluation of mass loss (top and bottom)
title_mass=f"Mass Loss Rate after {t_days:.2f} days"
plt.title(title_mass)
color = 'tab:blue'
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Mass Loss Rate (Top) (kg s**-1 m**-2)",color=color)
ax1.plot(time_hours,mass_loss_top_list,color=color)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Mass Loss Rate (Bottom) (kg s**-1 m**-2)",color=color)
ax2.plot(time_hours,mass_loss_bottom_list,color=color)
ax2.tick_params(axis='y',labelcolor=color)
plt.tight_layout()
plt.savefig("figures/mass_loss_temporal.png")
print(f"Mass change on top: {sum(mass_loss_top_list)*dt:.3f}")
print(f"Mass change on bottom: {sum(mass_loss_bottom_list)*dt:.3f}")
plt.close()

#Plot time evoluation of thickness loss (top and bottom)
title_mass2=f"Thickness Loss Rate after {t_days:.2f} days"
plt.title(title_mass2)
color = 'tab:blue'
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Thickness Loss Rate (Top) (kg s**-1 m**-2)",color=color)
ax1.plot(time_hours,thickness_loss_top_list,color=color)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Thickness Loss Rate (Bottom) (kg s**-1 m**-2)",color=color)
ax2.plot(time_hours,thickness_loss_bottom_list,color=color)
ax2.tick_params(axis='y',labelcolor=color)
plt.tight_layout()
plt.savefig("figures/thickness_loss_temporal.png")
print(f"Thickness change on top: {(sum(thickness_loss_top_list)*dt):.6f}")
print(f"Thickness change on bottom: {(sum(thickness_loss_bottom_list)*dt):.6f}")
plt.close()

#Plot time evolution of surface temperature and air temperature
title_T_it=f"Surface and Air Temperature Evolution after {t_days:.2f} days"
plt.plot(time_hours,top_ice_temp_list,label="Top of Ice Surface Temperature")
plt.plot(time_hours,air_temp_list,label="Air Temperature")
plt.title(title_T_it)
plt.xlabel("Time (hr)")
#plt.yticks(locs, map(lambda x: "%.3f" % x, locs*1e0))
plt.ylabel('Temperature (K)')
plt.legend()
plt.tight_layout()
plt.savefig("figures/surface_and _air_temp_temporal.png")
plt.close()

#%% Components to see what is going wrong
#
##Plot Time Evoluation of G_t, TiTop, T5
#cblue = 'tab:blue'
#cred = 'tab:red'
#
#fig, ax1 = plt.subplots()
#ax1.set_xlabel("Time (hr)")
#ax1.set_ylabel("Temperature (K)",color=cblue)
#ax1.plot(time_hours,T5_list,"b--",label="$T_{i,5mm}$")
#ax1.plot(time_hours,top_ice_temp_list,"b",label="$T_{i,top}$")
#plt.legend(loc=1)
#ax2 = ax1.twinx()
#ax2.set_ylabel("Flux (W m**-2)",color=cred)
#ax2.plot(time_hours,G_t_list,color=cred,label="$G_t$")
#ax2.tick_params(axis='y')
#plt.title(f"Components of $G_t$ after {t_days:.2f} days")
#plt.legend(loc=3)
#plt.tight_layout()
#plt.savefig("figures/components_Gt.png")
#plt.close()
#
##Plot Time Evolution of H, Ta, TiTop
#fig, ax1 = plt.subplots()
#ax1.set_xlabel("Time (hr)")
#ax1.set_ylabel("Temperature (K)",color=cblue)
#ax1.plot(time_hours,air_temp_list,"b--",label="$T_{a}$")
#ax1.plot(time_hours,top_ice_temp_list,"b",label="$T_{i,top}$")
#plt.legend(loc=1)
#ax2 = ax1.twinx()
#ax2.set_ylabel("Flux (W m**-2)",color=cred)
#ax2.plot(time_hours,H_t_list,color=cred,label="$H_t$")
#ax2.tick_params(axis='y')
#plt.title(f"Components of $H_t$ after {t_days:.2f} days")
#plt.legend(loc=3)
#plt.tight_layout()
#plt.savefig("figures/components_Ht.png")
#plt.close()
#
##Plot time evolution of Lst, qa, qitop
#fig, ax1 = plt.subplots()
#ax1.set_xlabel("Time (hr)")
#ax1.set_ylabel("Specific Humidity (kg/kg)",color=cblue)
#ax1.plot(time_hours,air_temp_list,"b",label="$q_{a}$")
#plt.legend(loc=1)
#ax2 = ax1.twinx()
#ax2.set_ylabel("Flux (W m**-2)",color=cred)
#ax2.plot(time_hours,Ls_t_list,color=cred,label="$L_{s,t}$")
#ax2.tick_params(axis='y')
#plt.title("Components of $L_{s,t}$ after"+f" {t_days:.2f} days")
#plt.legend(loc=3)
#plt.tight_layout()
#plt.savefig("figures/components_Lst.png")
#plt.close()

#%% Now again, clear up the folders to save storage space

import moviepy.editor as mp
clip = mp.VideoFileClip("figures/icemovie.gif")
clip.write_videofile("figures/icemovie.mp4")

folder = "figures/giffiles"
os.chmod(folder, 0o777)
filelist = [f for f in os.listdir(folder)]
for f in filelist:
    os.remove(os.path.join(folder, f))
os.remove("figures/icemovie.gif")

print("\nGif images and gif removed!")