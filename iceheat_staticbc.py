#%% Ice Heat

# 1D Heat Equation solver for block of ice floating on seawater using a
# forward time centered space (FTCS) finite difference method
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import numpy.polynomial.polynomial as poly

# Constants in SI Units
alpha = 0.3; # albedo
eps_ice = 0.96; # emissivity
sw_in = 500; # incoming shortwave, W/m^2
sw_net = (1-alpha)*sw_in; # net shortwave, W/m^2
sigma = 5.67*10**(-8); # Stefan-Boltzmann constant, W/(m^2K^4) 
lw_in = 350; # incoming longwave, W/m^2
rho_a = 1.225; # density of air, kg/m^3
rho_w  = 1020; # density of SEA water, kg/m^3
rho_i = 916.7; # density of ice, kg/m^3
V_a = 5; # velocity of air, m/s
V_w = 1; # velocity of water, m/s
T_a = 278.15; # temperature of bulk air, K
T_w = 272.15; # temperature of bulk water, K
c_pa = 1004; # specific heat of air, J/(kgK)
c_pw = 3985; # specific heat of SEA water, J/(kgK)
c_pi = 2027; # specific heat of ice, J/(kgK)
c_h = 0.01737; # bulk transfer coefficient for latent heat
c_e = 0.01737; # bulk transer coefficient for sensible heat
L_v = 2500000; # latent heat of vaporization, J/kg
L_f = 334000; # latent heat of fusion, J/kg
L_s = L_v + L_f; # latent heat of sublimation, J/kg
p_a = 101325; # pressure of air,  Pa
R_d = 287.0; # dry air constant, J/(kgK)
kappa_ice = 2.25; # thermal conductivity of ice, W/(mK)
q_a = 1.8; # specific humidity of air, g/kg
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
T_it = 275.15
T_ib = 273.15

#Create Solution Vector
Tsoln = np.zeros(n+1)
Tsoln_pr = np.full(n+1, 272.65)

# Now we have a initial linear distribution of temperature in the sea ice
plt.plot(x,Tsoln_pr,"g-",label="Initial Profile")
plt.title("Initial Distribution of Temperature in Sea Ice")

# Time parameters
dt = 0.5; # time between iterations
nt = 50000; # amount of iterations

# Calculate r, want ~0.25
r = ((alpha_ice)*(dt))/(dx*dx); # stability condition
print("The value of r is ", r)

#%% Start Iteration
for i in range(0,nt):
    # Run through the FTCS with these BC
    for j in range(1,n):
        Tsoln[j] = Tsoln_pr[j] + r*(Tsoln_pr[j+1]-2*Tsoln_pr[j]+Tsoln_pr[j-1])
    
    # steady BC, we impose them
    Tsoln[0] = 276.15
    Tsoln[-1] = 272.15
    #update Tsoln before next time step
    Tsoln_pr = Tsoln

# Plot the figure after nt iterations
label_string=f"After {int(nt)} iterations"
plt.plot(x,Tsoln,label=label_string)
title1=f"Distribution of Temperature in Sea Ice after {int(nt*dt)} seconds"
plt.title(title1)
plt.legend()


