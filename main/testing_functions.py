#%% Ice Heat

# 1D Heat Equation solver for block of ice floating on seawater using a
# Crank-Nicolson method
import numpy as np
import matplotlib.pyplot as plt


import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

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
    e = 611*np.exp((Ls/R_v)*((1/273)-(1/T)))
    return (eps_R*e)/(p_a+e*(eps_R-1))

def ice_q2(T):
    Ai = 3.41*10**12
    Bi = 6130
    ei=Ai*np.exp(-1.0*Bi/T)
    return (eps_R*ei)/(p_a+ei*(eps_R-1))

#def ice_q3(T):
    

x = np.linspace(245,270)
y1 = ice_q(x)
y2 = ice_q2(x)
#plt.plot(x,y1)
plt.plot(x,y2)
plt.show()
