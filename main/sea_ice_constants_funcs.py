# -*- coding: utf-8 -*-
"""
constants and functions
"""

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