# constants for the sea ice solver

from numpy import linspace

class cnst(object):
    
    ##### Constants #####
    
    # radiation
    alpha = 0.6; # albedo
    eps_ice = 0.96; # emissivity
    sigma = 5.67*10**(-8); # Stefan-Boltzmann constant, W/(m^2K^4) 
    lw_in = 200; # incoming longwave, W/m^2
    
    # densities
    rho_a = 1.225; # density of air, kg/m^3
    rho_w  = 1020; # density of SEA water, kg/m^3
    rho_i = 916.7; # density of ice, kg/m^3
    
    # dynamics
    V_a = 5; # velocity of air, m/s
    u_star_top = 0.2; # friction velocity of atmosphere, m/s
    V_w = 1; # velocity of water, m/s
    u_star_bottom = 0.1; #friction velocity of water, m/s
    c_h = 0.01737; # bulk transfer coefficient for latent heat
    c_e = 0.01737; # bulk transer coefficient for sensible heat
    
    # thermodynamics
    T_w = 272.15; # temperature of bulk water, K
    c_pa = 1004; # specific heat of air, J/(kgK)
    c_pw = 3985; # specific heat of SEA water, J/(kgK)
    c_pi = 2027; # specific heat of ice, J/(kgK)
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
    
    ##### Numerical Parameters #####
    
    # spatial mesh
    L = 2.0; # depth of sea ice
    n = 400; # number of nodes
    dx = L/n; # length between nodes
    x = linspace(0.0,L,n+1);
    
    # temporal parameters
    dt = 0.5; # time between iterations, in seconds
    nt = 50000; # amount of iterations
    total_t = nt*dt # dimensional seconds that the simulation runs
    total_t_days = (dt*nt)/86400.0 # dimensional days
    p_count = 120; # t_steps to write to file
    
    diff_time_scale = (float(L**2))/(alpha_ice) #in seconds
    
    