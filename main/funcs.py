import numpy as np
from constants import cnst

#%% the functions

def ice_q(T):
    
    """
    A function to calculate specific humidity q from temperature T 
    over a block of ice, using the Clausius-Clapeyron relationship
    """
    
    # calculate vapor pressure "e" via CC
    e = 611*np.exp((cnst.Ls/cnst.R_v)*((1.0/273.0)-(1.0/T)))
    
    # calculate specific humidity "q" assuming ideal gas
    q = (cnst.eps_R*e)/(cnst.p_a+e*(cnst.eps_R-1))
    return q


def sw_net(t):
    
    """
    A function to calculate the net short wave, starting at midnight, from
    time after midnight (where t is in seconds)
    """
    
    # convert to hours
    t_hours = (t/3600.0)%24
    
    # nighttime
    if (t_hours) < 7 or (t_hours) > 22:
        #print(f"hour = {(i*dt/(3600.0))%24}, dark")
        sw_in = 0.0
        
    # during the day, follow a half-sinusoidal profile
    else:
        sw_in = 500*np.sin((np.pi/15.0)*(t_hours-7.0))
        #print(f"hour = {(i*dt/(3600.0))%24}, light")
        #print(f"sw_in = {sw_in}")
    
    # account for albdeo
    shortwave_net = (1-cnst.alpha)*sw_in; # net shortwave, W/m^2 
    return shortwave_net


# Define function to calculate temperature based on time of day
def air_temp(t): #t is in seconds, so dt*i would be evaluated
    
    """
    A function to calculate the bulk air temperature, starting at midnight, 
    from time after midnight (where t is in seconds), based on a sine curve
    """
    
    # convert to hours
    t_hours = (t/3600.0)%24
    
    # calculate temp
    temp = 7.0*np.sin((np.pi/12.0)*(t_hours-13.0))+268
    return temp

