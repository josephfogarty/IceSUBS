# A code to convert surface file templates to T, zo, and q surface files
# to be used in the LES code

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

# Colormap properties for ice and water
cmap = ListedColormap(['xkcd:sky blue', 'xkcd:dark blue'])

# To convert temp of ice to specific humidity
def ice_q(T):
    
    # Atmospheric properties
    Ls = 2834000.0 # latent heat of sublimation, J/kg
    R_v = 461.5 # water vapor gas constant, J/(kgK)
    eps_R = 0.6218851571 # ratio of gas constants
    p_a = 101325.0 # air pressure, Pa
    
    # Calculate vapor pressure and specific humidity over ice
    e = 611*np.exp((Ls/R_v)*((1/273)-(1/T)))
    q = (eps_R*e)/(p_a+e*(eps_R-1))
    return q

# To convert temp of water/pond to specific humidity
def liquid_q(T):
    
    # Atmospheric properties
    Ls = 2834000.0 # latent heat of sublimation, J/kg
    R_v = 461.5 # water vapor gas constant, J/(kgK)
    eps_R = 0.6218851571 # ratio of gas constants
    p_a = 101325.0 # air pressure, Pa
    
    # Calculate vapor pressure and specific humidity over ice
    e = 611*np.exp((Ls/R_v)*((1/273)-(1/T)))
    q = (eps_R*e)/(p_a+e*(eps_R-1))
    return q

#%% Defining parameters

# Define rougnesses corresponding to surface type
zo_ice = 0.0002
zo_sea = 0.002
zo_pond = 0.001

# Define temperatures corresponding to surface type
T_ice = 270.15 #Kelvin, -3 degrees C
T_sea = 274.15 #Kelvin, 1 degrees C
T_pond = 275.15 # Kelvin, 2 degrees C

# Calculate specific humdity based n these temperatures above
q_ice = ice_q(T_ice)
q_sea = liquid_q(T_sea)
q_pond = liquid_q(T_pond)

#%% Loading Data

# Path of template matrix 
filename = "checker2.dat"
lp = os.path.join("surfacefiles", filename)

# Path to save the matrix
sp = os.path.join("LES_ready")

# Load the template matrix
master_sfc_template = np.loadtxt(lp)
plt.matshow(master_sfc_template,cmap=cmap)

#%% Filling in Data

# Create the temp matrix
temp_sfc = np.loadtxt(lp)
temp_sfc[temp_sfc == 1] = T_ice
temp_sfc[temp_sfc == 2] = T_sea
temp_sfc[temp_sfc == 3] = T_pond
print(np.unique(temp_sfc))

# Create roughness matrix
zo_sfc = np.loadtxt(lp)
zo_sfc[zo_sfc == 1] = zo_ice
zo_sfc[zo_sfc == 2] = zo_sea
zo_sfc[zo_sfc == 3] = zo_pond
print(np.unique(zo_sfc))

# Create humidity matrix
q_sfc = np.loadtxt(lp)
q_sfc[q_sfc == 1] = q_ice
q_sfc[q_sfc == 2] = q_sea
q_sfc[q_sfc == 3] = q_pond
print(np.unique(q_sfc))

# Save the matrices
np.savetxt(os.path.join(sp, "temp_remote.dat"), temp_sfc, delimiter=' ',fmt='%E')
np.savetxt(os.path.join(sp, "zo_remote.dat"), zo_sfc, delimiter=' ',fmt='%E')
np.savetxt(os.path.join(sp, "q_remote.dat"), q_sfc, delimiter=' ',fmt='%E')
