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
load_path = os.path.join("surfacefiles", filename)

# Path to save the matrix
save_path = os.path.join("LES_ready")

# Load the template matrix
master_sfc_template = np.loadtxt(load_path)
plt.matshow(master_sfc_template,cmap=cmap)

#%% Filling in Data

# Create the three matrices
temp_sfc_matrix = master_sfc_template
temp_sfc_matrix[temp_sfc_matrix == 1] = T_ice
temp_sfc_matrix[temp_sfc_matrix == 2] = T_sea
temp_sfc_matrix[temp_sfc_matrix == 3] = T_pond
print(np.unique(temp_sfc_matrix))

# Create temperature matrix
zo_sfc_matrix = master_sfc_template
zo_sfc_matrix[zo_sfc_matrix == 1] = zo_ice
zo_sfc_matrix[zo_sfc_matrix == 2] = zo_sea
zo_sfc_matrix[zo_sfc_matrix == 3] = zo_pond
print(np.unique(zo_sfc_matrix))

# Create humidity matrix
q_sfc_matrix = master_sfc_template
q_sfc_matrix[q_sfc_matrix == 1] = q_ice
q_sfc_matrix[q_sfc_matrix == 2] = q_sea
q_sfc_matrix[q_sfc_matrix == 3] = q_pond
print(np.unique(q_sfc_matrix))

# Save the matrices
np.savetxt(os.path.join(save_path, "temp_remote.dat"), temp_sfc_matrix, delimiter=' ',fmt='%1.3f')
np.savetxt(os.path.join(save_path, "zo_remote.dat"), zo_sfc_matrix, delimiter=' ',fmt='%1.3f')
np.savetxt(os.path.join(save_path, "q_remote.dat"), q_sfc_matrix, delimiter=' ',fmt='%1.3f')
