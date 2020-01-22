# A code to convert surface file templates to T, zo, and q surface files
# to be used in the LES code

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import os

# Colormap properties for ice, water, and pond
cmap = ListedColormap(['xkcd:off white', 'xkcd:midnight blue', 'xkcd:cyan'])
bounds = [0.5,1.5,2.5,3.5]
norm = BoundaryNorm(bounds,cmap.N)

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
    Lv = 2260000.0 # latent heat of evaporation, J/kg
    R_v = 461.5 # water vapor gas constant, J/(kgK)
    eps_R = 0.6218851571 # ratio of gas constants
    p_a = 101325.0 # air pressure, Pa
    
    # Calculate vapor pressure and specific humidity over ice
    e = 611*np.exp((Lv/R_v)*((1/273)-(1/T)))
    q = (eps_R*e)/(p_a+e*(eps_R-1))
    return q

# Define rougnesses corresponding to surface type
zo_ice = 0.0002
zo_sea = 0.002
zo_pond = 0.001

# Define temperatures corresponding to surface type
T_ice = 274.15 #Kelvin, -3 degrees C
T_sea = 274.15 #Kelvin, 1 degrees C
T_pond = 275.15 # Kelvin, 2 degrees C

# Calculate specific humdity based n these temperatures above
q_ice = ice_q(T_ice)
q_sea = liquid_q(T_sea)
q_pond = liquid_q(T_pond)

# DONT TOUCH ANYTHING ABOVE HERE
# unless you are changing sea ice/ocean parameters
#%% Loading Data - changing these parameters

# Path of template matrix
lp = os.path.join("array_text_files","ideal_patterns","half_and_half.txt")

# Path to save the matrix for LES
sp_les = os.path.join("LES_ready","half_patterns","case509_roughdiff")

# path to save the images of maps or patterns
#sp_img = os.path.join("img", "observed_ice_maps")

#%% loading matrix and giving information about it

# Load the template matrix
master_sfc_template = np.loadtxt(lp)
master_sfc_template = master_sfc_template[:64,:64] # set the size
plt.matshow(master_sfc_template,cmap=cmap) # matshow

# save the map as an image for presentations
#plt.imshow(master_sfc_template, cmap=cmap,norm=norm)
#plt.axis('off')
#plt.title(filename[:-4] + " - resampled & filled")
#plt.savefig(os.path.join(sp_img, filename[:-4] + "_" + ponding + "_for_LES"), bbox_inches='tight')

# information about statistics of the map
unique, counts = np.unique(master_sfc_template, return_counts=True)
percentage_dict = dict(zip(unique, counts*100/master_sfc_template.size))
print(f"\nStatistics for this loaded matrix: {str(percentage_dict)}")

#%% Filling in Data - DONT TOUCH ANYTHING BELOW HERE

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
np.savetxt(os.path.join(sp_les, "T_s_remote_ice.txt"), temp_sfc, delimiter=' ', fmt='%E')
np.savetxt(os.path.join(sp_les, "zo_remote_ice.txt"), zo_sfc, delimiter=' ', fmt='%E')
np.savetxt(os.path.join(sp_les, "q_s_remote_ice.txt"), q_sfc, delimiter=' ', fmt='%E')

# Finish
print("\n  Three templates created and text arrays saved to {sp_les}!")