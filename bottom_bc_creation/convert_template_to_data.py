# A code to convert surface file templates to T, zo, and q surface files
# to be used in the LES code

# Import needed libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
import os

# Colormap properties for ice and water
cmap = ListedColormap(['xkcd:off white', 'xkcd:midnight blue'])
bounds = [0.5,1.5,2.5]
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

# define a function to create strips based on an array of 1s and 2s
def ideal_arrays(arr, Nx, frac_sea):
    
    # values
    ice = 1
    sea = 2
    Ny = Nx
    
    # create cutoff indices
    cutoff_index_sea_pond = Nx - int((frac_sea/100.0)*Nx)
    
    # create the array
    arr = np.full((Nx,Ny),ice)
    arr[:,cutoff_index_sea_pond:] = sea
    
    #transpose it
    arr_trans = arr.T
    
    # return
    return arr.astype(float), arr_trans.astype(float)

# Define rougnesses corresponding to surface type
zo_ice = 0.0002
zo_sea = 0.002
zo_pond = 0.001

# Define temperatures corresponding to surface type
T_ice = 270.15 #Kelvin, -3 degrees C
T_sea = 274.15 #Kelvin, 1 degrees C
T_pond = 275.15 # Kelvin, 2 degrees C

# Calculate specific humdity based n these temperatures above
#q_ice = ice_q(T_ice)
#q_sea = liquid_q(T_sea)
#q_pond = liquid_q(T_pond)

# DONT TOUCH ANYTHING ABOVE HERE
# unless you are changing sea ice/ocean parameters
#%% Loading Data - changing these parameters

# CHANGE HERE: Path of template matrix text files
# this should be the path of a map file
lp = os.path.join("array_text_files","observed_ice_maps","low_reso","esiber_2000jul06a_2c_64x64.out")
# ideal maps will be created later using function defined above

# CHANGE HERE: Paths to save all the matrices for the LES
sp_les = os.path.join("LES_ready","maptest","esiber_2000_jul06","reso64_map")
sp_les_ideal = os.path.join("LES_ready","maptest","esiber_2000_jul06","reso64_ideal")
sp_les_ideal_trans = os.path.join("LES_ready","maptest","esiber_2000_jul06","reso64_ideal_trans")



# path to save the images of maps or patterns
#sp_img = os.path.join("img", "observed_ice_maps")

#%% loading matrix and giving information about it

# Load the template matrix
master_sfc_template = np.loadtxt(lp)
master_sfc_template = master_sfc_template[:64,:64] # set the size

# information about statistics of the map
unique, counts = np.unique(master_sfc_template, return_counts=True)
percentage_dict = dict(zip(unique, counts*100/master_sfc_template.size))
print(f"\n  Statistics for this loaded matrix: {str(percentage_dict)}")

# ger the fractions from uploaded map file
frac_ice = round(percentage_dict[1])
frac_sea = round(percentage_dict[2])
print(f"  Ice fraction plus sea fraction is {frac_ice + frac_sea}%!")

# now use the imported data to create two other arrays
arr_ideal, arr_ideal_trans = ideal_arrays(master_sfc_template, 64, frac_sea)

# view all three figures
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 5), ncols=3)
ax1.set_title("master_sfc_template")
im_map = ax1.imshow(master_sfc_template,cmap=cmap)
fig.colorbar(im_map, ax=ax1)
ax2.set_title("arr_ideal")
im_ideal = ax2.imshow(arr_ideal,cmap=cmap)
fig.colorbar(im_ideal, ax=ax2)
ax3.set_title("arr_ideal_trans")
im_ideal_trans = ax3.imshow(arr_ideal_trans,cmap=cmap)
fig.colorbar(im_ideal_trans, ax=ax3)
plt.show()


#%% Filling in Data - DONT TOUCH ANYTHING BELOW HERE

# a function that takes an array and returns two different arrays, for temp and zo
def create_maps(array):
    
    # Create the temp matrix
    temp_sfc = array.copy()
    temp_sfc[temp_sfc == 1] = T_ice
    temp_sfc[temp_sfc == 2] = T_sea
    temp_sfc[temp_sfc == 3] = T_pond
    print("  unique temps: ", np.unique(temp_sfc))
    
    # Create roughness matrix
    zo_sfc = array.copy()
    zo_sfc[zo_sfc == 1] = zo_ice
    zo_sfc[zo_sfc == 2] = zo_sea
    zo_sfc[zo_sfc == 3] = zo_pond
    print("  unique roughnesses: ",np.unique(zo_sfc))
    
    # Create humidity matrix
    #q_sfc = array
    ##q_sfc[q_sfc == 1] = q_ice
    ##q_sfc[q_sfc == 2] = q_sea
    ##q_sfc[q_sfc == 3] = q_pond
    #print(np.unique(q_sfc))
    
    return temp_sfc, zo_sfc

# create Ts and zo for map, ideal, and ideal_trans
print("\n  for original map")
ts_map, zo_map = create_maps(master_sfc_template)
print("\n  for ideal map")
ts_map_ideal, zo_map_ideal = create_maps(arr_ideal)
print("\n  for ideal_trans map")
ts_map_ideal_trans, zo_map_ideal_trans = create_maps(arr_ideal_trans)

# Save the map matrices for temp and z0
np.savetxt(os.path.join(sp_les, "T_s_remote_ice.txt"), ts_map, delimiter=' ', fmt='%E')
np.savetxt(os.path.join(sp_les, "zo_remote_ice.txt"), zo_map, delimiter=' ', fmt='%E')
#np.savetxt(os.path.join(sp_les, "q_s_remote_ice.txt"), q_sfc, delimiter=' ', fmt='%E')
print(f"\n  T_s and zo templates created, and text arrays for the map saved to {sp_les}!")

# Save the IDEAL matrices for temp and z0
np.savetxt(os.path.join(sp_les_ideal, "T_s_remote_ice.txt"), ts_map_ideal, delimiter=' ', fmt='%E')
np.savetxt(os.path.join(sp_les_ideal, "zo_remote_ice.txt"), zo_map_ideal, delimiter=' ', fmt='%E')
#np.savetxt(os.path.join(sp_les, "q_s_remote_ice.txt"), q_sfc, delimiter=' ', fmt='%E')
print(f"  T_s and zo templates created, and text arrays for ideal saved to {sp_les_ideal}!")

# Save the IDEAL_TRANS matrices for temp and z0
np.savetxt(os.path.join(sp_les_ideal_trans, "T_s_remote_ice.txt"), ts_map_ideal_trans, delimiter=' ', fmt='%E')
np.savetxt(os.path.join(sp_les_ideal_trans, "zo_remote_ice.txt"), zo_map_ideal_trans, delimiter=' ', fmt='%E')
#np.savetxt(os.path.join(sp_les, "q_s_remote_ice.txt"), q_sfc, delimiter=' ', fmt='%E')
print(f"  T_s and zo templates created, and text arrays for ideal_trans saved to {sp_les_ideal_trans}!")







