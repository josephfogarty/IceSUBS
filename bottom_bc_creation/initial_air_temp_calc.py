# A code to get a starting initial air temperature for an LES simulation

# import needed modules
import numpy as np
import os

# load path for the template
lp = os.path.join("array_text_files","observed_ice_maps","without_ponds","beaufo_2000aug31a_3c.out")

arr = np.loadtxt(lp)

unique, counts = np.unique(arr, return_counts=True)
per_dict = dict(zip(unique, counts/arr.size))

# Define temperatures corresponding to surface type
T_ice = 269.35 #Kelvin
T_sea = 273.35 #Kelvin
T_pond = 272.35 # Kelvin

air_temp = T_ice*per_dict[1] + T_sea*per_dict[2]

print(f"The initial air temp is {air_temp}")