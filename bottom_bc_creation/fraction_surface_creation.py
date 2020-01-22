"""
This script will take an existing ice map of a certain dimension 
and create an "ideal" fractional map with all of the ice and all of
the ocean in "bands," e.g. |ice(75%)|ocean(25%)|
"""

# import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# colormap properties for ice, water, and pond
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
cmap = ListedColormap(['black','xkcd:off white', 'xkcd:midnight blue', 'xkcd:cyan'])
bounds = [-0.4,0.5,1.5,2.5,3.5]
norm = BoundaryNorm(bounds,cmap.N)

#%% parameters

# representative values
ice = 1
sea = 2
pond = 3

#shape
Nx = 64
Ny = Nx

# put in fractions (in percent)
frac_ice = 50
frac_sea = 50
frac_pond = 0

#%% create the array

# create cutoff indices
cutoff_index_sea_pond = Nx - int(((frac_sea + frac_pond)/100.0)*Nx)
cutoff_index_pond = Nx - (frac_pond/100.0)*Nx

# create the array
arr = np.full((Nx,Ny),ice)
arr[:,cutoff_index_sea_pond:] = sea
if float(frac_pond) != 0.0:
    arr[:,cutoff_index_pond:] = pond

#transpose it
arr_trans = arr.T

#%% saving the numerical array 

#save path CHANGE THIS
sp = os.path.join("array_text_files","ideal_patterns", "half_and_half.txt")
sp_trans = os.path.join("array_text_files","ideal_patterns", "half_and_half_trans.txt")
np.savetxt(sp, arr)
np.savetxt(sp_trans, arr_trans)

#create a padded array for a picture
#arr_pad = np.pad(arr, pad_width=1, mode='constant', constant_values=-80)
#arr_pad_trans = np.pad(arr_trans, pad_width=1, mode='constant', constant_values=-80)

#%% saving the picture

# plot and save the map as a picture
fig = plt.figure(figsize = (6,6))
plt.imshow(arr,cmap=cmap,norm=norm)
plt.axis('off')
plt.tight_layout()
#plt.title(f"Fraction of Ice ({frac_ice}%), Sea ({frac_sea}%), and Pond ({frac_pond}%)")
filename = f"ice_{round(frac_ice)}_sea_{round(frac_sea)}_pond_{round(frac_pond)}.jpg"
plt.savefig(os.path.join("img","ideal_patterns",filename), bbox_inches='tight')
plt.close()

## now the transpose
#fig_trans = plt.figure(figsize = (6,6))
#plt.imshow(arr_trans,cmap=cmap,norm=norm)
##plt.title(f"Fraction of Ice ({frac_ice}%), Sea ({frac_sea}%), and Pond ({frac_pond}%)")
#plt.axis('off')
#plt.tight_layout()
#filename = f"ice_{round(frac_ice)}_sea_{round(frac_sea)}_pond_{round(frac_pond)}_TRANS.jpg"
#plt.savefig(os.path.join("img","ideal_patterns",filename), bbox_inches='tight')
#plt.close()


