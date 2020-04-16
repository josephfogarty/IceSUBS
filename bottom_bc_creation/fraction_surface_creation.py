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
cmap = ListedColormap(['xkcd:off white', 'xkcd:midnight blue'])
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
frac_ice = 10
frac_sea = 90
frac_pond = 0

#save path CHANGE THIS
filename = f"ice{round(frac_ice)}_sea{round(frac_sea)}_pond{round(frac_pond)}"
sp = os.path.join("array_text_files","ideal_patterns", filename+".txt")
sp_trans = os.path.join("array_text_files","ideal_patterns", filename+"_trans.txt")

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

#%% for diagonal arrays 

# save path for diagonals
sp_dia = os.path.join("array_text_files","ideal_patterns","diag_test")

# create full arrays
ice_arr = np.full((Nx,Ny),ice)
sea_arr = np.full((Nx,Ny),sea)

# make them diagonal
diag_only_ice = np.triu(ice_arr)
diag_only_sea = np.tril(sea_arr)
# zero out diagonal for the sea array, then add
np.fill_diagonal(diag_only_sea,0)
diag_ice = diag_only_ice + diag_only_sea

# create the other three arrays
diag_sea = diag_ice.T
diag_ice_trans = np.flipud(diag_sea)
diag_sea_trans = np.flipud(diag_ice)

# view all four figures
fig, (ax1, ax2, ax3, ax4) = plt.subplots(figsize=(13, 5), ncols=4)

ax1.set_title("diag_ice")
im_di = ax1.imshow(diag_ice,cmap=cmap)
#fig.colorbar(im_di, ax=ax1)

ax2.set_title("diag_ice_trans")
im_di_trans = ax2.imshow(diag_ice_trans,cmap=cmap)
#fig.colorbar(im_di_trans, ax=ax2)

ax3.set_title("diag_sea")
im_ds = ax3.imshow(diag_sea,cmap=cmap)
#fig.colorbar(im_ds, ax=ax3)

ax4.set_title("diag_sea_trans")
im_ds_trans = ax4.imshow(diag_sea_trans,cmap=cmap)
#fig.colorbar(im_ds_trans, ax=ax4)

plt.show()

# save all four figures
np.savetxt(os.path.join(sp_dia,"diag_ice.txt"), diag_ice)
np.savetxt(os.path.join(sp_dia,"diag_ice_trans.txt"), diag_ice_trans)
np.savetxt(os.path.join(sp_dia,"diag_sea.txt"), diag_sea)
np.savetxt(os.path.join(sp_dia,"diag_sea_trans.txt"), diag_sea_trans)