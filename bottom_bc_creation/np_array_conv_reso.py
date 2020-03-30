"""
This script will take an ice map (an array) of a certain resolution and
"decrease" this resolution by finding an average of the cells in the "tile"

NEED TO EDIT TO INCLUDE PONDS EVENTUALLY
"""
# import needed libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# Colormap properties for ice, water, and pond
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
cmap = ListedColormap(['black','xkcd:off white', 'xkcd:midnight blue', 'xkcd:cyan'])
bounds = [-0.4,0.5,1.5,2.5,3.5]
norm = BoundaryNorm(bounds,cmap.N)

#%% define the converting function

def conv_res(data, rows, cols):
    
    # the matrix to return
    shrunk = np.zeros((rows,cols))
    
    # iterate through rows and columns
    for i in range(0,rows):
        for j in range(0,cols):
            
            # get the indices
            row_sp = int(data.shape[0]/rows)
            col_sp = int(data.shape[1]/cols)
            
            # each sub area
            zz = data[i*row_sp : i*row_sp + row_sp, j*col_sp : j*col_sp + col_sp]
            
            # assign the average to  the returned matrix
            shrunk[i,j] = round(np.mean(zz))
    
    # return        
    return shrunk

#%% Import the array
    
# load the text file of the array
ice_map = "esiber_2000jul06a_2c.out"
lp = os.path.join("array_text_files","observed_ice_maps","without_ponds",ice_map)
loaded_mat = np.loadtxt(lp)


new_reso_mat = conv_res(loaded_mat, 64, 64)

print("unique values for original: ", np.unique(loaded_mat))
print("unique values for new: ", np.unique(new_reso_mat))

#%% Plot and save

#plot
fig = plt.figure(figsize = (7,3.5))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(loaded_mat,cmap=cmap,norm=norm)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(new_reso_mat,cmap=cmap,norm=norm)
plt.tight_layout()

# save path
sp_array = os.path.join("array_text_files","observed_ice_maps","low_reso",ice_map[:-4]+"_64x64.out")
sp_img = os.path.join("img", "observed_ice_maps",ice_map[:-4]+"_64x64_comp.jpg")

#save array as text file and image
plt.savefig(sp_img)
np.savetxt(sp_array, new_reso_mat)






