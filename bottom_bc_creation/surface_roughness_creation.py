#Code to create a heterogeneous surface

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# Colormap properties for ice, water, and pond
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
cmap = ListedColormap(['black','xkcd:off white', 'xkcd:midnight blue', 'xkcd:cyan'])
bounds = [-0.4,0.5,1.5,2.5,3.5]
norm = BoundaryNorm(bounds,cmap.N)

# Define grid size, must be divisible by 8
Nx = 96
Ny = 96

# Create half and half surface
surface = np.tile(np.concatenate((\
          np.repeat(1, int(Ny/2)),\
          np.repeat(2, int(Ny/2)))),(Nx,1))

# Create half and half (the other way) surface
surface_flip = surface.transpose()

# Create checkerboard 2x2 surface template
surface1 = np.tile(np.concatenate((\
          np.repeat(1, int(Ny/2)),\
          np.repeat(2, int(Ny/2)))),(int(Nx/2),1))
surface2 = np.fliplr(surface1)
surfacechecker2 = np.concatenate((surface1, surface2))

# Create checkerboard 4x4 surface template
surface1 = np.tile(np.concatenate((\
           np.repeat(1, int(Ny/4)),\
           np.repeat(2, int(Ny/4)))),(int(Nx/4),2))
surface2 = np.fliplr(surface1)
surfacechecker4 = np.concatenate((surface1, surface2, surface1, surface2))

# Create checkerboard 8x8 surface template
surface1 = np.tile(np.concatenate((\
          np.repeat(1, int(Ny/8)),\
          np.repeat(2, int(Ny/8)))),(int(Nx/8),4))
surface2 = np.fliplr(surface1)
surfacechecker8 = np.concatenate((surface1, surface2, surface1, surface2))
surfacechecker8 = np.concatenate((surfacechecker8, surfacechecker8))

#%% Write all surface template arrays to files

# save path for text files
sp = os.path.join("array_text_files", "ideal_patterns")

# save path for images
sp_img = os.path.join("img","ideal_patterns")

np.savetxt(os.path.join(sp,'halfandhalf.txt'), surface, delimiter=' ',fmt='% 4d')
np.savetxt(os.path.join(sp,'halfandhalf_flip.txt'), surface_flip, delimiter=' ',fmt='% 4d')
np.savetxt(os.path.join(sp,'checker2.txt'), surfacechecker2, delimiter=' ',fmt='% 4d')
np.savetxt(os.path.join(sp,'checker4.txt'), surfacechecker4, delimiter=' ',fmt='% 4d')
np.savetxt(os.path.join(sp,'checker8.txt'), surfacechecker8, delimiter=' ',fmt='% 4d')

#save images (checkerboards)

# save images (transpose comparison)
fig = plt.figure(figsize = (4,4))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(surface,cmap=cmap,norm=norm)
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(surface_flip,cmap=cmap,norm=norm)
plt.tight_layout()
plt.savefig(os.path.join(sp_img,"transpose_comparison.jpg"))

