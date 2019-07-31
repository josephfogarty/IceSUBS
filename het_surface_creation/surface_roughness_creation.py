#Code to create a heterogeneous surface

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['xkcd:sky blue', 'xkcd:dark blue'])

# Define grid size, must be divisible by 8
Nx = 96
Ny = 96

# Create half and half surface
surface = np.tile(np.concatenate((\
          np.repeat(1, int(Ny/2)),\
          np.repeat(2, int(Ny/2)))),(Nx,1))

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

# Write all surface template arrays to files
np.savetxt(f'surfacefiles/halfandhalf.dat', surface, delimiter=' ',fmt='% 4d')
np.savetxt(f'surfacefiles/checker2.dat', surfacechecker2, delimiter=' ',fmt='% 4d')
np.savetxt(f'surfacefiles/checker4.dat', surfacechecker4, delimiter=' ',fmt='% 4d')
np.savetxt(f'surfacefiles/checker8.dat', surfacechecker8, delimiter=' ',fmt='% 4d')

# Visualize the checkerboards
plt.matshow(surface,cmap=cmap)
plt.matshow(surfacechecker2,cmap=cmap)
plt.matshow(surfacechecker4,cmap=cmap)
plt.matshow(surfacechecker8,cmap=cmap)
#plt.close("all")