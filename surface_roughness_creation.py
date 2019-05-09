#Code to create a heterogeneous surface

#import libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['xkcd:sky blue', 'xkcd:dark blue'])



#define two rougnesses corresponding to ice/sea
zo_ice = 0.0002
zo_sea = 0.002
zo = [zo_ice, zo_sea]

#define two temperatures corresponding to ice/sea and above zo's
T_ice = 270.15 #Kelvin, -3 degrees C
T_sea = 274.15 #Kelvin, 1 degrees C
temp = [T_ice, T_sea]

#put together in a list of lists
params = [zo, temp]

#create strings for filenaming
params_str = ['zo', 'temp']


#define grid size
Nx = 96
Ny = 96

#fill half and half
for i in range(len(params)):

    #create the surface
    surface = np.tile(np.concatenate((\
              np.repeat(params[i][0], int(Ny/2)),\
              np.repeat(params[i][1], int(Ny/2)))),(Nx,1))
    
    #write to a file
    np.savetxt(f'surfacefiles/halfandhalf{params_str[i]}.out', surface, delimiter=',',fmt='%.4f')


#fill checkerboard 2x2
for i in range(len(params)):
    
    #create the surface
    surface1 = np.tile(np.concatenate((\
              np.repeat(params[i][0], int(Ny/2)),\
              np.repeat(params[i][1], int(Ny/2)))),(int(Nx/2),1))
    surface2 = np.fliplr(surface1)
    surfacechecker2 = np.concatenate((surface1, surface2))
    
    #write to a file  
    np.savetxt(f'surfacefiles/checker2{params_str[i]}.out', surfacechecker2, delimiter=' ',fmt='%.4f')
    
#fill checkerboard 4x4
for i in range(len(params)):
    
    #create the surface
    surface1 = np.tile(np.concatenate((\
              np.repeat(params[i][0], int(Ny/4)),\
              np.repeat(params[i][1], int(Ny/4)))),(int(Nx/4),2))
    surface2 = np.fliplr(surface1)
    surfacechecker4 = np.concatenate((surface1, surface2, surface1, surface2))
    
    #write to a file  
    np.savetxt(f'surfacefiles/checker4{params_str[i]}.out', surfacechecker4, delimiter=' ',fmt='%.4f')

#fill checkerboard 8x8

for i in range(len(params)):
    
    #create the surface
    surface1 = np.tile(np.concatenate((\
              np.repeat(params[i][0], int(Ny/8)),\
              np.repeat(params[i][1], int(Ny/8)))),(int(Nx/8),4))
    surface2 = np.fliplr(surface1)
    surfacechecker8 = np.concatenate((surface1, surface2, surface1, surface2))
    surfacechecker8 = np.concatenate((surfacechecker8, surfacechecker8))
    #write to a file  
    np.savetxt(f'surfacefiles/checker8{params_str[i]}.out', surfacechecker8, delimiter=' ',fmt='%.4f')


plt.matshow(surface,cmap=cmap)
plt.matshow(surfacechecker2,cmap=cmap)
plt.matshow(surfacechecker4,cmap=cmap)
plt.matshow(surfacechecker8,cmap=cmap)


#%% close all matshows

plt.close("all")