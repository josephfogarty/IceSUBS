# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:41:02 2019

@author: jf38
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Colormap properties for ice, water, and pond
from matplotlib.colors import ListedColormap
from matplotlib.colors import BoundaryNorm
cmap = ListedColormap(['black','xkcd:off white', 'xkcd:midnight blue', 'xkcd:cyan'])
bounds = [-0.4,0.5,1.5,2.5,3.5]
norm = BoundaryNorm(bounds,cmap.N)

#values
ice = 1
sea = 2
pond = 3

#shape
Nx = 192
Ny = Nx

# put in fractions (in percent)
frac_ice = 25
frac_sea = 75
frac_pond = 0

# create cutoff indices
cutoff_index_sea_pond = Nx - int(((frac_sea + frac_pond)/100.0)*Nx)

cutoff_index_pond = Nx - (frac_pond/100.0)*Nx

arr = np.full((Nx,Ny),ice)
arr[:,cutoff_index_sea_pond:] = sea
if float(frac_pond) != 0.0:
    arr[:,cutoff_index_pond:] = pond

arr = np.pad(arr, pad_width=1, mode='constant', constant_values=-80)


fig = plt.figure(figsize = (6,6))
#ax1 = fig.add_subplot(1,2,1)
plt.imshow(arr,cmap=cmap,norm=norm)

#plt.imshow(arr, cmap=cmap,norm=norm)
#plt.axis('off')
plt.title(f"Fraction of Ice ({frac_ice}%), Sea ({frac_sea}%), and Pond ({frac_pond}%)")
filename = f"ice_{round(frac_ice)}_sea_{round(frac_sea)}_pond_{round(frac_pond)}.jpg"
plt.savefig(os.path.join("img","ideal_patterns",filename), bbox_inches='tight')



