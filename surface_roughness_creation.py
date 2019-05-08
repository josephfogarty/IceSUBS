#Code to create a heterogeneous surface

#import libraries
import numpy as np


#define two rougnesses
zo1 = 0.002
zo2 = 0.0002

#define grid size
Nx = 96
Ny = 96

#create array
surface = np.ones((Ny,Nx)) #switched to match the LES code

#fill half and half
for i in surface:
    for j in surface[i]:
        print(surface[i],j)


#simulate "ice breaking up"



