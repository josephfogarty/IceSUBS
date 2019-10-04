# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 20:59:52 2019

@author: jf38
"""

# shuffling tiles script

#%%

mat_2d = np.reshape(np.arange(0,100),(10,10))

import numpy as np

def shuff(arr, n_sections):

    assert arr.shape[0]==arr.shape[1], "arr must be square"
    assert arr.shape[0]%n_sections == 0, "arr size must divideable into equal n_sections"

    size = arr.shape[0]//n_sections


    new_arr = np.empty_like(arr)
    ## randomize section's row index
    for i, rand_i in enumerate(np.random.permutation(n_sections)):
        ## randomize section's column index
        for j, rand_j in  enumerate(np.random.permutation(n_sections)):
            new_arr[i*size:(i+1)*size, j*size:(j+1)*size] = \
                arr[rand_i*size:(rand_i+1)*size, rand_j*size:(rand_j+1)*size]

    return new_arr

#set tile size
n = 5

# shuffle
result = shuff(shuff(shuff(shuff(mat_2d, n).T,n).T,n).T,n).T

plt.matshow(mat_2d)
plt.matshow(result)