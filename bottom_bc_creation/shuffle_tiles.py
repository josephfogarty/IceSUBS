"""
A code to take a numpy array and "shuffle" the values
"""

import numpy as np
import matplotlib.pyplot as plt

arr = np.arange(100).reshape(10,10)

def shuffle(arr, n_sections):

    assert arr.shape[0]==arr.shape[1], "arr must be square"
    assert arr.shape[0]%n_sections == 0, "arr size must divideable into equal n_sections"

    size = arr.shape[0]//n_sections


    new_arr = np.empty_like(arr)
    ## randomize section's row index

    rand_indxes = np.random.permutation(n_sections*n_sections)

    for i in range(n_sections):
        ## randomize section's column index
        for j in  range(n_sections):

            rand_i = rand_indxes[i*n_sections + j]//n_sections
            rand_j = rand_indxes[i*n_sections + j]%n_sections

            new_arr[i*size:(i+1)*size, j*size:(j+1)*size] = \
                arr[rand_i*size:(rand_i+1)*size, rand_j*size:(rand_j+1)*size]

    return new_arr

result = shuffle(arr, 5)

plt.matshow(arr)
plt.matshow(result)
plt.show()
