"""
Created on Fri Nov 15 13:41:02 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# load path
lp1 = os.path.join("LES_ready","half_patterns","ideal","T_s_remote_ice.txt")
lp2 = os.path.join("LES_ready","half_patterns","trans","T_s_remote_ice.txt")

loaded_mat1 = np.loadtxt(lp1)
loaded_mat2 = np.loadtxt(lp2)

plt.matshow(loaded_mat1)
plt.colorbar()

plt.matshow(loaded_mat2)
plt.colorbar()