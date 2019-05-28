"""
Goal: Take output from different schemes and create plots and make a movie
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Heat Solver Movie', artist='JJF',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

fig = plt.figure()
l, = plt.plot([], [], 'k-o')

n = 400

file_location = f'main/solutions/ice_solver_{n+1}nodes.txt'

loaded_matrix_cn = np.loadtxt(file_location, dtype='f', delimiter=' ')
with writer.saving(fig, f"cn_{len(loaded_matrix_cn[1])}_node_solution.mp4", 100):    
    x = np.linspace(0.0, 2.0, len(loaded_matrix_cn[0])) 
    for i in range(len(loaded_matrix_cn)):
        print(f"%={i/len(loaded_matrix_cn):.4f}")
        y = loaded_matrix_cn[i]
        plt.plot(x,y,'g')
        plt.title(f"Time Evolution of Heat Equation Solver - CN")
        writer.grab_frame()
        plt.clf()
print('\ncrank movie done')        










