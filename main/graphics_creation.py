"""
iceheat_graphics:
plotting and figures from output of sea ice heat equation solver
@jjf1218
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from constants import cnst
#import funcs as fn

#%% import the solutions

# import the heat equation solution from the solver output
lp_ice_heat = os.path.join("solutions",f"ice_solver_401nodes_{cnst.nt}tsteps.txt")
ice_heat_solution_array = np.loadtxt(lp_ice_heat)

# import the temporal fluxes from the solver output as a dataframe
lp_temporal_fluxes = os.path.join("solutions",f"fluxes1D_401nodes_{cnst.nt-1}tsteps.csv")
temporal_fluxes_df = pd.read_csv(lp_temporal_fluxes)

#%% create plots for the 1D fluxes

# create time array because it is hard to do in constants file
time_hours = [cnst.nt*cnst.dt/3600.0 for cnst.nt in range(0,cnst.nt)]

# import all variables as a dictionary
flux_dict = temporal_fluxes_df.to_dict('list')


#%%
# plot the top fluxes
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time_hours,flux_dict["R_net"],label="Radiation Net Flux")
ax.plot(time_hours,flux_dict["H_t"],label="Sensible Heat Flux")
ax.plot(time_hours,flux_dict["Ls_t"],label="Latent Heat (s) Flux")
ax.plot(time_hours,flux_dict["G_t"],label="Conductive Heat Flux")
ax.plot(time_hours,flux_dict["top_flux_sum"],label="Top Flux Sum")
ax.set_title(f"Time Evolution of Top Fluxes after {cnst.total_t_days:.2f} days")
ax.set_ylabel(r'Flux (W m$^{-2}$)')
ax.set_xlabel("Time (hr)")
ax.set_xlim(left=0)
ax.set_xticks(np.arange(min(time_hours), max(time_hours)+1, 24))
ax.grid()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=5)


fig.savefig(os.path.join('figures','top_fluxes_temporal.png'))

plt.close()



# plot the bottom fluxes
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time_hours,flux_dict["Lf_b"],label="Latent Heat (f) Flux")
ax.plot(time_hours,flux_dict["G_b"],label="Conductive Heat Flux")
ax.plot(time_hours,flux_dict["H_b"],label="Sensible Heat Flux")
ax.plot(time_hours,flux_dict["bottom_flux_sum"],label="Bottom Flux Sum")
ax.set_title(f"Time Evolution of Bottom Fluxes after {cnst.total_t_days:.2f} days")
ax.set_xlabel("Time (hr)")
ax.set_ylabel(r'Flux (W m$^{-2}$)')
ax.set_xticks(np.arange(min(time_hours), max(time_hours)+1, 24))
ax.grid()
ax.legend()
fig.savefig(os.path.join('figures','bottom_fluxes_temporal.png'))
plt.close()

# plot the radiation budget
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time_hours,flux_dict["LW_in"],label=r"$LW_{in}$")
ax.plot(time_hours,flux_dict["LW_out"],label=r"$LW_{out}$")
ax.plot(time_hours,flux_dict["SW_net"],label=r"$SW_{net}$")
ax.plot(time_hours,flux_dict["R_net"],label=r"$R_{net}$")
ax.set_title(f"Time Evolution of Top Radiative Fluxes after {cnst.total_t_days:.2f} days")
ax.set_ylabel(r'Flux (W m$^{-2}$)')
ax.set_xlabel("Time (hr)")
ax.set_xticks(np.arange(min(time_hours), max(time_hours)+1, 24))
#plt.axhline(y=0, color='k')
ax.grid()
#ax.legend()
fig.savefig(os.path.join('figures','radiative_fluxes_temporal.png'))
plt.close()

# plot surface temp and air temp comparison
fig, ax = plt.subplots(figsize=(8,5))
ax.plot(time_hours,flux_dict["T_s"],label=r"$\theta_s$")
ax.plot(time_hours,flux_dict["T_a"],label=r"$\theta_a$")
ax.set_title(f"Surface and Air Temperature Evolution after {cnst.total_t_days:.2f} days")
ax.set_xlabel("Time (hr)")
ax.set_ylabel('Temperature (K)')
ax.set_xticks(np.arange(min(time_hours), max(time_hours)+1, 24))
ax.legend()
ax.grid()
fig.savefig(os.path.join('figures','surface_and_air_temp_temporal.png'))
plt.close()

## plot time evoluation of mass loss (top and bottom)
#title_mass=f"Mass Loss Rate after {t_days:.2f} days"
#plt.title(title_mass)
#fig, ax1 = plt.subplots()
#ax1.set_xlabel("Time (hr)")
#ax1.set_ylabel("Mass Loss Rate (Top) (kg s**-1 m**-2)",color=bluecol)
#ax1.plot(time_hours,mass_loss_top_list,color=bluecol)
#ax2 = ax1.twinx()
#ax2.set_ylabel("Mass Loss Rate (Bottom) (kg s**-1 m**-2)",color=redcol)
#ax2.plot(time_hours,mass_loss_bottom_list,color=redcol)
#ax2.tick_params(axis='y')
#plt.tight_layout()
#plt.grid()
#plt.savefig("figures/mass_loss_temporal.png")
#print(f"Mass change on top: {sum(mass_loss_top_list)*dt:.3f}")
#print(f"Mass change on bottom: {sum(mass_loss_bottom_list)*dt:.3f}")
#plt.close()
#
## plot thickness loss (top and bottom)
#title_mass2=f"Thickness Loss Rate after {t_days:.2f} days"
#plt.title(title_mass2)
#fig, ax1 = plt.subplots()
#ax1.set_xlabel("Time (hr)")
#ax1.set_ylabel("Thickness Loss Rate (Top) (kg s**-1 m**-2)",color=bluecol)
#ax1.plot(time_hours,thickness_loss_top_list,color=bluecol)
#ax2 = ax1.twinx()
#ax2.set_ylabel("Thickness Loss Rate (Bottom) (kg s**-1 m**-2)",color=redcol)
#ax2.plot(time_hours,thickness_loss_bottom_list,color=redcol)
#ax2.tick_params(axis='y')
#plt.tight_layout()
#plt.grid()
#plt.savefig("figures/thickness_loss_temporal.png")
#print(f"Thickness change on top: {(sum(thickness_loss_top_list)*dt):.6f}")
#print(f"Thickness change on bottom: {(sum(thickness_loss_bottom_list)*dt):.6f}")
#plt.close()

#%% plot the ice heat solution at selected points

# hours of the day to plot after n days
n_days = 7
t_hr = [0, 8, 16]
#after n days at these hours, where are the values (in pcount)?
t_hr_row_num = [(n_days*86400 + t*3600) for t in t_hr]

# cut x in half to only look at top half of ice
x_top_half = cnst.x[:int(len(cnst.x)/2)]

row_indices = []
# for each row at the specified second
for time in t_hr_row_num:
    # add 
    row_indices.append(np.where(ice_heat_solution_array[:,0]==time)[0][0])

fig,ax=plt.subplots()
for row in row_indices:
    row_data = ice_heat_solution_array[row,1:len(x_top_half)+1]
    ax.plot(row_data,x_top_half,label=r"t=%s hr"%(((row*2.0)-(n_days*1440))/60))
ax.legend()
ax.set_title("Temperature Profiles for a Block of Sea Ice at Different Times")
ax.set_ylabel("z")
ax.set_ylim(x_top_half[-1],x_top_half[0])
ax.set_xlabel(r"$T$")
#ax.axvline(x=0,color='black')
ax.axhline(y=0,color='black')
plt.savefig(os.path.join("figures","ice_heat_solution.jpg"))


#%% create animation of the heat in ice block - SOON

"""
Goal: Take output from different schemes and create plots and make a movie
"""
#
##import needed libraries
#import numpy as np
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt
#import matplotlib.animation as manimation
#
##create the movie writer object
#FFMpegWriter = manimation.writers['ffmpeg']
#metadata = dict(title='Heat Solver Movie', artist='JJF',
#                comment='Movie support!')
#writer = FFMpegWriter(fps=15, metadata=metadata)
#
##figure specifications
#fig = plt.figure()
#l, = plt.plot([], [], 'k-o')
#n = 400
#nt = 1000000
#
##where the data will be taken from
#file_location = f'solutions/ice_solver_{n+1}nodes_{nt}tsteps.txt'
#
##import data as a matrix
#loaded_matrix = np.loadtxt(file_location, dtype='f', delimiter=' ')
#
##start writing the movie file
#with writer.saving(fig, f"main/figures/ice_solution_movie.mp4", 100):    
#    x = np.linspace(0.0, 2.0, len(loaded_matrix[0])) 
#    for i in range(len(loaded_matrix)):
#        print(f"%={i/len(loaded_matrix):.4f}")
#        y = loaded_matrix[i]
#        plt.plot(y,x,'g')
#        plt.title(f"Time Evolution of Heat Equation Solver - CN")
#        plt.xlabel("Temperature (K)")
#        plt.ylabel("Depth (m)")
#        writer.grab_frame()
#        plt.clf()
#print('\ncrank movie done') 



