#%% Modeling Sunrise for Barrow, AK based on ~Apr 6,
# Times in local time


# Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Solar Parameters?
alpha = 0.3; # albedo
sw_in = 500; # incoming shortwave, W/m^2
sw_net = (1-alpha)*sw_in; # net shortwave, W/m^2

# Time parameters
dt = 0.5; # time between iterations, seconds
nt = 600600; # amount of iterations

time = []
sw_in_plot = []
temp_list = []

#for i in range(0,nt):
#    # calculate solar radiation
#    t_hours = ((i*dt)/3600.0)
#    if t_hours%24 < 7 or t_hours%24 > 22:
#        print(f"hour = {(i*dt/(3600.0))%24}, dark")
#        sw_in = 0.0
#    else:
#        sw_in = 500*np.sin((np.pi/15.0)*((t_hours%24)-7.0))
#        print(f"hour = {(i*dt/(3600.0))%24}, light")
#        print(f"sw_in = {sw_in}")
#    # calculate ambient temperature
#    temp = 7.0*np.sin((np.pi/12.0)*(t_hours-13.0))+268
#    # append to lists
#    time.append(i*dt/3600)
#    sw_in_plot.append(sw_in)
#    temp_list.append(temp)

for i in range(0,nt):
    
    # define the input time in hours for a diurnal cycle
    t_hrs = ((i*dt)/3600)%24
    
    # define parameters
    B_0 = 500.0
    B_1 = -100.0
    
    #calculate solar radiation
    Q = max([B_0*np.sin(2*np.pi*(t_hrs)/24),B_1])
    print(f"time={t_hrs:.4f}, rad={Q:.3f}")
    
    # calculate ambient temperature
    temp = 7.0*np.sin((np.pi/12.0)*(t_hrs-13.0))+268
    
    # append to lists
    time.append(i*dt/3600)
    sw_in_plot.append(Q)
    temp_list.append(temp)

color = 'tab:blue'
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Air Temperature (K)",color=color)
ax1.plot(time,temp_list,color=color)#label="Incoming Solar Radiation")
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel("Net Solar Flux (W m**-2)",color=color)
ax2.plot(time,sw_in_plot,color=color)#,label="Ambient Temperature")
ax2.tick_params(axis='y',labelcolor=color)
plt.title("Modeled Air Temperature and Incoming Solar Flux for Barrow, AK, in Early Spring")
plt.tight_layout()