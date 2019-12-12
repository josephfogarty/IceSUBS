"""
iceheat_graphics:
plotting and figures from output of sea ice heat equation solver

@jjf1218
"""

#colors
bluecol = 'tab:blue'
redcol = 'tab:red'


#Bottom Fluxes
title_bottom = f"Time Evolution of Bottom Fluxes after {t_days:.2f} days"
plt.plot(time_hours,Lf_b_list,label="Latent Heat (f) Flux")
plt.plot(time_hours,G_b_list,label="Conductive Heat Flux")
plt.plot(time_hours,H_b_list,label="Sensible Heat Flux")
plt.plot(time_hours,bottom_flux_sum_list,label="Bottom Flux Sum")
plt.title(title_bottom)
plt.xlabel("Time (hr)")
plt.ylabel('Flux (W m**-2)')
plt.grid()
plt.legend(prop={'size':20})
#plt.tight_layout()
plt.savefig("figures/bottom_fluxes_temporal.png")
plt.close()

#Top fluxes
title_top=f"Time Evolution of Top Fluxes after {t_days:.2f} days"
plt.plot(time_hours,rad_net_list,label="Radiation Net Flux")
plt.plot(time_hours,H_t_list,label="Sensible Heat Flux")
plt.plot(time_hours,Ls_t_list,label="Latent Heat (s) Flux")
plt.plot(time_hours,G_t_list,label="Conductive Heat Flux")
plt.plot(time_hours,top_flux_sum_list,label="Top Flux Sum")
plt.title(title_top)
plt.ylabel('Flux (W m**-2)')
plt.xlabel("Time (hr)")
plt.legend()
plt.xlim(left=0)
plt.grid()
plt.tight_layout()
plt.savefig("figures/top_fluxes_temporal.png")
plt.close()

#Radiation budget
title_top_rad=f"Time Evolution of Top Radiative Fluxes after {t_days:.2f} days"
plt.plot(time_hours,lw_in_list,label="Longwave In Flux")
plt.plot(time_hours,lw_out_list,label="Longwave Out Flux")
plt.plot(time_hours,sw_net_list,label="Shortwave Net Flux")
plt.plot(time_hours,rad_net_list,label="Total Net Radiative Flux")
plt.title(title_top_rad)
plt.ylabel('Flux (W m**-2)')
plt.xlabel("Time (hr)")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("figures/radiative_fluxes_temporal.png")
plt.close()

#Plot time evoluation of mass loss (top and bottom)
title_mass=f"Mass Loss Rate after {t_days:.2f} days"
plt.title(title_mass)
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Mass Loss Rate (Top) (kg s**-1 m**-2)",color=bluecol)
ax1.plot(time_hours,mass_loss_top_list,color=bluecol)
ax2 = ax1.twinx()
ax2.set_ylabel("Mass Loss Rate (Bottom) (kg s**-1 m**-2)",color=redcol)
ax2.plot(time_hours,mass_loss_bottom_list,color=redcol)
ax2.tick_params(axis='y')
plt.tight_layout()
plt.grid()
plt.savefig("figures/mass_loss_temporal.png")
print(f"Mass change on top: {sum(mass_loss_top_list)*dt:.3f}")
print(f"Mass change on bottom: {sum(mass_loss_bottom_list)*dt:.3f}")
plt.close()

#Thickness loss (top and bottom)
title_mass2=f"Thickness Loss Rate after {t_days:.2f} days"
plt.title(title_mass2)
fig, ax1 = plt.subplots()
ax1.set_xlabel("Time (hr)")
ax1.set_ylabel("Thickness Loss Rate (Top) (kg s**-1 m**-2)",color=bluecol)
ax1.plot(time_hours,thickness_loss_top_list,color=bluecol)
ax2 = ax1.twinx()
ax2.set_ylabel("Thickness Loss Rate (Bottom) (kg s**-1 m**-2)",color=redcol)
ax2.plot(time_hours,thickness_loss_bottom_list,color=redcol)
ax2.tick_params(axis='y')
plt.tight_layout()
plt.grid()
plt.savefig("figures/thickness_loss_temporal.png")
print(f"Thickness change on top: {(sum(thickness_loss_top_list)*dt):.6f}")
print(f"Thickness change on bottom: {(sum(thickness_loss_bottom_list)*dt):.6f}")
plt.close()

#Surface temp and air temp comparison
title_T_it=f"Surface and Air Temperature Evolution after {t_days:.2f} days"
plt.plot(time_hours,top_ice_temp_list,label="Top of Ice Surface Temperature")
plt.plot(time_hours,air_temp_list,label="Air Temperature")
plt.title(title_T_it)
plt.xlabel("Time (hr)")
plt.ylabel('Temperature (K)')
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("figures/surface_and _air_temp_temporal.png")
plt.close()

#%% for POSTEr


#Radiation budget

plt.rcParams.update({'font.size': 28}) # smaller font for smaller fig
fig1d, ax1d = plt.subplots(figsize=(15,8))
ax1d.axhline(y=0, color='k')
ax1d.axvline(x=0, color='k')
ax1d.plot(time_hours,lw_in_list,label=r"$LW_{in}$",linewidth=3.3)
ax1d.plot(time_hours,lw_out_list,label=r"$LW_{out}$",linewidth=3.3)
ax1d.plot(time_hours,sw_net_list,label=r"$SW_{net}$",linewidth=3.3)
ax1d.plot(time_hours,rad_net_list,label=r"$R_{net}$",linewidth=3.3)
ax1d.set_title(r"$R_{net}$ Diurnal Evolution")
ax1d.set_ylabel(r'$R$ (W m$^{-2}$)')
ax1d.set_xlabel(r'$t$ (hr)')

plt.rcParams.update({'font.size': 22}) # smaller font for smaller fig
ax1d.legend(loc='best')
ax1d.grid()
plt.show()
#plt.close()
#%%
# temperature time series

plt.rcParams.update({'font.size': 28}) # smaller font for smaller fig
figtemp, axtemp = plt.subplots(figsize=(15,8))
#axtemp.axhline(y=0, color='k')
axtemp.axvline(x=0, color='k')
axtemp.plot(time_hours,top_ice_temp_list,label=r"$T_s$",linewidth=3.3)
axtemp.plot(time_hours,air_temp_list,label=r"$T_a$",linewidth=3.3)
axtemp.set_title(r"$T$ Diurnal Evolution")
axtemp.set_ylabel(r'$T$ (K)')
axtemp.set_xlabel(r'$t$ (hr)')

plt.rcParams.update({'font.size': 24}) # smaller font for smaller fig
axtemp.legend(loc='best')
axtemp.grid()
plt.show()



title_top_rad=f"Time Evolution of Top Radiative Fluxes after {t_days:.2f} days"
plt.plot(time_hours,lw_in_list,label="Longwave In Flux")
plt.plot(time_hours,lw_out_list,label="Longwave Out Flux")
plt.plot(time_hours,sw_net_list,label="Shortwave Net Flux")
plt.plot(time_hours,rad_net_list,label="Total Net Radiative Flux")
plt.title(title_top_rad)
plt.ylabel('Flux (W m**-2)')
plt.xlabel("Time (hr)")
plt.grid()

plt.legend()
plt.tight_layout()
plt.savefig("figures/poster_radiative_fluxes_temporal.png")
plt.close()

#Surface temp and air temp comparison
title_T_it=f"Surface and Air Temperature Evolution after {t_days:.2f} days"
plt.plot(time_hours,top_ice_temp_list,label="Top of Ice Surface Temperature")
plt.plot(time_hours,air_temp_list,label="Air Temperature")
plt.title(title_T_it)
plt.xlabel("Time (hr)")
plt.ylabel('Temperature (K)')
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig("figures/poster_surface_and _air_temp_temporal.png")
plt.close()
