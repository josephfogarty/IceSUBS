# Heat Equation Solver in Sea Ice

## Overview

This repository is designed to hold all of the codes for solving one-dimensional the heat equation numerically in a block of sea ice floating on sea water. Eventually, this code will be coupled with the Large Eddy Simulation (LES) code that is being used in the Environmental Fluid Mechanics laboratory at Princeton University.

Collaborators:
- Joseph J. Fogarty
- Elie Bou-Zeid

## Physical System

We now describe our physical system and problem statement. We consider a block of sea ice floating in the Arctic ocean that is 2 meters deep, with the top above the water and the bottom below the water. If we consider the temperature to be constant in the horizontal, then we are able to solve the 1D heat equation in the vertical for the temperature of the ice, giving a temperature profile of the ice block.

There are a number of fluxes that occur on the bottom and top of this ice sheet. The figure below summarizes all of the fluxes.

![Sea Ice Flux Diagram](https://github.com/jjf1218/seaicecode/blob/master/img/seaicefluxes.jpg "Sea Ice Flux Diagram")

### Top Fluxes

Net Radiation: The sum of net shortwave and longwave fluxes. Net shortwave flux is the solar radiation that is not reflected by the surface, which is described further in the model below. We are assuming no other outgoing shortwave from the surface. Incoming longwave is any longwave not emitted by the surface - this number is set to a constant in the model. Outgoing longwave simply follows the Stefan-Boltzmann law.

Sensible Heat Convection: This represents the flux of heat into and out of the surface, depending on the temperature of the two mediums.

Latent Heat of Sublimation: This represents the amount of latent heat into and out of the surface, depending on if sublimation or deposition occurs at each time step.

Latent Heat of Evaporation: This is currently turned off in the model. We are assuming that the only process taking place at the surface is sublimation. However, it would be implemented the same as sublimation, the only change being in the latent heat capacity.

### Bottom Fluxes

## Numerics of the Code

As mentioned above, the domain for this ice block is 2 meters. Using a spatial mesh of 400 cells, we have a length of 5 mm between nodes. We set a dt of 0.5 seconds, however this number may be changed in the code.

The main numerical method utilized here is the Crank-Nicolson method, an implicit method that is second-order accurate in space and time. The Crank-Nicolson method gives rise to solving a sparse linear system every time step. The Python module `scipy.sparse` allows for quick iterations of solving these sparse linear systems.

## Models in the Code

There are a few models in the code based off of real-world observations and theory. These will eventually be replaced with actual, continuous observations, but for now, these models work well to crudely study the fluxes in sea ice.

### Humidity Equation

First we define a function to calculate the specific humidity above ice (at a certain density) given a certain temperature:

```python
# Calculate q from T
def ice_q(T):
    e = 611*np.exp((Ls/R_v)*((1/273)-(1/T)))
    return (eps_R*e)/(p_a+e*(eps_R-1))
```
This fucntion is based on the integrated Clausius-Calpeyron equation, using values for ice instead of water.

### Solar Radiation Model

There is also a function to calculate the total shortwave flux via solar radiation. The sunrise and sunset values (0700 and 2200, respectively) were estimated from picking some arbitrary day in early April and estimating the sunrise and sunset times (via timeanddate.com). A sinusoidal profile was then fit around this data.

```python
#Function to calculate incoming solar value, starting at midnight
def sw_net(t): #t is in seconds, so dt*i would be evaluated
    t_hours = (t/3600.0)%24
    if (t_hours) < 7 or (t_hours) > 22:
        #print(f"hour = {(i*dt/(3600.0))%24}, dark")
        sw_in = 0.0
    else:
        sw_in = 500*np.sin((np.pi/15.0)*(t_hours-7.0))
        #print(f"hour = {(i*dt/(3600.0))%24}, light")
        #print(f"sw_in = {sw_in}")
    shortwave_net = (1-alpha)*sw_in; # net shortwave, W/m^2 
    return shortwave_net
```

### Air Temperature Model

We also set a crude model for air temperature. Daily maximum and minimum temperature data for Barrow, AK (retrieved from the [Alaska Climate Research Center](http://climate.gi.alaska.edu/AKweather/Index.html "AK Climate Research Center")) for April 6th (arbitrary date) was averaged for nearly 100 years of data, resulting in an average minimum and maximum. A sine curve was then fitted to these values to represent a 24-hour cycle.

```python
# Define function to calculate temperature based on time of day
def air_temp(t): #t is in seconds, so dt*i would be evaluated
    t_hours = (t/3600.0)%24
    temp = 7.0*np.sin((np.pi/12.0)*(t_hours-13.0))+268
    return temp
```

The consants used in all of these functions may be found in the first lines of code in `main.py`, and can be changed to your liking.

## Animation Script

The script `animator.py` takes teh output produced by the main code and creates an MPEG file of the time evolution of the temperature profile. It utilizes the `matplotlib` libraries `matplotlib.animation` as well as `matplotlib.pyplot`.


