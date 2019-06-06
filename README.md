# Heat Equation Solver in Sea Ice

## Overview

This repository is designed to hold all of the codes for solving one-dimensional the heat equation numerically in a block of sea ice floating on sea water. Eventually, this code will be coupled with the Large Eddy Simulation (LES) code that is being used in the Environmental Fluid Mechanics laboratory at Princeton University.

Collaborators:
- Joseph J. Fogarty
- Elie Bou-Zeid

## Physical System

We now describe our physical system and problem statement. We consider a block of sea ice floating in the Arctic ocean that is 2 meters deep, with the top above the water and the bottom below the water. If we consider the temperature to be constant in the horizontal, then we are able to solve the 1D heat equation in the vertical for the temperature of the ice, giving a temperature profile of the ice block.

## Numerics of the Code

As mentioend above, the domain for this ice block is 2 meters. Using a spatial mesh of 400 cells, we have a length of 5 mm between nodes. We set a dt of 0.5, however this number may be changed in the code.

The main numerical method utilized here is the Crank-Nicolson method, an implicit method that is second-order accurate in space and time. The Crank-Nicolson method gives rise to solving a sparse linear system every time step. The Python module `scipy.sparse` allows for quick iterations of solving these systems.

## Models in the Code

There are a few models in the code based off of real-world observations and theory. First we define a function to calculate the specific humidity above ice given a certain temperature:

```python
# Calculate q from T
def ice_q(T):
    e = 611*np.exp((Ls/R_v)*((1/273)-(1/T)))
    return (eps_R*e)/(p_a+e*(eps_R-1))
```

There is also a function to calculate the total shortwave flux via solar radiation. The sunrise and sunset values (0700 and 2200, respectively) were estimated from picking some arbitrary day in early April and looking at the sunrise and sunset times (via timeanddate.com)


The consants used may be found in the appendix.



