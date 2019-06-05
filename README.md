# Heat Equation Solver in Sea Ice

## Overview

This repository is designed to hold all of the codes for solving one-dimensional the heat equation numerically in a block of sea ice floating on sea water. Eventually, this code will be coupled with the Large Eddy Simulation (LES) code that is being used in the Environmental Fluid Mechanics laboratory at Princeton University.

Collaborators:
- Joseph J. Fogarty
- Elie Bou-Zeid

## Physical System

We now describe our physical system and problem statement. We consider a block of sea ice floating in the Arctic ocean. If we consider the temperature $$u$$ to be constant in the horizontal, then we are able to solve the following 1D system in the vertical for the temperature of the ice, $u$:

                                                         2                            
                              partial u           partial  u                          
begin{equation}begin{aligned} --------- & = alpha ---------- end{aligned}end{equation}
                              partial t                    2                          
                                                  partial x                           



and

\begin{equation}
\alpha=\frac{k}{\rho c_{p}}.
\end{equation}

where $\alpha$ is the thermal diffusivity in \si{\square\meter\per\second}, $k$ is the thermal conductivity in \si{\watt\per\metre\per\kelvin}, $\rho$ is the density in \si{\kilo\gram\per\cubic\meter}, and $c_{p}$ is the specific heat at constant pressure in \si{\joule\per\kilo\gram\per\kelvin}. For values these values that correspond to ice, we calculate $\alpha$:

\begin{equation}
\alpha = \frac{k}{\rho c_{p}} = \frac{2.25 \si{\watt\per\kelvin\per\meter}}{\left(916.7 \si{\kilo\gram\per\cubic\meter}\right)\left(2027 \si{\joule\per\kilo\gram\per\kelvin}\right)}\approx1.211\times10^{-6}.
\end{equation}

We will see later on that this small $\alpha$ allows for higher values of $dt$.
Note that even though this is being solved for in the vertical, we will consider the domain of ice as in the $x$-direction, where $x\in [0,L]$. $x=0$ will be the surface of the ice (the end exposed to the air), and $x=L$ will be the bottom of the ice (the end that is underwater).



