! 1D crank-nicolson solver using FORTRAN
! Ported from original iceheat.py code
implicit none

! define constants
REAL :: alpha=0.6, eps_ice=0.96, sigma=5.67E-8, lw_in=200.0
! albedo, ice emissivity, S-B constant W/(m^2K^4), LW_in (W/m^2)
REAL :: rho_w=1020.0, rho_a=1.225, rho_i=916.7, V_a=5.0, V_w=1.0
! seawater, air, & ice density (kg/m^3), air & water velocity (m/s)
REAL :: u_star_top=0.2, u_star_bottom=0.1, T_w=272.15
! air & water friction velocity (m/s), water temp (K)
REAL ::  c_pa=1004.0, c_pw=3985.0, c_pi=2027.0
! specific heats of air, water, ice (J/(kgK))
REAL :: c_h=0.01737, c_e=0.01737
! bulk transfer coefficient for latent & sensible heat
REAL :: Lv=2500000.0, Lf=334000.0, Ls=Lv+Lf
! latent heat of vaporization, fusion, and sublimation (J/kg)
REAL :: p_a = 101325.0, R_d=287.0, R_v=461.5
! air pressure (Pa), dry air and water vapor gas constants (J/(kgK))
REAL ::  eps_R= R_d/R_v, kappa_ice=2.25
! ratio of gas constants, thermal conductivity of ice (W/(mK))
REAL :: q_a=0.0018, alpha_ice = kappa_ice/(c_pi*rho_i)
! specific humidity of air (kg_w/kg_a), thermal diffusivity of ice (m^2/s)





!functions--------------
! ice_q - calculates q from T over ice
program testing_functions
!This program test subroutines
implicit none
real ice_q
print *, "Do these subroutines work?"
print *, "The humidity over ice for T=273 is", ice_q(273.0)
print *, "Bye!"
end

real function ice_q(T)
  implicit none
  REAL :: T, ei, q, Ai=3410000000000, Bi=6130, eps_R=
  ei = Ai*exp((-1.0*Bi)/T)
  q = (eps_R*e)/(p_a*(eps_R-1))
  return
end




!--------------subroutines--------------
! ice_q - calculates q from T over ice
subroutine ice_q(T)
  implicit none
  REAL :: T, e, Ls, R_v, eps_R, p_a

  e = 611*exp((Ls/R_v)*((1/273)-(1/T))
  q = (eps_R*e)/(p_a*(eps_R-1))

  return
end

!sw_net(t) - calculates the total incoming shortwave
!-----------
subroutine sw_net(t)
  implicit none
  real :: t_hours = mod(t/3600.0,24)
