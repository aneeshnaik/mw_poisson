#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example of a runscript for mw_poisson.

Created: Sat Jul  4 17:51:57 2020
Author: A. P. Naik
"""
import numpy as np
from mw_poisson import potential_solve
from mw_poisson.constants import kpc


# parameters for spherical harmonic expansion
l_max = 80
N_q = 2001
N_theta = 2500
r_min = 1e-4 * kpc
r_max = 1e+4 * kpc

# disc and spheroid parameters from McMillan (2017)
d1 = {'sigma_0': 895.679 * (M_sun / pc**2), 'R_0': 2.49955 * kpc, 'R_h': 0,
      'z_0': 300 * pc, 'form': 'exponential'}
d2 = {'sigma_0': 183.444 * (M_sun / pc**2), 'R_0': 3.02134 * kpc, 'R_h': 0,
      'z_0': 900 * pc, 'form': 'exponential'}
d3 = {'sigma_0': 53.1319 * (M_sun / pc**2), 'R_0': 7 * kpc, 'R_h': 4 * kpc,
      'z_0': 85 * pc, 'form': 'sech'}
d4 = {'sigma_0': 2179.95 * (M_sun / pc**2), 'R_0': 1.5 * kpc, 'R_h': 12 * kpc,
      'z_0': 45 * pc, 'form': 'sech'}
s1 = {'alpha': 1.8, 'beta': 0, 'q': 0.5, 'r_cut': 2.1 * kpc,
      'r_0': 0.075 * kpc, 'rho_0': 98.351 * (M_sun / pc**3)}
s2 = {'alpha': 2, 'beta': 1, 'q': 1, 'r_cut': np.inf,
      'r_0': 19.5725 * kpc, 'rho_0': 0.00853702 * (M_sun / pc**3)}

# output filename
filename = 'EXAMPLE'

# solve for potential
r, theta, pot = potential_solve(ndiscs=4, dpars=[d1, d2, d3, d4],
                                nspheroids=2, spars=[s1,s2],
                                l_max=l_max, N_q=N_q, N_theta=N_theta,
                                r_min=r_min, r_max=r_max, verbose=True)

# save
np.savez(filename, r=r, theta=theta, pot=pot)
