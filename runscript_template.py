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


# run parameters
l_max = 6
N_q = 501
N_theta = 200
N_phi = 20
r_min = 1e-4 * kpc
r_max = 1e+4 * kpc

# output filename
filename = 'EXAMPLE'

r, theta, phi, pot = potential_solve(l_max=l_max, N_q=N_q,
                                     N_theta=N_theta, N_phi=N_phi,
                                     r_min=r_min, r_max=r_max, verbose=True)
np.savez(filename, r=r, theta=theta, phi=phi, pot=pot)
