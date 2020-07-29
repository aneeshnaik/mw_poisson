#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various functions used to solve Poisson's equation.

See README for further details about mw_poisson and usage examples.

Created: July 2020
Author: A. P. Naik
"""
import numpy as np
from .constants import kpc, pi, M_sun, G
from .util import print_progress
from .profiles import sigma, sigma_p, sigma_pp, zeta, bigH, bigH_p, rho_sph
from scipy.special import sph_harm


def potential_disc(pos, ndiscs, dpars):
    """
    Calculate analytic disc-plane component of the potential (theory.pdf eq.6).

    Parameters
    ----------
    pos : array, shape (N, 3) or (N1, N2, N3, ..., 3)
        Positions at which to evaluate potential, in 3D Galactocentric
        Cartesian coordinates. UNITS: m
    ndiscs : int
        Number of disc components in galaxy.
    dpars : list of dicts, length ndiscs
        See 'MilkyWay' class documentation for more info about dpars.

    Returns
    -------
    phi_d : array, shape (N) or (N1, N2, N3, ...)
        Potentials at given positionss. UNITS: m^2 / s^2

    """
    # get z and spherical radius r from pos
    z = pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)

    # loop over stellar and gas discs
    phi_d = np.zeros_like(z)
    for i in range(ndiscs):
        phi_d += 4 * pi * G * sigma(r, **dpars[i]) * bigH(z, **dpars[i])
    return phi_d


def potential_sh(ndiscs, dpars, nspheroids, spars, l_max, N_q, N_theta,
                 r_min, r_max, verbose):
    """
    Calculate potential using spherical harmonic expansion (theory.pdf eq.13).

    Parameters
    ----------
    ndiscs : int
        Number of disc components in galaxy.
    dpars : list of dicts, length ndiscs
        See 'MilkyWay' class documentation for more info about dpars.
    nspheroids : int
        Number of spheroidal components in galaxy.
    spars : list of dicts
        See 'MilkyWay' class documentation for more info about dpars.

    The remaining parameters are exactly those of the MilkyWay.solve_potential
    class method; see documentation there for more info about the parameters.

    Returns
    -------
    r : array, shape (N_q)
        Array of spherical radii at which potential is calculated. UNITS: m
    theta : array, shape (N_theta)
        Array of polar angles at which potential is calculated. UNITS: radians
    pot : array, shape (N_q, N_theta)
        Potential obtained from spherical harmonic expansion, on a spherical
        grid. UNITS: m^2/s^2

    """
    # convert distances to kpc to give more tractable numbers
    r_min /= kpc
    r_max /= kpc

    # create grids of q=ln r, theta
    q_min = np.log(r_min)
    q_max = np.log(r_max)
    h_q = (q_max - q_min) / N_q
    h_theta = pi / N_theta
    q = np.linspace(q_min + 0.5 * h_q, q_max - 0.5 * h_q, N_q)
    r = np.exp(q)
    theta = np.linspace(0.5 * h_theta, pi - 0.5 * h_theta, N_theta)
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
    sth = np.sin(theta_grid)
    cth = np.cos(theta_grid)

    # convert to Cartesian coords
    x_grid = r_grid * sth
    y_grid = np.zeros_like(x_grid)
    z_grid = r_grid * cth
    pos_grid = np.stack((x_grid, y_grid, z_grid), axis=-1)

    # sample rho at these coords, and convert to Msun / kpc**3
    rho_grid = rho_effective(pos_grid * kpc, ndiscs, dpars, nspheroids, spars)
    rho_grid *= (kpc**3 / M_sun)
    rhosth = rho_grid * sth

    # loop over spherical harmonics
    if verbose:
        print("Performing harmonic expansion...")
    pot = np.zeros_like(rho_grid)
    for l in range(l_max + 1):
        if verbose:
            print_progress(l, l_max + 1, interval=1)

        # get Ylm* at these coords
        Y_grid = sph_harm(0, l, 0, theta_grid[0])[None, ...]
        Ystar_grid = Y_grid.conjugate()

        # perform theta summation
        integrand = 2 * pi * rhosth * Ystar_grid
        S = np.sum(integrand, axis=1)

        # radial summation
        rl = np.exp(l * q, dtype=np.float128)
        rlp1 = np.exp((l + 1) * q, dtype=np.float128)
        s_int = np.exp((l + 3) * q, dtype=np.float128) * S
        s_ext = np.append((np.exp((-l + 2) * q, dtype=np.float128) * S)[1:], 0)
        C_int = np.cumsum(s_int)
        C_ext = np.flipud(np.cumsum(np.flipud(s_ext)))
        C = h_theta * h_q * (C_int / rlp1 + rl * C_ext)
        C = C.astype(complex)
        potlm = -4 * pi * G * C[:, None] * Y_grid / (2 * l + 1)
        pot += potlm.real

    # convert potential and radii back to SI
    pot *= (M_sun / kpc)
    r *= kpc

    return r, theta, pot


def rho_effective(pos, ndiscs, dpars, nspheroids, spars):
    """
    Calculate effective density SH solver (theory.pdf eq.8).

    Parameters
    ----------
    pos : array, shape (N, 3) or (N1, N2, N3, ..., 3)
        3D Cartesian positions at which to evaluate density. UNITS: m
    ndiscs : int
        Number of disc components in galaxy.
    dpars : list of dicts, length ndiscs
        See 'MilkyWay' class documentation for more info about dpars.
    nspheroids : int
        Number of spheroidal components in galaxy.
    spars : list of dicts
        See 'MilkyWay' class documentation for more info about dpars.

    Returns
    -------
    rho : array, shape (N) or (N1, N2, N3, ...)
        Effective density at given positions. UNITS: kg / m^3
    """
    # get cylindrical and spherical radii from pos
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(R**2 + z**2)

    # loop over spheroids and add actual densities
    rho = np.zeros_like(r)
    for i in range(nspheroids):
        rho += rho_sph(pos, **spars[i])

    # loop over discs and add effective densities
    for i in range(ndiscs):

        # evaluate functions
        sigR = sigma(R, **dpars[i])
        sigr = sigma(r, **dpars[i])
        sig_pr = sigma_p(r, **dpars[i])
        sig_ppr = sigma_pp(r, **dpars[i])
        h = zeta(z, **dpars[i])
        H = bigH(z, **dpars[i])
        H_p = bigH_p(z, **dpars[i])

        # calculate effective density
        t1 = (sigR - sigr) * h
        t2 = -sig_ppr * H
        t3 = -(2 / r) * sig_pr * (H + z * H_p)
        rho += t1 + t2 + t3

    return rho
