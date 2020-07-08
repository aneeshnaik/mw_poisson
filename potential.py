#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate gravitational potential associated with McMillan (2017) MW model.

Created: Tue Jun 30 09:41:12 2020
Author: A. P. Naik
Description: Using the disc decomposition + spherical harmonic expansion method
described in Dehnen and Binney (1998), solve for the gravitational potential
associated with the McMillan (2017) 4-component model of the Milky Way. This
essentially is a pure python version of McMillan's GalPot code.

In a nutshell, the potential for a given discoid mass distribution is
decomposed into an analytic disc component (here calculated by the
'potential_disc' function below), and another component given by solving
Poisson's equation for an 'effective' density. This effective density is much
less confined to the disc plane than the original density, so a spherical
harmonic method can be used much more inexpensively than if it were used on the
original mass distribution. The effective density is given by 'rho_effective'
below, and the spherical harmonic solver is in 'potential_sh'.
"""
import numpy as np
from .constants import kpc, pi, M_sun, G
from .util import print_progress
from .profiles import sigma, sigma_p, sigma_pp, zeta, bigH, bigH_p, rho_sph
from scipy.special import sph_harm
from scipy.interpolate import RectBivariateSpline as RBS
import sys


def potential_disc(pos, ndiscs, dpars):
    """Calculate analytic disc-plane component of the potential."""
    # get z and spherical radius r from pos
    z = pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)

    # loop over stellar and gas discs
    phi_d = np.zeros_like(z)
    for i in range(ndiscs):
        phi_d += 4 * pi * G * sigma(r, **dpars[i]) * bigH(z, **dpars[i])
    return phi_d


def potential_sh(ndiscs, dpars, nspheroids, spars, l_max, N_q, N_theta,
                 r_min, r_max, verbose=False):
    """Calculate potential using spherical harmonic expansion."""
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
    """Effective density to feed to spherical harmonic solver."""
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


def potential_solve(ndiscs, dpars, nspheroids, spars,
                    l_max=16, N_q=2001, N_theta=750,
                    r_min=1e-4 * kpc, r_max=1e+4 * kpc, verbose=False):

    # get spherical coords and potential from multipole expansion
    r, theta, pot_ME = potential_sh(ndiscs, dpars, nspheroids, spars,
                                    l_max, N_q, N_theta,
                                    r_min, r_max, verbose=verbose)

    # convert to Cartesians and add disc component
    r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
    x_grid = r_grid * np.sin(theta_grid)
    y_grid = np.zeros_like(x_grid)
    z_grid = r_grid * np.cos(theta_grid)
    pos_edge = np.stack((x_grid, y_grid, z_grid), axis=-1)
    pot = potential_disc(pos_edge, ndiscs, dpars) + pot_ME

    return r, theta, pot


def create_accel_fn(r, theta, pot):

    q = np.log(r)

    # first derivs via finite differencing
    dpdq = np.diff(pot, axis=0) / np.diff(q)[:, None]
    dpdth = np.diff(pot, axis=1) / np.diff(theta)[None, :]

    # average derivs in 'other' dimension to make appropriate grid shapes
    dpdq = 0.5 * (dpdq[:, 1:] + dpdq[:, :-1])
    dpdth = 0.5 * (dpdth[1:, :] + dpdth[:-1, :])

    # cell centre coordinates
    q_cen = 0.5 * (q[1:] + q[:-1])
    th_cen = 0.5 * (theta[1:] + theta[:-1])
    q_g, theta_g = np.meshgrid(q_cen, th_cen, indexing='ij')
    r_g = np.exp(q_g)

    # convert derivs to cylindrical coords
    sth = np.sin(theta_g)
    cth = np.cos(theta_g)
    dpdR = (dpdq * sth + dpdth * cth) / r_g
    dpdz = (dpdq * cth - dpdth * sth) / r_g

    # interpolators for cylindrical coords
    fR = RBS(q_cen, th_cen, -dpdR)
    fz = RBS(q_cen, th_cen, -dpdz)

    def accel(pos):
        r = np.linalg.norm(pos, axis=-1)
        q = np.log(r)
        theta = np.arccos(pos[..., 2] / r)
        az = fz.ev(q, theta)

        x = pos[..., 0]
        y = pos[..., 1]
        R = np.sqrt(x**2 + y**2)
        aR = fR.ev(q, theta)
        ax = aR * x / R
        ay = aR * y / R

        a = np.stack((ax, ay, az), axis=-1)
        return a

    return accel
