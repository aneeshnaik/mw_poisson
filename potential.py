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
from constants import pc, kpc, pi, M_sun, G, Myr
from scipy.special import sph_harm
from scipy.interpolate import RectBivariateSpline as RBS
import matplotlib.pyplot as plt


def sech(x):
    """Hyperbolic sech function."""
    return 1 / np.cosh(x)


def rho_bulge(pos):
    """
    Galactic bulge density.

    Calculate density of axisymmetric Bissantz-Gerhard bulge, using parameters
    from McMillan (2017). The functional form is Eq. (1) in McMillan (2017).

    Parameters
    ----------
    pos : array-like, shape (..., 3)
        Positions at which to evaluate density. UNITS: metres.
    """
    # get z and cylindrical radius from pos
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    R = np.sqrt(x**2 + y**2)

    # parameter values from McMillan (2017)
    q = 0.5
    alpha = 1.8
    r_0 = 0.075 * kpc
    r_cut = 2.1 * kpc
    rho_0 = 98.4 * (M_sun / pc**3)

    # calculate density
    rp = np.sqrt(R**2 + (z / q)**2)
    rho = (rho_0 / (1 + rp / r_0)**alpha) * np.exp(-(rp / r_cut)**2)
    return rho


def rho_halo(pos):
    """
    Galactic bulge density.

    Calculate density of NFW halo, using parameters from McMillan (2017). The
    functional form is Eq. (5) in McMillan (2017).

    Parameters
    ----------
    pos : array-like, shape (..., 3)
        Positions at which to evaluate density. UNITS: metres.
    """
    # get spherical radius from pos
    r = np.linalg.norm(pos, axis=-1)

    # parameters from McMillan (2017)
    r_h = 19.6 * kpc
    rho_0 = 0.00854 * (M_sun / pc**3)

    # calculate density
    x = r / r_h
    rho = rho_0 / (x * (1 + x)**2)
    return rho


def sigma_expdisc(R, sigma_0, R_0):
    """Surface density of exponential disc."""
    sig = sigma_0 * np.exp(-R / R_0)
    return sig


def sigma_p_expdisc(R, sigma_0, R_0):
    """First derivative of exponential disc surface density."""
    const = -sigma_0 / R_0
    sig_p = const * np.exp(-R / R_0)
    return sig_p


def sigma_pp_expdisc(R, sigma_0, R_0):
    """Second derivative of exponential disc surface density."""
    const = sigma_0 / R_0**2
    sig_pp = const * np.exp(-R / R_0)
    return sig_pp


def zeta_expdisc(z, z_0):
    """Vertical density profile of exponential disc."""
    const = (1 / (2 * z_0))
    h = const * np.exp(-np.abs(z) / z_0)
    return h


def bigH_expdisc(z, z_0):
    """Equivalent to second integral of zeta_expdisc; 0 in disc-plane."""
    const = z_0 / 2
    x = np.abs(z) / z_0
    H = const * (np.exp(-x) - 1 + x)
    return H


def bigH_p_expdisc(z, z_0):
    """First derivative of bigH function above."""
    H_p = np.zeros_like(z)
    mask = z != 0
    zi = z[mask]
    H_p[mask] = 0.5 * (zi / np.abs(zi)) * (1 - np.exp(-np.abs(zi) / z_0))
    return H_p


def sigma_holedisc(R, sigma_0, R_0, R_h):
    """Surface density of exponential disc with central hole."""
    x = (R_h / R) + (R / R_0)
    sig = sigma_0 * np.exp(-x)
    return sig


def sigma_p_holedisc(R, sigma_0, R_0, R_h):
    """1st deriv. of surface density of exponential disc with central hole."""
    x = (R_h / R) + (R / R_0)
    sig = (sigma_0 / R) * np.exp(-x) * ((R_h / R) - (R / R_0))
    return sig


def sigma_pp_holedisc(R, sigma_0, R_0, R_h):
    """2nd deriv. of surface density of exponential disc with central hole."""
    x = (R_h / R) + (R / R_0)
    fac = (R_h / R)**2 - 2 * (R_h / R_0) + (R / R_0)**2 - 2 * (R_h / R)
    sig = (sigma_0 / R**2) * np.exp(-x) * fac
    return sig


def zeta_sechdisc(z, z_0):
    """Vertical density profile of sech^2 disc."""
    const = (1 / (4 * z_0))
    mask = np.abs(z) < (25 * z_0)
    h = np.zeros_like(z)
    h[mask] = const * sech(z[mask] / (2 * z_0))**2
    return h


def bigH_sechdisc(z, z_0):
    """Equivalent to second integral of zeta_sechdisc; 0 in disc-plane."""
    mask = np.abs(z) < (25 * z_0)
    H = z_0 * (np.abs(z) / (2 * z_0) - np.log(2))
    H[mask] = z_0 * np.log(np.cosh(z[mask] / (2 * z_0)))
    return H


def bigH_p_sechdisc(z, z_0):
    """First derivative of bigH function above."""
    H_p = 0.5 * np.tanh(z / (2 * z_0))
    return H_p


def potential_disc(pos):
    """Calculate analytic disc-plane component of the potential."""
    # get z and spherical radius r from pos
    z = pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)

    # loop over stellar and gas discs
    phi_d = np.zeros_like(z)
    for disc in ['st_thin', 'st_thick', 'HI', 'H2']:

        # for each component, choose appropriate functions and parameters
        if disc in ['st_thin', 'st_thick']:
            sigma = sigma_expdisc
            bigH = bigH_expdisc
            if disc == 'st_thin':
                z0 = 300 * pc
                dpars = {'sigma_0': 896 * (M_sun / pc**2),
                         'R_0': 2.5 * kpc}
            else:
                z0 = 900 * pc
                dpars = {'sigma_0': 183 * (M_sun / pc**2),
                         'R_0': 3.02 * kpc}
        else:
            sigma = sigma_holedisc
            bigH = bigH_sechdisc
            if disc == 'HI':
                z0 = 85 * pc
                dpars = {'sigma_0': 53.1 * (M_sun / pc**2),
                         'R_0': 7 * kpc, 'R_h': 4 * kpc}
            else:
                z0 = 45 * pc
                dpars = {'sigma_0': 2180 * (M_sun / pc**2),
                         'R_0': 1.5 * kpc, 'R_h': 12 * kpc}

        phi_d += 4 * pi * G * sigma(r, **dpars) * bigH(z, z0)
    return phi_d


def potential_sh(l_max, N_q, N_theta, N_phi, r_min, r_max, verbose=False):
    """Calculate potential using spherical harmonic expansion."""
    # convert distances to kpc to give more tractable numbers
    r_min /= kpc
    r_max /= kpc

    # create grids of q=ln r, theta, phi
    q_min = np.log(r_min)
    q_max = np.log(r_max)
    h_q = (q_max - q_min) / N_q
    h_theta = pi / N_theta
    h_phi = 2 * pi / N_phi
    q = np.linspace(q_min + 0.5 * h_q, q_max - 0.5 * h_q, N_q)
    r = np.exp(q)
    theta = np.linspace(0.5 * h_theta, pi - 0.5 * h_theta, N_theta)
    phi = np.linspace(0.5 * h_phi, 2 * pi - 0.5 * h_phi, N_phi)
    r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')

    # convert to Cartesian coords
    x_grid = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = r_grid * np.cos(theta_grid)
    pos_grid = np.stack((x_grid, y_grid, z_grid), axis=-1)

    # sample rho at these coords, and convert to Msun / kpc**3
    rho_grid = rho_effective(pos_grid * kpc)
    rho_grid *= (kpc**3 / M_sun)

    # loop over spherical harmonics
    if verbose:
        print("Performing harmonic expansion...")
    pot = np.zeros_like(rho_grid, dtype=np.complex128)
    for l in range(l_max + 1):
        for m in range(-l, l + 1, 1):
            if verbose:
                print(l, m)

            # get Ylm* at these coords
            Ystar_grid = sph_harm(m, l, phi_grid, theta_grid).conjugate()
            Y_grid = sph_harm(m, l, phi_grid, theta_grid)

            # perform theta/phi summation
            integrand = rho_grid * Ystar_grid * np.sin(theta_grid)
            S = np.sum(integrand, axis=(1, 2))

            # radial summation
            summand_int = np.exp((l + 3) * q) * S
            summand_ext = np.append((np.exp((-l + 2) * q) * S)[1:], 0)
            C_int = np.cumsum(summand_int)
            C_ext = np.flipud(np.cumsum(np.flipud(summand_ext)))
            C = h_theta * h_phi * h_q * (C_int / r**(l + 1) + r**l * C_ext)
            C = C.astype(complex)
            potlm = -4 * pi * G * C[:, None, None] * Y_grid / (2 * l + 1)
            pot += potlm

    # convert potential back to SI
    pot *= (M_sun / kpc)

    return r * kpc, theta, phi, pot.real


def rho_effective(pos):
    """Effective density to feed to spherical harmonic solver."""
    # get cylindrical and spherical radii from pos
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    R = np.sqrt(x**2 + y**2)
    r = np.sqrt(R**2 + z**2)

    # spheroidal densities
    rho = rho_bulge(pos) + rho_halo(pos)

    # add disc effective densities one by one: stellar and gas discs
    for disc in ['st_thin', 'st_thick', 'HI', 'H2']:

        # for each component, choose appropriate functions and parameters
        if disc in ['st_thin', 'st_thick']:
            sigma = sigma_expdisc
            sigma_p = sigma_p_expdisc
            sigma_pp = sigma_pp_expdisc
            zeta = zeta_expdisc
            bigH = bigH_expdisc
            bigH_p = bigH_p_expdisc
            if disc == 'st_thin':
                z0 = 300 * pc
                dpars = {'sigma_0': 896 * (M_sun / pc**2),
                         'R_0': 2.5 * kpc}
            else:
                z0 = 900 * pc
                dpars = {'sigma_0': 183 * (M_sun / pc**2),
                         'R_0': 3.02 * kpc}
        else:
            sigma = sigma_holedisc
            sigma_p = sigma_p_holedisc
            sigma_pp = sigma_pp_holedisc
            zeta = zeta_sechdisc
            bigH = bigH_sechdisc
            bigH_p = bigH_p_sechdisc
            if disc == 'HI':
                z0 = 85 * pc
                dpars = {'sigma_0': 53.1 * (M_sun / pc**2),
                         'R_0': 7 * kpc, 'R_h': 4 * kpc}
            else:
                z0 = 45 * pc
                dpars = {'sigma_0': 2180 * (M_sun / pc**2),
                         'R_0': 1.5 * kpc, 'R_h': 12 * kpc}

        # evaluate functions
        sigR = sigma(R, **dpars)
        sigr = sigma(r, **dpars)
        sig_pr = sigma_p(r, **dpars)
        sig_ppr = sigma_pp(r, **dpars)
        h = zeta(z, z0)
        H = bigH(z, z0)
        H_p = bigH_p(z, z0)

        # calculate effective density
        t1 = (sigR - sigr) * h
        t2 = -sig_ppr * H
        t3 = -(2 / r) * sig_pr * (H + z * H_p)
        rho += t1 + t2 + t3

    return rho


def potential(l_max=16, N_q=2001, N_theta=750, N_phi=20,
              r_min=1e-4 * kpc, r_max=1e+4 * kpc, verbose=False):

    # get spherical coords and potential from multipole expansion
    r, theta, phi, pot_ME = potential_sh(l_max, N_q, N_theta, N_phi,
                                         r_min, r_max, verbose=verbose)

    # convert to Cartesians and add disc component
    r_grid, theta_grid, phi_grid = np.meshgrid(r, theta, phi, indexing='ij')
    x_grid = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y_grid = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z_grid = r_grid * np.cos(theta_grid)
    pos_edge = np.stack((x_grid, y_grid, z_grid), axis=-1)
    pot = potential_disc(pos_edge) + pot_ME

    return r, theta, phi, pot


def create_accel_fn(r, theta, phi, pot):

    q = np.log(r)

    # azimuthally average the potential
    pot = np.average(pot, axis=2)

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


if __name__ == '__main__':
    
    l_max = 20
    r, theta, phi, pot_verbose = potential_sh(l_max=l_max, N_q=501, N_theta=250, N_phi=20,
                                              r_min=1e-4 * kpc, r_max=1e+4 * kpc, verbose=True)

    for i in range(l_max+1):
        print(i, np.max(np.abs(pot_verbose[i]/pot_verbose[-1] - 1)))

