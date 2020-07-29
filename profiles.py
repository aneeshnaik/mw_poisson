#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various density functions.

See README for further details about mw_poisson and usage examples.

Created: July 2020
Author: A. P. Naik
"""
import numpy as np
from .util import sech


def rho_sph(pos, alpha, beta, q, r_cut, r_0, rho_0):
    """
    Density of spheroid (theory.pdf eq.15).

    Parameters
    ----------
    pos : array, shape (N, 3) or (N1, N2, N3, ..., 3)
        3D Cartesian positions at which to evaluate density. UNITS: m
    alpha : float
        Outer slope.
    beta : float
        Inner slope.
    q : float
        Flattening
    r_cut : float
        Exponential cutoff radius. UNITS: m
    r_0 : float
        Scale radius. UNITS: m
    rho_0 : float
        Scale density. UNITS: kg / m^3

    Returns
    -------
    rho : array, shape (N) or (N1, N2, N3, ...)
        Density at given positions. UNITS: kg / m^3
    """
    # get flattened radius coordinate
    x = pos[..., 0]
    y = pos[..., 1]
    z = pos[..., 2]
    R = np.sqrt(x**2 + y**2)
    rp = np.sqrt(R**2 + (z / q)**2)

    denom = (rp / r_0)**beta * (1 + rp / r_0)**alpha
    rho = (rho_0 / denom) * np.exp(-(rp / r_cut)**2)
    return rho


def zeta(z, z_0, form, **kwargs):
    """
    Normalised vertical density of disc (theory.pdf eq.19 or 22).

    Parameters
    ----------
    z : array, shape (N) or (N1, N2, N3, ...)
        Heights at which to evaluate density. UNITS: m
    z_0 : float
        Scale height of disc. UNITS: m
    form : string, {'exponential','sech'}
        Whether to use exponential or sech^2 vertical profile (see theory.pdf).

    Returns
    -------
    h : array, shape (N) or (N1, N2, N3, ...)
        Density at given scale height, normalised such that integral over z
        equals 1. UNITS: m^-1
    """
    if form == 'exponential':
        const = (1 / (2 * z_0))
        h = const * np.exp(-np.abs(z) / z_0)
    elif form == 'sech':
        const = (1 / (4 * z_0))
        mask = np.abs(z) < (25 * z_0)
        h = np.zeros_like(z)
        h[mask] = const * sech(z[mask] / (2 * z_0))**2
    else:
        assert False
    return h


def bigH(z, z_0, form, **kwargs):
    """
    Second integral of zeta function above (theory.pdf eq.20 or 23).

    Parameters
    ----------
    z : array, shape (N) or (N1, N2, N3, ...)
        Heights at which to evaluate density. UNITS: m
    z_0 : float
        Scale height of disc. UNITS: m
    form : string, {'exponential','sech'}
        Whether to use exponential or sech^2 vertical profile (see theory.pdf).

    Returns
    -------
    H : array, shape (N) or (N1, N2, N3, ...)
        H function at given scale height. UNITS: m
    """
    if form == 'exponential':
        const = z_0 / 2
        x = np.abs(z) / z_0
        H = const * (np.exp(-x) - 1 + x)
    elif form == 'sech':
        mask = np.abs(z) < (25 * z_0)
        H = z_0 * (np.abs(z) / (2 * z_0) - np.log(2))
        H[mask] = z_0 * np.log(np.cosh(z[mask] / (2 * z_0)))
    else:
        assert False
    return H


def bigH_p(z, z_0, form, **kwargs):
    """
    First derivative of bigH function above (theory.pdf eq.21 or 24).

    Parameters
    ----------
    z : array, shape (N) or (N1, N2, N3, ...)
        Heights at which to evaluate. UNITS: m
    z_0 : float
        Scale height of disc. UNITS: m
    form : string, {'exponential','sech'}
        Whether to use exponential or sech^2 vertical profile (see theory.pdf).

    Returns
    -------
    H_p : array, shape (N) or (N1, N2, N3, ...)
        H_p function at given scale height. UNITS: dimensionless
    """
    if form == 'exponential':
        H_p = np.zeros_like(z)
        mask = z != 0
        zi = z[mask]
        H_p[mask] = 0.5 * (zi / np.abs(zi)) * (1 - np.exp(-np.abs(zi) / z_0))
    elif form == 'sech':
        H_p = 0.5 * np.tanh(z / (2 * z_0))
    else:
        assert False
    return H_p


def sigma(R, sigma_0, R_0, R_h, **kwargs):
    """
    Surface density of exponential disc with central hole (theory.pdf eq.16).

    Parameters
    ----------
    R : array, shape (N) or (N1, N2, N3, ...)
        Cylindrical radii at which to evaluate.
    sigma_0 : float
        Density normalisation. UNITS: kg/m^2
    R_0 : float
        Scale radius. UNITS: m
    R_h : float
        Hole radius. UNITS: m

    Returns
    -------
    sig : array, shape (N) or (N1, N2, N3, ...)
        Surface density of disc at given radii. UNITS: kg/m^2
    """
    x = (R_h / R) + (R / R_0)
    sig = sigma_0 * np.exp(-x)
    return sig


def sigma_p(R, sigma_0, R_0, R_h, **kwargs):
    """
    First derivative of sigma function above (theory.pdf eq.17).

    Parameters
    ----------
    R : array, shape (N) or (N1, N2, N3, ...)
        Cylindrical radii at which to evaluate.
    sigma_0 : float
        Density normalisation. UNITS: kg/m^2
    R_0 : float
        Scale radius. UNITS: m
    R_h : float
        Hole radius. UNITS: m

    Returns
    -------
    sig_p : array, shape (N) or (N1, N2, N3, ...)
        d(sigma)/dR at given radii. UNITS: kg/m^3
    """
    x = (R_h / R) + (R / R_0)
    sig = -(sigma_0 / R_0) * np.exp(-x) * (1 - (R_h * R_0 / R**2))
    return sig


def sigma_pp(R, sigma_0, R_0, R_h, **kwargs):
    """
    Second derivative of sigma function above (theory.pdf eq.18).

    Parameters
    ----------
    R : array, shape (N) or (N1, N2, N3, ...)
        Cylindrical radii at which to evaluate.
    sigma_0 : float
        Density normalisation. UNITS: kg/m^2
    R_0 : float
        Scale radius. UNITS: m
    R_h : float
        Hole radius. UNITS: m

    Returns
    -------
    sig_pp : array, shape (N) or (N1, N2, N3, ...)
        d^2(sigma)/dR^2 at given radii. UNITS: kg/m^4
    """
    x = (R_h / R) + (R / R_0)
    fac = (1 - R_h * R_0 / R**2)**2 - 2 * R_h * R_0**2 / R**3
    sig = (sigma_0 / R_0**2) * np.exp(-x) * fac
    return sig
