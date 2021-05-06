#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MilkyWay class is main object of mw_poisson.

See README for further details about mw_poisson and usage examples.

Created: July 2020
Author: A. P. Naik
"""
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI, interp1d
from .constants import kpc, pi, G
from .potential import potential_disc, potential_sh
from .profiles import rho_sph, zeta, sigma


class MilkyWay:
    """
    Main object of mw_poisson package.

    For a given input density profile, the 'solve_potential' method solves for
    the gravitational potential on a spherical grid. The potential and
    acceleration at any given position can then be interpolated from this grid.
    Also there are various utility functions such as density at a given point
    or total mass enclosed by a given radius.

    See README for examples of usage.

    Parameters
    ----------
    ndiscs : int
        Number of disc components in galaxy.
    dpars : list of dicts, length ndiscs
        Each dict should contain 5 items, as specified in 'Disc Parameters'
        below.
    nspheroids : int
        Number of spheroidal components in galaxy.
    spars : list of dicts
        Each dict should contain 6 items, as specified in 'Spheroid Parameters'
        below.
    r_min : float, optional
        Inner limit of radial integration and various interpolators. The
        default is 1e-4 * kpc. UNITS: m
    r_max : float, optional
        Outer limit of radial integration and various interpolators. The
        default is 1e+4 * kpc. UNITS: m

    Disc Parameters
    ---------------
    sigma_0 : float
        Density normalisation. UNITS: kg/m^2
    R_0 : float
        Scale radius. UNITS: m
    R_h : float
        Hole radius. UNITS: m
    z_0 : float
        Scale height of disc. UNITS: m
    form : string, {'exponential','sech'}
        Scale density. UNITS: kg / m^3

    Spheroid Parameters
    -------------------
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

    Methods
    -------
    solve_potential(l_max, N_q, N_theta, r_min, r_max, verbose)
        For density profile specified by the spheroid and disc parameters,
        solve for the gravitational potential.
    potential(pos)
        Interpolate potential at given positions.
    acceleration(pos)
        Interpolate acceleration at given positions.
    density(pos)
        Evaluate density at given positions.
    mass_enclosed(pos)
        Evaluate enclosed spherical mass at given positions.
    """

    def __init__(self, ndiscs, dpars, nspheroids, spars,
                 r_min=1e-4 * kpc, r_max=1e+4 * kpc):
        self.ndiscs = ndiscs
        self.dpars = dpars
        self.nspheroids = nspheroids
        self.spars = spars
        self.r_min = r_min
        self.r_max = r_max
        self.SolnFlag = False
        self.__create_mass_interpolator()
        return

    def solve_potential(self, l_max=80, N_q=4001, N_theta=2500, verbose=False):
        """
        Solve for the gravitational potential.

        Parameters
        ----------
        l_max : int, optional
            Multipole at which to truncate spherical harmonic expansion. The
            default is 80.
        N_q : int, optional
            Number of (log-spaced) radial bins for radial integration. The
            default is 2001.
        N_theta : int, optional
            Number of polar angular bins for angular integration. Needs to be
            an even number so that acceleration is evaluated on disc-plane.
            The default is 2500.
        verbose : bool, optional
            Whether to print progress bar in spherical harmonic expansion. The
            default is False.

        Returns
        -------
        None.
        """
        # insist N_theta is even, so acceleration evaluated on midplane
        assert N_theta % 2 == 0, "Need N_theta to be even"

        # get spherical coords and potential from multipole expansion
        r, theta, pot_ME = potential_sh(self.ndiscs, self.dpars,
                                        self.nspheroids, self.spars,
                                        l_max, N_q, N_theta,
                                        self.r_min, self.r_max,
                                        verbose=verbose)

        # convert to Cartesians
        r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
        x_grid = r_grid * np.sin(theta_grid)
        y_grid = np.zeros_like(x_grid)
        z_grid = r_grid * np.cos(theta_grid)
        pos = np.stack((x_grid, y_grid, z_grid), axis=-1)

        # add analytic disc component (theory.pdf eq.6)
        pot = potential_disc(pos, self.ndiscs, self.dpars) + pot_ME

        # create interpolating functions for potential and acceleration
        self.__create_potacc_interpolators(r, theta, pot)

        # flag indicates that solution for potential has been found
        self.SolnFlag = True

        return

    def __create_mass_interpolator(self):
        """Set up self.__f_lnmass function, which interpolates mass.

        Note: Inside self.r_min M_enc=0, outside self.r_max, M_enc=M_enc(r_max)
        """
        N_q = 2000  # number of cells in radial dimension
        N_th = 2000  # number of cells in theta dimension

        # set up r grid
        q_min = np.log(self.r_min)
        q_max = np.log(self.r_max)
        q_edges = np.linspace(q_min, q_max, num=N_q + 1)
        q_cen = 0.5 * (q_edges[1:] + q_edges[:-1])
        dq = np.diff(q_edges)[0]

        # set up theta grid
        th_edges = np.linspace(0, pi / 2, num=N_th + 1)
        th_cen = 0.5 * (th_edges[1:] + th_edges[:-1])
        dth = np.diff(th_edges)[0]

        # calculate density on grid
        r_g, th_g = np.meshgrid(np.exp(q_cen), th_cen, indexing='ij')
        x = r_g * np.sin(th_g)
        y = np.zeros_like(x)
        z = r_g * np.cos(th_g)
        pos = np.stack((x, y, z), axis=-1)
        rho = self.density(pos)

        # integrate
        const = dq * dth * 4 * pi
        dM = rho * np.sin(th_g) * r_g**3 * const
        dMshell = np.sum(dM, axis=1)
        lnM_enc = np.log(np.cumsum(dMshell))

        self.__f_lnmass = interp1d(q_edges[1:], lnM_enc, bounds_error=False,
                                   fill_value=(-np.inf, lnM_enc[-1]))
        self.M_max = np.exp(lnM_enc[-1])

        return

    def __create_potacc_interpolators(self, r, theta, pot):
        """Set up interpolators for potential and acceleration."""
        q = np.log(r)
        self.__f_pot = RGI((q, theta), pot)

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

        # on disc plane, set z gradient to zero exactly
        i = th_cen.size // 2
        dpdz[:, i] = 0

        # add values on polar axis
        Nq = q_cen.size
        dpdR_new = np.hstack((np.zeros((Nq, 1)), dpdR, np.zeros((Nq, 1))))
        dpdz_ax = np.sqrt(dpdR[:, 0]**2 + dpdz[:, 0]**2)[:, None]
        dpdz_new = np.hstack((dpdz_ax, dpdz, -dpdz_ax))
        th_cen = np.hstack((0, th_cen, pi))

        # interpolators for cylindrical coords
        self.__f_aR = RGI((q_cen, th_cen), -dpdR_new)
        self.__f_az = RGI((q_cen, th_cen), -dpdz_new)

        return

    def potential(self, pos):
        """
        Interpolate potential at given positions.

        If r > self.r_max (1e+4 kpc by default) than GM/r law is assumed, where
        M is M(r_max). If r < self.r_min (1e+4 kpc by default) then
        pot = pot(r_min).

        Parameters
        ----------
        pos : array, shape (3) or (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate potential, in 3D Galactocentric
            Cartesian coordinates. UNITS: m

        Returns
        -------
        pot : float or array, shape (N) or (N1, N2, N3, ...)
            Potentials at given positionss. UNITS: m^2 / s^2
        """
        # check potential solution exists
        assert self.SolnFlag

        # get spherical coordinates from pos
        r = np.linalg.norm(pos, axis=-1)
        q = np.log(r)
        theta = np.arccos(pos[..., 2] / r)

        # interpolate potential
        pot = self.__f_pot.ev(q, theta)

        # if only one position supplied, return float, not array
        if pos.ndim == 1:
            pot = float(pot)

        # extrapolate GM/r law beyond self.r_max
        M = self.M_max
        if pos.ndim == 1:
            if r > self.r_max:
                pot += G * M / self.r_max - G * M / r
        else:
            mask = r > self.r_max
            pot[mask] += G * M / self.r_max - G * M / r[mask]

        return pot

    def acceleration(self, pos):
        """
        Interpolate acceleration at given positions.

        if r > self.r_max (1e+4 kpc by default) than GM/r^2 law is assumed,
        where M is M(r_max). If r < self.r_min (1e+4 kpc by default) then a=0.

        Parameters
        ----------
        pos : array, shape (3) or (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate acceleration, in 3D Galactocentric
            Cartesian coordinates. UNITS: m

        Returns
        -------
        acc : array, shape (3) or (N, 3) or (N1, N2, N3, ..., 3)
            Accelerations at given positions, in 3D Galactocentric
            Cartesian coordinates. UNITS: m / s^2
        """
        # check potential solution exists
        assert self.SolnFlag

        # get spherical coordinates from pos
        r = np.linalg.norm(pos, axis=-1)
        q = np.log(r)
        theta = np.arccos(pos[..., 2] / r)
        xi = np.stack((q, theta), axis=-1)

        # interpolate accels
        az = self.__f_az(xi)
        aR = self.__f_aR(xi)

        # recast into x and y accels (zero when R=0)
        x = pos[..., 0]
        y = pos[..., 1]
        R = np.sqrt(x**2 + y**2)
        ax = np.zeros_like(aR)
        ay = np.zeros_like(aR)
        ax[R != 0] = aR[R != 0] * x[R != 0] / R[R != 0]
        ay[R != 0] = aR[R != 0] * y[R != 0] / R[R != 0]

        # stack ax ay az
        a = np.stack((ax, ay, az), axis=-1)

        # extrapolate GM/r^2 law beyond self.r_max
        M = self.M_max
        if pos.ndim == 1:
            if r > self.r_max:
                a = - G * M * pos / r**3
        else:
            mask = r > self.r_max
            a[mask] = - G * M * pos[mask] / r[mask, None]**3

        return a

    def density(self, pos, eps=0):
        """
        Evaluate density at given positions.

        Parameters
        ----------
        pos : array, shape (3) or (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate density, in 3D Galactocentric
            Cartesian coordinates. UNITS: m
        eps : float
            Gravitational softening. Default is 0. UNITS: m

        Returns
        -------
        rho : float or array, shape (N) or (N1, N2, N3, ...)
            Densities at given positions. UNITS: kg/m^3
        """
        # get cylindrical and spherical radii from pos
        x = pos[..., 0]
        y = pos[..., 1]
        z = pos[..., 2]
        R = np.sqrt(x**2 + y**2)

        # loop over spheroids and add densities
        rho = np.zeros_like(R)
        for i in range(self.nspheroids):
            rho += rho_sph(pos, eps=eps, **self.spars[i])

        # loop over discs and add densities
        for i in range(self.ndiscs):
            rho += sigma(R, eps=eps, **self.dpars[i]) * zeta(z, eps=eps, **self.dpars[i])

        # if only one position supplied, return float, not array
        if pos.ndim == 1:
            rho = float(rho)

        return rho

    def mass_enclosed(self, pos):
        """
        Evaluate enclosed spherical mass at given positions.

        Note: Inside self.r_min M_enc=0, outside self.r_max, M_enc=M_enc(r_max)

        Parameters
        ----------
        pos : array, shape (3) or (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate mass, in 3D Galactocentric
            Cartesian coordinates. UNITS: m

        Returns
        -------
        mass : float or array, shape (N) or (N1, N2, N3, ...)
            Enclosed spherical mass at given positions. UNITS: kg
        """
        r = np.linalg.norm(pos, axis=-1)
        q = np.log(r)
        mass = np.exp(self.__f_lnmass(q))
        return mass
