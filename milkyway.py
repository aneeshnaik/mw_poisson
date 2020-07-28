#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MilkyWay class is main object of mw_poisson.

See README for further details about mw_poisson and usage examples.

Created: July 2020
Author: A. P. Naik
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS, interp1d
from .constants import kpc, pi
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

    def __init__(self, ndiscs, dpars, nspheroids, spars):
        self.ndiscs = ndiscs
        self.dpars = dpars
        self.nspheroids = nspheroids
        self.spars = spars
        self.SolnFlag = False
        self.__create_mass_interpolator()
        return

    def solve_potential(self, l_max=80, N_q=2001, N_theta=2500,
                        r_min=1e-4 * kpc, r_max=1e+4 * kpc, verbose=False):
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
        r_min : float, optional
            Inner limit of radial integration. The default is 1e-4 * kpc.
            UNITS: m
        r_max : float, optional
            Outer limit of radial integration. The default is 1e+4 * kpc.
            UNITS: m
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
                                        r_min, r_max, verbose=verbose)

        # convert to Cartesians and add analytic disc component
        r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
        x_grid = r_grid * np.sin(theta_grid)
        y_grid = np.zeros_like(x_grid)
        z_grid = r_grid * np.cos(theta_grid)
        pos = np.stack((x_grid, y_grid, z_grid), axis=-1)
        pot = potential_disc(pos, self.ndiscs, self.dpars) + pot_ME

        # create interpolating functions for potential and acceleration
        self.__create_potacc_interpolators(r, theta, pot)

        # flag indicates that solution for potential has been found
        self.SolnFlag = True

        return

    def __create_mass_interpolator(self):
        """Set up self.__f_lnmass function, which interpolates mass."""
        N_q = 2000  # number of cells in radial dimension
        N_th = 2000  # number of cells in theta dimension
        r_min = 1e-4 * kpc
        r_max = 1e+4 * kpc

        # set up r grid
        q_edges = np.linspace(np.log(r_min), np.log(r_max), num=N_q + 1)
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
                                   fill_value=(0, lnM_enc[-1]))

        return

    def __create_potacc_interpolators(self, r, theta, pot):
        """Set up interpolators for potential and acceleration."""
        q = np.log(r)
        self.__f_pot = RBS(q, theta, pot)

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
        self.__f_aR = RBS(q_cen, th_cen, -dpdR_new)
        self.__f_az = RBS(q_cen, th_cen, -dpdz_new)

        return

    def potential(self, pos):
        """
        Interpolate potential at given positions.

        Parameters
        ----------
        pos : array, shape (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate potential, in 3D Galactocentric
            Cartesian coordinates. UNITS: m

        Returns
        -------
        pot : array, shape (N) or (N1, N2, N3, ...)
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
        return pot

    def acceleration(self, pos):
        """
        Interpolate acceleration at given positions.

        Parameters
        ----------
        pos : array, shape (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate acceleration, in 3D Galactocentric
            Cartesian coordinates. UNITS: m

        Returns
        -------
        acc : array, shape (N, 3) or (N1, N2, N3, ..., 3)
            Accelerations at given positions, in 3D Galactocentric
            Cartesian coordinates. UNITS: m / s^2
        """
        # check potential solution exists
        assert self.SolnFlag

        # get spherical coordinates from pos
        r = np.linalg.norm(pos, axis=-1)
        q = np.log(r)
        theta = np.arccos(pos[..., 2] / r)

        # interpolate z accel
        az = self.__f_az.ev(q, theta)

        # interpolate R accel
        x = pos[..., 0]
        y = pos[..., 1]
        R = np.sqrt(x**2 + y**2)
        aR = self.__f_aR.ev(q, theta)

        # recast into x and y accels (zero when R=0)
        ax = np.zeros_like(aR)
        ay = np.zeros_like(aR)
        ax[R != 0] = aR[R != 0] * x[R != 0] / R[R != 0]
        ay[R != 0] = aR[R != 0] * y[R != 0] / R[R != 0]

        # stack ax ay az
        a = np.stack((ax, ay, az), axis=-1)

        return a

    def density(self, pos):
        """
        Evaluate density at given positions.

        Parameters
        ----------
        pos : array, shape (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate density, in 3D Galactocentric
            Cartesian coordinates. UNITS: m

        Returns
        -------
        rho : array, shape (N) or (N1, N2, N3, ...)
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
            rho += rho_sph(pos, **self.spars[i])

        # loop over discs and add densities
        for i in range(self.ndiscs):
            rho += sigma(R, **self.dpars[i]) * zeta(z, **self.dpars[i])

        return rho

    def mass_enclosed(self, pos):
        """
        Evaluate enclosed spherical mass at given positions.

        Parameters
        ----------
        pos : array, shape (N, 3) or (N1, N2, N3, ..., 3)
            Positions at which to evaluate mass, in 3D Galactocentric
            Cartesian coordinates. UNITS: m

        Returns
        -------
        mass : array, shape (N) or (N1, N2, N3, ...)
            Enclosed spherical mass at given positions. UNITS: kg
        """
        r = np.linalg.norm(pos, axis=-1)
        q = np.log(r)
        mass = np.exp(self.__f_lnmass(q))
        return mass
