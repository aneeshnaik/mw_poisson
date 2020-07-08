#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUMMARY HERE.

Created: Wed Jul  8 18:40:06 2020
Author: A. P. Naik
Description: DESCRIPTION HERE.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS


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