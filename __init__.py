#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mw_poisson is a Poisson solver for axisymmetric galactic potentials.

The main object is the MilkyWay class, which sets up a galactic density
profile, solves for the potential, then provides various functions that can
interpolate the density, potential, and acceleration at any given point.

See README for further details and usage examples.

Created: July 2020
Author: A. P. Naik
"""
from .milkyway import MilkyWay

__all__ = ['MilkyWay']
