#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Various useful functions for mw_poisson package.

See README for further details about mw_poisson and usage examples.

Created: July 2020
Author: A. P. Naik
"""
import numpy as np
import sys


def sech(x):
    """Hyperbolic sech function."""
    return 1 / np.cosh(x)


def print_progress(i, n, interval=1):
    """
    Display a percentage progress bar within a for loop.

    This function prints (to standard output) a self-updating progress bar,
    showing percentage completion of a for loop. Should be called at each
    iteration of the loop (see example below). The 'interval' parameter can
    optionally be used to adjust the update rate.

    Parameters
    ----------
    i : int
        Current iteration number.
    n : int
        Length of for loop.
    interval : int
        How often to update progress bar, in number of iterations. Default is
        1, i.e. every iteration. For larger loops, it can make sense to have
        interval = n//50 or similar.

    Example
    -------
    >>> n = 1234
    >>> for i in range(n):
    ...     print_progress(i, n)
    ...     do_something()

    """
    # write 100% for final iteration regardless of whether interval is correct
    if i + 1 == n:
        sys.stdout.write('\r')
        sys.stdout.write("[%-20s] 100%%" % ('=' * 20))
        sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
    # if 'interval' number of iters, then print progress bar
    elif (i + 1) % interval == 0:
        sys.stdout.write('\r')
        j = (i + 1) / n
        sys.stdout.write("[%-20s] %d%%" % ('=' * int(20 * j), 100 * j))
        sys.stdout.flush()

    return
