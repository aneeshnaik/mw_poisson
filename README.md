# mw_poisson
`mw_poisson` is a python-3 package providing a Poisson solver for the (axisymmetric) Milky Way potential, essentially an alternative to [GalPot](https://github.com/PaulMcMillan-Astro/GalPot). 

The Poisson solver is based on the method described in [Dehnen and Binney (1998)](https://ui.adsabs.harvard.edu/abs/1998MNRAS.294..429D/abstract), designed for the solution of Poisson's equation in the context of a discoid mass distribution. Essentially, the potential is decomposed into an analytic disc-plane component and another component less confined to the midplane. This latter component can then be found by numerical means, using a spherical harmonic expansion. Because it is less strongly confined to the disc plane, the spherical harmonic method converges faster.

The parameterisation of the Milky Way density profile is the empirical model of [McMillan (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.465...76M/abstract). As well as the best-fitting model of McMillan (2017), other parameter values for the various components can also be adopted, including some 'meta-parameters' such as the slope of the dark matter halo.

Further details about the mathematical background can be found in theory.pdf.

## Prerequisites

This code was written and implemented with python 3.7, and requires the following external packages (the version numbers in parentheses indicate the versions employed at time of writing, not particular dependencies):

* `scipy` (1.5.0)
* `numpy` (1.18.5)


## Usage Examples

The code block below demonstrates how to set up a galaxy with the parameters of [McMillan (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.465...76M/abstract) and solve for its potential.

```python
import numpy as np
from mw_poisson import MilkyWay
from mw_poisson.constants import kpc, M_sun, pc

# disc and spheroid parameters from McMillan (2017)
d1 = {'sigma_0': 895.679 * (M_sun / pc**2), 'R_0': 2.49955 * kpc, 'R_h': 0,
      'z_0': 300 * pc, 'form': 'exponential'}
d2 = {'sigma_0': 183.444 * (M_sun / pc**2), 'R_0': 3.02134 * kpc, 'R_h': 0,
      'z_0': 900 * pc, 'form': 'exponential'}
d3 = {'sigma_0': 53.1319 * (M_sun / pc**2), 'R_0': 7 * kpc, 'R_h': 4 * kpc,
      'z_0': 85 * pc, 'form': 'sech'}
d4 = {'sigma_0': 2179.95 * (M_sun / pc**2), 'R_0': 1.5 * kpc, 'R_h': 12 * kpc,
      'z_0': 45 * pc, 'form': 'sech'}
s1 = {'alpha': 1.8, 'beta': 0, 'q': 0.5, 'r_cut': 2.1 * kpc,
      'r_0': 0.075 * kpc, 'rho_0': 98.351 * (M_sun / pc**3)}
s2 = {'alpha': 2, 'beta': 1, 'q': 1, 'r_cut': np.inf,
      'r_0': 19.5725 * kpc, 'rho_0': 0.00853702 * (M_sun / pc**3)}

# set up Milky Way
gal = MilkyWay(ndiscs=4, dpars=[d1, d2, d3, d4], nspheroids=2, spars=[s1, s2])
gal.solve_potential(verbose=True)
```
Note that all units are in SI units (m, kg, s), and mw_poisson.constants provides various conversions to commonly used astrophysical units. The optional `verbose` flag displays a progress bar during the spherical harmonic expansion. The MilkyWay documentation gives further information about how to play around with the various arguments and parameters.

Having found a solution for the potential, one can then obtain values for the potential as a function of position, by providing a 3D position to MilkyWay.potential in Cartesian Galactocentric coordinates.
```python
>>> pos = 10 * kpc * np.ones((3))
>>> gal.potential(pos)
-142270571570.52524
```
MilkyWay.potential can also take an array of positions as an argument, provided only that the size of the last dimension is 3.
```python
>>> pos = 10 * kpc * np.ones([2, 4, 3])*10*kpc
>>> gal.potential(pos2D)
array([[-1.42270572e+11, -1.42270572e+11, -1.42270572e+11, -1.42270572e+11],
       [-1.42270572e+11, -1.42270572e+11, -1.42270572e+11, -1.42270572e+11]])
```
In a similar manner, one can also access other quantities via `MilkyWay.acceleration(pos)` and `MilkyWay.density(pos)` and `MilkyWay.mass_enclosed(pos)`.

## Author

This code was written by **Aneesh Naik** ([website](https://aneeshnaik.github.io/)).

## License

Copyright (2020) Aneesh Naik.

`mw_poisson` is free software made available under the MIT license. For details see LICENSE.
