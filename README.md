# mw_poisson
`mw_poisson` is a python-3 package providing a Poisson solver for the (axisymmetric) Milky Way potential, essentially an alternative to [GalPot](https://github.com/PaulMcMillan-Astro/GalPot). 

The Poisson solver is based on the method described in [Dehnen and Binney (1998)](https://ui.adsabs.harvard.edu/abs/1998MNRAS.294..429D/abstract), designed for the solution of Poisson's equation in the context of a discoid mass distribution. Essentially, the potential is decomposed into an analytic disc-plane component and another component less confined to the midplane. This latter component can then be found by numerical means, using a spherical harmonic expansion. Because it is less strongly confined to the disc plane, the spherical harmonic method converges faster.

The parameterisation of the Milky Way density profile is the empirical model of [McMillan (2017)](https://ui.adsabs.harvard.edu/abs/2017MNRAS.465...76M/abstract). As well as the best-fitting model of McMillan (2017), other parameter values for the various components can also be adopted, including some 'meta-parameters' such as the slope of the dark matter halo.

## Prerequisites

This code was written and implemented with python 3.7, and requires the following external packages (the version numbers in parentheses indicate the versions employed at time of writing, not particular dependencies):

* `scipy` (1.5.0)
* `numpy` (1.18.5)


## Usage



## Author

## License
