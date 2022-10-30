This repository contains replication files for "Rate-Optimal Cluster-Randomized Designs for Spatial Interference." The files were coded for Python 3 and require the following dependencies: numpy, scipy, networkx, scikit-learn, and pandas.

Contents:
* `functions.py`: functions for simulating data and computing estimates.
* `monte_carlo.py`: run this file to replicate monte carlo results.

To replicate the results, run `monte_carlo.py` three times, once with parameter `weight_matrix='RGG'` and twice with `weight_matrix='invdist'` for respective parameters `kappa=4` and `kappa=5`.