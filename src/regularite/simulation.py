import numpy as np


def mvt_brownien(n, rng):
    """Trajectoire brownienne standard sur [0, 1], aux points k/n, k = 0..n."""
    increment = np.sqrt(1 / n) * rng.standard_normal(n)
    trajectoire = np.cumsum(np.concatenate([[0.0], increment]))
    return trajectoire