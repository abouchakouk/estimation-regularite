"""Bruit additif : observation, plancher et coude."""

import numpy as np
from scipy.special import gamma


def bruiter(c, sigma, rng):
    """Observation c_{j,k} + 2^{j/2} eps xi_{j,k}, avec eps = sigma / sqrt(n), n = 2^J."""
    n = 2 ** len(c)
    eps = sigma / np.sqrt(n)
    return [c[j] + 2**(j / 2) * eps * rng.standard_normal(2**j) for j in range(len(c))]


def moment_gaussien(p):
    """m_p = E|xi|^p pour xi ~ N(0, 1)."""
    return (2**(p / 2) * gamma((p + 1) / 2)) / (np.sqrt(np.pi))


def plancher(j, p, eps):
    """F(j, p) = m_p (2^{j/2} eps)^p : valeur de S(j, p) pour du bruit seul."""
    return moment_gaussien(p) * (2**(np.asarray(j) / 2) * eps)**p


def coude(eps, alpha, beta):
    """Niveau j_eps où, pour p = 2, le signal lacunaire passe sous le plancher."""
    return (2 * np.log2(1/eps)) / (2*beta + alpha + 1) 