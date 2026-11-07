import numpy as np

from regularite.bruit import bruiter, moment_gaussien, plancher, coude
from regularite.estimation import fonction_structure


def test_moments_gaussiens():
    """m_2 = 1, m_4 = 3, m_1 = sqrt(2/pi)."""
    assert np.isclose(moment_gaussien(2), 1)
    assert np.isclose(moment_gaussien(4), 3)
    assert np.isclose(moment_gaussien(1), np.sqrt(2 / np.pi))


def test_plancher_bruit_seul():
    """Sur du bruit seul, S(j, p) / F(j, p) est proche de 1 aux niveaux fins."""
    J, sigma = 16, 0.05
    eps = sigma / np.sqrt(2**J)
    c_zero = [np.zeros(2**j) for j in range(J)]
    y = bruiter(c_zero, sigma, np.random.default_rng(0))
    j = 14
    for p in (1, 2, 4):
        rapport = fonction_structure(y, p)[j] / plancher(j, p, eps)
        ecart_relatif = np.sqrt(moment_gaussien(2 * p) / moment_gaussien(p) ** 2 - 1) / np.sqrt(2**j)
        assert abs(rapport - 1) < 5 * ecart_relatif


def test_coude_egalite_signal_plancher():
    """Au coude, le signal 2^{-j(2 beta + alpha)} égale le plancher 2^j eps^2."""
    alpha, beta, eps = 0.5, 0.6, 1e-4
    j_eps = coude(eps, alpha, beta)
    assert np.isclose(2.0 ** (-j_eps * (2 * beta + alpha)), plancher(j_eps, 2, eps))