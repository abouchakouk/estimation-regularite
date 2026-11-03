import numpy as np

from regularite.lacunaires import coefficients_lacunaires
from regularite.estimation import fonction_structure

J, ALPHA, BETA = 10, 0.5, 0.6


def test_nombre_et_hauteur_des_bosses():
    """Au niveau j : 2^j coefficients, dont N_j non nuls, tous égaux à 2^{-j beta}."""
    c = coefficients_lacunaires(J, ALPHA, BETA, np.random.default_rng(0))
    for j in range(J):
        N_j = int(np.floor(2 ** (j * (1 - ALPHA))))
        assert len(c[j]) == 2**j                       # taille du niveau j
        assert np.count_nonzero(c[j]) == N_j           # nombre de coefficients non nuls
        non_nuls = c[j][c[j] != 0]
        assert np.allclose(non_nuls, 2**(-j*BETA))  # leur valeur commune


def test_fonction_structure_exacte():
    """S(j, p) = 2^{-j} N_j 2^{-j beta p}, pour plusieurs p."""
    c = coefficients_lacunaires(J, ALPHA, BETA, np.random.default_rng(0))
    j = np.arange(J)
    N = np.floor(2 ** (j * (1 - ALPHA)))
    for p in (0.5, 1, 2, 4):
        attendu = (1 / (2 ** j) )* N * (1 / (2 ** (j * BETA * p)))                  # formule vectorisée en j
        assert np.allclose(fonction_structure(c, p), attendu)