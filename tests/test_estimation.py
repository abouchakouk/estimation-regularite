import numpy as np

from regularite.lacunaires import coefficients_lacunaires
from regularite.estimation import zeta_chapeau, pc_chapeau

P_GRILLE = np.arange(0.25, 5.01, 0.25)


def test_zeta_loi_puissance_pure():
    """Si c_{j,k} = 2^{-js} pour tout k, alors S(j,p) = 2^{-jsp} et zeta_chapeau(p) = s p exactement."""
    s = 0.7
    c = [np.full(2 ** j, 2.0 ** (-j * s)) for j in range(16)]
    assert np.allclose(zeta_chapeau(c, P_GRILLE), P_GRILLE * s)


def test_pc_cas_lineaire():
    """Si zeta(p) = 2p, alors p_c = 1/2."""
    assert np.isclose(pc_chapeau(P_GRILLE, 2 * P_GRILLE), 1 / 2)


def test_lacunaire_decalage_uniforme():
    """zeta_chapeau(p) - (beta p + alpha) = (1 - alpha) - b_N pour tout p, et p_c = b_N / beta."""
    alpha, beta = 0.5, 0.6
    c = coefficients_lacunaires(16, alpha, beta, np.random.default_rng(0))
    j = np.arange(5, 15)
    N = np.floor(2 ** (j * (1 - alpha)))
    b_N = np.polyfit(j, np.log2(N), 1)[0]                              # pente de log2 N_j sur j
    zeta = zeta_chapeau(c, P_GRILLE)
    assert np.allclose(zeta - (beta * P_GRILLE + alpha), 1 - alpha - b_N)
    assert np.isclose(pc_chapeau(P_GRILLE, zeta), b_N / beta)
