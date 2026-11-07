import numpy as np

from regularite.estimation import spectre_geometrique
from regularite.lacunaires import zeta_lacunaire
from regularite.cascades import zeta_cascade

P_GRILLE = np.arange(0.25, 5.01, 0.25)


def test_spectre_lacunaire_theorique():
    """Avec la vraie zeta : D(h) = p_c h sur [beta, 1/p_c]."""
    alpha, beta = 0.5, 0.6
    pc = (1 - alpha) / beta
    h, D = spectre_geometrique(P_GRILLE, zeta_lacunaire(P_GRILLE, alpha, beta))
    assert np.isclose(h[0], beta)
    assert np.isclose(h[-1], 1 / pc)
    assert np.allclose(D, pc * h)


def test_spectre_cascade_theorique():
    """Avec la vraie zeta : parabole au milieu, corde D = h au-delà de 1 - lambda^2."""
    lam = 0.3
    h, D = spectre_geometrique(P_GRILLE, zeta_cascade(P_GRILLE, lam))
    parabole = 1 - (1 + lam**2 - h)**2 / (4 * lam**2)
    milieu = (h > 0.6) & (h < 0.9)
    assert np.allclose(D[milieu], parabole[milieu], atol=0.01)
    droite = h > 1 - lam**2 + 0.01
    assert np.allclose(D[droite], h[droite])