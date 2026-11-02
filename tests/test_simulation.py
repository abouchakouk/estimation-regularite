import numpy as np

from regularite.simulation import mvt_brownien
from regularite.simulation import fbm_davies_harte

def test_taille_et_depart():
    """La trajectoire a n+1 points et part de 0."""
    rng = np.random.default_rng(0)
    B = mvt_brownien(100, rng)
    assert len(B) == 101   # la longueur de B vaut 101
    assert B[0] == 0   # B[0] vaut 0


def test_reproductible():
    """Même graine, même trajectoire."""
    B1 = mvt_brownien(100, np.random.default_rng(42))
    B2 = mvt_brownien(100, np.random.default_rng(42))
    assert np.array_equal(B1, B2)


def test_variance_en_1():
    """Var(B_1) = 1 : on vérifie sur M trajectoires indépendantes."""
    rng = np.random.default_rng(0)
    M = 20000
    valeurs_finales = np.array([mvt_brownien(50, rng)[-1] for _ in range(M)])
    assert abs(valeurs_finales.var() - 1) / np.sqrt(2 / M) < 5 

"""test sur le fBM"""

def test_fbm_taille_et_depart():
    """La trajectoire a n+1 points et part de 0."""
    B = fbm_davies_harte(100, 0.7, np.random.default_rng(0))
    assert len(B) == 101
    assert B[0] == 0


def test_fbm_variance_en_1():
    """Var(B_H(1)) = 1, pour plusieurs valeurs de H."""
    M = 20000
    for H in (0.2, 0.5, 0.8):
        rng = np.random.default_rng(0)
        valeurs = np.array([fbm_davies_harte(64, H, rng)[-1] for _ in range(M)])
        assert abs(valeurs.var() - 1) / np.sqrt(2 / M) < 5 


def test_fbm_variance_en_un_demi():
    """Var(B_H(1/2)) = 2^(-2H) : vérifie la structure de corrélation."""
    M = 20000
    H = 0.8
    rng = np.random.default_rng(0)
    valeurs = np.array([fbm_davies_harte(64, H, rng)[32] for _ in range(M)])
    sigma2 = 2 ** (-2 * H)
    assert abs(valeurs.var() - sigma2) / (sigma2 * np.sqrt(2 / M)) < 5
