import numpy as np

from regularite.simulation import mvt_brownien


def test_taille_et_depart():
    """La trajectoire a n+1 points et part de 0."""
    rng = np.random.default_rng(0)
    B = mvt_brownien(100, rng)
    assert ...   # la longueur de B vaut 101
    assert ...   # B[0] vaut 0


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
    assert abs(valeurs_finales.var() - 1) < ...