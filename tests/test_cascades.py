import numpy as np

from regularite.cascades import masses_cascade, signal_cascade, zeta_cascade


LAM = 0.3

def test_masses_taille_et_positivite():
    """2^J masses, toutes strictement positives.
    """
    m = masses_cascade(10, LAM, np.random.default_rng(0))
    assert len(m) == 2 ** 10
    assert np.all(m > 0) # indice : np.all(condition)


def test_masse_totale_en_moyenne():
    """E[mu([0, 1])] = 1 : test statistique sur M cascades indépendantes."""

    rng = np.random.default_rng(0)
    M = 2000
    totales = np.array([masses_cascade(10, LAM, rng).sum() for _ in range(M)])
    erreur_type = totales.std() / np.sqrt(M) # écart-type de la moyenne empirique
    assert abs(totales.mean() - 1) / erreur_type < 5


def test_signal_periodique():
    """Après retrait de la tendance, X(1) = 0."""
    X = signal_cascade(10, LAM, np.random.default_rng(0))
    assert len(X) == 2**10
    assert np.isclose(X[-1], 0)



def test_zeta_cascade_proprietes():
    """zeta(0) = 0, zeta(1) = 1, zeta(2) = 2(1 + lambda^2) - 4 lambda^2, et continuité en p_W = 1/lambda."""
    assert np.isclose(zeta_cascade(0, LAM), 0)
    assert np.isclose(zeta_cascade(1, LAM), 1)
    assert np.isclose(zeta_cascade(2, LAM), 2 * (1 + LAM ** 2) - 4 * LAM ** 2)
    pW = 1 / LAM
    assert np.isclose(zeta_cascade(pW - 1e-9, LAM), zeta_cascade(pW + 1e-9, LAM))