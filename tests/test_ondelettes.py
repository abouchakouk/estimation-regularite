import numpy as np
import pywt

from regularite.ondelettes import coefficients_linf, periodiser
from regularite.simulation import fbm_davies_harte
from regularite.estimation import zeta_chapeau


def test_tailles():
    """Le niveau j contient 2^j coefficients, pour j = 0..J-1."""
    c = coefficients_linf(np.random.default_rng(0).standard_normal(2**10))
    assert [len(cj) for cj in c] == [2 ** j for j in range(10)]


def test_constante_donne_zero():
    """Un signal constant a tous ses coefficients d'ondelettes nuls."""
    c = coefficients_linf(np.full(2**10, 3.0))
    assert all(np.allclose(cj, 0) for cj in c)


def test_normalisation_une_seule_bosse():
    """Un signal fait d'une seule ondelette de hauteur h redonne c_{j0,k0} = h."""
    J, j0, k0, h = 10, 6, 17, 0.25
    n = 2**J
    coeffs = [np.zeros(1)] + [np.zeros(2**j) for j in range(J)]
    coeffs[1 + j0][k0] = h * np.sqrt(n / 2**j0)
    x = pywt.waverec(coeffs, "db4", mode="periodization")
    c = coefficients_linf(x)
    assert np.isclose(c[j0][k0], h)
    c[j0][k0] = 0
    assert all(np.allclose(cj, 0) for cj in c)


def test_fbm_zeta_2_proche_de_2H():
    """Pour un fBm, zeta(2) = 2H : vérification grossière sur une réalisation."""
    n, H = 2**14, 0.7
    B = fbm_davies_harte(n, H, np.random.default_rng(0))
    c = coefficients_linf(periodiser(B))
    zeta2 = zeta_chapeau(c, np.array([2.0]), j_min=5, j_max=12)[0]
    assert abs(zeta2 - 2 * H) < 0.15


def test_periodiser_taille_et_raccord():
    """La sortie a n = 2^J points et se termine en 0 (raccord sans saut)."""
    x = fbm_davies_harte(2**10, 0.3, np.random.default_rng(0))
    y = periodiser(x)
    assert len(y) == 2**10
    assert np.isclose(y[-1], 0)


def test_periodiser_ne_change_que_le_bord():
    """Moments nuls : les coefficients intérieurs (3 <= k <= 2^j - 4) sont inchangés."""
    J = 12
    x = fbm_davies_harte(2**J, 0.3, np.random.default_rng(0))
    c_brut = coefficients_linf(x[1:])
    c_per = coefficients_linf(periodiser(x))
    for j in range(3, J):
        assert np.allclose(c_brut[j][3:-3], c_per[j][3:-3], atol=1e-10)