"""Transformée en ondelettes discrète et passage à la normalisation L-infini.
"""
import warnings
import numpy as np
import pywt

def coefficients_linf(x, ondelette="db4"):
    """Coefficients c[j], j = 0..J-1, en normalisation L-infini : c_{j,k} = 2^{j/2} 
    d_{j,k} / sqrt(n).
    """
    n = len(x)
    J = int(np.log2(n)) # n = 2^J ; pense à int(...)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = pywt.wavedec(x, ondelette, mode="periodization", level=J) # wavedec, périodisée, J niveaux
    details = coeffs[1:]
    return [2 ** (j / 2) * details[j] / np.sqrt(n) for j in range(J)]

def periodiser(x):
    """Rend périodique un signal donné aux points k/n, k = 0..n (longueur n + 1).

    Renvoie y_k = x_k - x_0 - (k/n)(x_n - x_0), k = 1..n (longueur n) : y_n = y_0 = 0,
    donc pas de saut au raccord de la transformée périodisée. Seuls les coefficients de
    bord (k <= 2 et k >= 2^j - 3 pour db4) sont modifiés : les moments nuls annulent
    les coefficients intérieurs d'une fonction affine.
    """
    x = np.asarray(x, dtype=float)
    n = len(x) - 1
    k = np.arange(1, n + 1)
    return x[1:] - x[0] - k / n * (x[-1] - x[0])