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