"""Séries d'ondelettes lacunaires (coefficients en normalisation L-infini)."""
import numpy as np


def coefficients_lacunaires(J, alpha, beta, rng):
    """Coefficients c[j], j = 0..J-1 : N_j = floor(2^{j(1-alpha)}) bosses de hauteur 2^{-j beta}."""
    c = []
    for j in range(J):
        N_j = int(np.floor(2**(j*(1-alpha))))                 # nombre de bosses au niveau j
        positions = rng.choice( 2**j, size=N_j, replace=False)    # N_j positions distinctes parmi 0..2^j - 1
        niveau = np.zeros(2**j)              # tableau de 2^j zéros
        niveau[positions] = 2**(-j*beta)   # hauteur des bosses
        c.append(niveau)
    return c


def zeta_lacunaire(p, alpha, beta):
    """Fonction d'échelle théorique de la série lacunaire : zeta(p) = beta p + alpha."""
    return beta * np.asarray(p) + alpha

