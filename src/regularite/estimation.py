"""Outils d'estimation multifractale à partir des coefficients d'ondelettes."""
import numpy as np


def fonction_structure(c, p):
    """S(j, p) = 2^{-j} sum_k |c_{j,k}|^p, pour chaque niveau j de la liste c."""
    S = np.zeros(len(c))
    for j in range(len(c)):
        S[j] = 2 ** (-j) * np.sum(np.abs(c[j])**p)
    return S

def zeta_chapeau(c, p_grille, j_min=5, j_max=14):
    """Estimateur de zeta(p) : moins la pente de log2 S(j, p) sur j dans [j_min, j_max]."""
    j = np.arange(j_min, j_max + 1)                               # les niveaux j_min..j_max (inclus)
    zeta = np.zeros(len(p_grille))
    for i, p in enumerate(p_grille):
        y = np.log2(fonction_structure(c, p))                           # log2 S(j, p) sur ces niveaux
        pente = np.polyfit(j, y[j], 1)
        zeta[i] = - pente[0]
    return zeta

def pc_chapeau(p_grille, zeta):
    """Solution de zeta(p) = 1, par interpolation linéaire sur la grille (zeta croissante)."""
    return np.interp(1, zeta, p_grille)