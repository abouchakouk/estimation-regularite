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


def spectre_geometrique(p_grille, zeta, nb_h=200):
    """Spectre par la construction des cordes dans le plan (t, s), t = 1/p.

    D(h) = 1 - max_t (s(t) - h) / t, le max portant sur les t = 1/p de la grille tels
    que t <= 1/p_c, plus le point (1/p_c, 1/p_c). h parcourt [pente finale de zeta, 1/p_c].
    """
    pc = pc_chapeau(p_grille, zeta)
    t = [1/p for p in pc]
    s = t * zeta
    garde = t <= 1 / pc
    t = np.append(t[garde], 1 / pc)
    s = np.append(s[garde], 1 / pc)
    h_min = zeta[-2] - zeta[-1] / p_grille[-2] - p_grille[-1]
    h = np.linspace(h_min, 1 / pc, nb_h)
    pentes = (s[None, :] - h[:, None]) / t[None, :]
    D = 1 - pentes.max(axis=1)
    return h, D