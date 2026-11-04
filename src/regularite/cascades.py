"""Cascade log-normale de Mandelbrot et fonction X(t) = mu([0, t])."""

import numpy as np

def masses_cascade(J, lam, rng):
    """Masses mu(I_{J,k}), k = 0..2^J - 1, d'une cascade log-normale.
    Poids log-normaux W = exp(sigma_W Z - sigma_W^2 / 2), d'espérance 1.
    Paramètre : lambda = sigma_W / sqrt(2 ln 2).
    """
    sigma_W = lam * np.sqrt(2 * np.log(2)) # à partir de lambda
    masses = np.ones(1)
    for _ in range(J):
        W = np.exp(sigma_W * rng.standard_normal(2 * len(masses)) - sigma_W ** 2 / 2) # 2 * len(masses) poids log-normaux d'espérance 1
        masses = np.repeat(masses, 2) * W / 2 # dédoublement, poids, division par 2
    return masses
    
def signal_cascade(J, lam, rng):
    """X(k/n) = mu([0, k/n]) - (k/n) mu([0, 1]), k = 1..n, avec n = 2^J.
    On retire la tendance linéaire pour que le signal soit périodique (X(0) = X(1) = 0) 
    : sinon la transformée périodisée voit un saut au bord, qui fausse S(j, p)."""
        
    n = 2**J
    X = np.cumsum(masses_cascade(J, lam, rng)) # fonction de répartition : somme cumulée des masses
    t = np.arange(1, n + 1) / n
    return X- X[-1] * t
        
def zeta_cascade(p, lam):
    """zeta théorique (Molchan) : parabole si p <= 1/lambda, droite sinon.
    """
    p = np.asarray(p, dtype=float)
    parabole = p * (1 + lam**2) - lam**2 * p**2
    droite = 1 + (1 - lam) ** 2 * p
    return np.where(p <= 1 / lam, parabole, droite)