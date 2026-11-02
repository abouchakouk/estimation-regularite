import numpy as np


def mvt_brownien(n, rng):
    """Trajectoire brownienne standard sur [0, 1], aux points k/n, k = 0..n."""
    increment = np.sqrt(1 / n) * rng.standard_normal(n)
    trajectoire = np.cumsum(np.concatenate([[0.0], increment]))
    return trajectoire

def autocov_fgn(k, H):
    """Autocovariance du bruit gaussien fractionnaire de paramètre H, aux décalages k."""
    return 1/2 * (np.abs(k+1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k-1)**(2*H))

def valeurs_propres_circulantes(n, H):
    """Valeurs propres du plongement circulant de taille 2n de la covariance du fGn."""
    j = np.arange(2 * n)
    c = autocov_fgn(np.minimum(j, 2 * n - j), H) 
    return np.fft.fft(c).real

def fbm_davies_harte(n, H, rng):
    """fBm d'indice H sur [0, 1], aux points k/n, k = 0..n (méthode de Davies-Harte)."""
    lam = np.maximum(valeurs_propres_circulantes(n, H), 0)
    Z = rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)
    V = np.fft.fft(np.sqrt(lam / (2 * n)) * Z)
    G = V.real[:n]
    return np.concatenate([[0.0], np.cumsum(n ** (-H) * G)])