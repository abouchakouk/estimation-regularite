"""E2 : effet du bruit sur zeta estimée et sur s(t) = t zeta(1/t)."""

import numpy as np
import matplotlib.pyplot as plt

from regularite.lacunaires import coefficients_lacunaires, zeta_lacunaire
from regularite.bruit import bruiter
from regularite.estimation import zeta_chapeau, pc_chapeau

J, ALPHA, BETA, NB_REALISATIONS = 16, 0.5, 0.6, 20
SIGMAS = (0.01, 0.05, 0.2)
p_grille = np.arange(0.25, 5.01, 0.25)
t = 1 / p_grille
rng = np.random.default_rng(1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(p_grille, zeta_lacunaire(p_grille, ALPHA, BETA), "k-", label="théorie")
axes[1].plot(t, t * zeta_lacunaire(p_grille, ALPHA, BETA), "k-", label="théorie")
axes[1].plot([0, 1.3], [0, 1.3], "k:", lw=0.8, label="diagonale $s = t$")

for i, sigma in enumerate(SIGMAS):
    zetas = []
    for _ in range(NB_REALISATIONS):
        c = bruiter(coefficients_lacunaires(J, ALPHA, BETA, rng), sigma, rng)
        zetas.append(zeta_chapeau(c, p_grille))
    zeta = np.mean(zetas, axis=0)
    i2 = np.argmin(np.abs(p_grille - 2))
    texte_pc = f"{pc_chapeau(p_grille, zeta):.2f}" if zeta.max() >= 1 else "non défini"
    print(f"sigma = {sigma} : zeta(2) = {zeta[i2]:.3f}, p_c = {texte_pc}")
    axes[0].plot(p_grille, zeta, "o-", color=f"C{i}", ms=3, label=rf"$\sigma = {sigma}$")
    axes[1].plot(t, t * zeta, "o-", color=f"C{i}", ms=3, label=rf"$\sigma = {sigma}$")

axes[0].set_xlabel("$p$")
axes[0].set_ylabel(r"$\hat\zeta(p)$ apparent")
axes[1].set_xlabel("$t = 1/p$")
axes[1].set_ylabel(r"$\hat s(t) = t\,\hat\zeta(1/t)$")
for ax in axes:
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figures/E_bruit/e2.png", dpi=150, bbox_inches="tight")