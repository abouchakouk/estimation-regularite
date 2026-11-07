"""E1, série lacunaire : zeta estimée contre zeta théorique (sans bruit)."""

import numpy as np
import matplotlib.pyplot as plt

from regularite.lacunaires import coefficients_lacunaires, zeta_lacunaire
from regularite.estimation import zeta_chapeau, pc_chapeau

J, ALPHA, BETA = 16, 0.5, 0.6
p_grille = np.arange(0.25, 5.01, 0.25)
rng = np.random.default_rng(0)

c = coefficients_lacunaires(J, ALPHA, BETA, rng)
zeta = zeta_chapeau(c, p_grille, j_min=5, j_max=14)     
theorie = zeta_lacunaire(p_grille, ALPHA, BETA)

print("écart moyen :", np.mean(zeta - theorie))
print(
    "p_c estimé :", pc_chapeau(p_grille, zeta), " (théorie :", (1 - ALPHA) / BETA, ")"
)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(p_grille, theorie, "k-", label=r"théorie $\beta p + \alpha$")
ax.plot(p_grille, zeta, "o", color="C0", label=r"$\hat\zeta(p)$")
ax.set_xlabel("$p$")
ax.set_ylabel(r"$\zeta(p)$")
ax.set_title(rf"Série lacunaire, $\alpha = {ALPHA}$, $\beta = {BETA}$, sans bruit")
ax.legend()
fig.savefig("figures/E_bruit/e1_lacunaire.png", dpi=150, bbox_inches="tight")