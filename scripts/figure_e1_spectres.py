"""E1 : spectres estimés contre spectres théoriques (série lacunaire et cascade)."""

import numpy as np
import matplotlib.pyplot as plt

from regularite.lacunaires import coefficients_lacunaires
from regularite.cascades import signal_cascade
from regularite.ondelettes import coefficients_linf
from regularite.estimation import zeta_chapeau, pc_chapeau, spectre_geometrique

J, ALPHA, BETA, LAM, NB_REALISATIONS = 16, 0.5, 0.6, 0.3, 20
p_grille = np.arange(0.25, 5.01, 0.25)
rng = np.random.default_rng(0)

zeta_lac = zeta_chapeau(coefficients_lacunaires(J,ALPHA,BETA,rng),p_grille)  # zeta estimée d'une série lacunaire
zeta_cas = np.mean(
    [zeta_chapeau(coefficients_linf(signal_cascade(J, LAM, rng)), p_grille)
     for _ in range(NB_REALISATIONS)],
    axis=0,
)  # zeta estimée moyenne de 20 cascades

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

h, D = spectre_geometrique(p_grille, zeta_lac)
pc = (1 - ALPHA) / BETA
h_th = np.linspace(BETA, 1 / pc, 100)
axes[0].plot(h_th, pc * h_th, "k-", label=r"théorie $p_c h$")
axes[0].plot(h, D, "C0--", label=rf"estimé, $\hat p_c = {pc_chapeau(p_grille, zeta_lac):.3f}$")
axes[0].set_title("Série lacunaire")

h, D = spectre_geometrique(p_grille, zeta_cas)
h_th = np.linspace((1 - LAM) ** 2, 1, 200)
parabole = 1 - (1 + LAM**2 - h_th)**2 / (4 * LAM**2)
axes[1].plot(h_th, parabole, "k-", label="parabole (vrai spectre)")
axes[1].plot(h, D, "C1--", label="estimé")
axes[1].set_title(rf"Cascade, $\lambda = {LAM}$")

for ax in axes:
    ax.set_xlabel("$h$")
    ax.set_ylabel("$D(h)$")
    ax.legend()
fig.tight_layout()
fig.savefig("figures/e1_spectres.png", dpi=150, bbox_inches="tight")