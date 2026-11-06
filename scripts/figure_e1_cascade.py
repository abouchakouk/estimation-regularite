"""E1, cascade : zeta estimée (moyenne et écart-type sur 20 réalisations)."""

import numpy as np
import matplotlib.pyplot as plt

from regularite.cascades import signal_cascade, zeta_cascade
from regularite.ondelettes import coefficients_linf
from regularite.estimation import zeta_chapeau

J, LAM, NB_REALISATIONS = 16, 0.3, 20
p_grille = np.arange(0.25, 5.01, 0.25)
rng = np.random.default_rng(0)

zetas = np.zeros((NB_REALISATIONS, len(p_grille)))
for r in range(NB_REALISATIONS):
    x = signal_cascade(J, LAM, rng)
    zetas[r] = zeta_chapeau(coefficients_linf(x), p_grille)

moyenne = zetas.mean(axis=0)
ecart_type = zetas.std(axis=0)
for p in (2, 4):
    i = np.argmin(np.abs(p_grille - p))
    th = zeta_cascade(p, LAM)
    print(f"p = {p} : {moyenne[i]:.3f} +- {ecart_type[i]:.3f}  (théorie {th:.3f})")

fig, ax = plt.subplots(figsize=(6, 4))
p_fin = np.linspace(0, 5, 200)
ax.plot(p_fin, zeta_cascade(p_fin, LAM), "k-", label="théorie (Molchan)")
ax.errorbar(p_grille, moyenne, yerr=ecart_type, fmt="o", color="C1", capsize=3,
            label=r"$\hat\zeta(p)$")
ax.axvline(1 / LAM, color="gray", linestyle=":", label=r"$p_W = 1/\lambda$")
ax.set_xlabel("$p$")
ax.set_ylabel(r"$\zeta(p)$")
ax.set_title(rf"Cascade log-normale, $\lambda = {LAM}$, {NB_REALISATIONS} réalisations")
ax.legend()
fig.savefig("figures/e1_cascade.png", dpi=150, bbox_inches="tight")