"""P2, contrôle : pentes locales s_j(1/p) = zeta_j(p) / p d'un fBm rugueux (20 trajectoires)."""

import time
import numpy as np
import matplotlib.pyplot as plt

from regularite.simulation import fbm_davies_harte
from regularite.ondelettes import coefficients_linf, periodiser
from regularite.estimation import pentes_locales, zeta_chapeau

J, NB_TRAJ = 16, 20
H_VALEURS = (0.06, 0.3)
P_VALEURS = np.array([1.0, 2.0, 4.0])
GAMMES = ((5, 14), (5, 11))
rng = np.random.default_rng(0)

debut = time.perf_counter()
s_loc = np.zeros((len(H_VALEURS), NB_TRAJ, J - 1, len(P_VALEURS)))   # s_j(1/p)
s_reg = np.zeros((len(H_VALEURS), NB_TRAJ, len(GAMMES)))            # s(1/2) par régression
for a, H in enumerate(H_VALEURS):
    for r in range(NB_TRAJ):
        c = coefficients_linf(periodiser(fbm_davies_harte(2 ** J, H, rng)))
        s_loc[a, r] = pentes_locales(c, P_VALEURS) / P_VALEURS
        for g, (j1, j2) in enumerate(GAMMES):
            s_reg[a, r, g] = zeta_chapeau(c, [2.0], j1, j2)[0] / 2
print(f"temps de calcul : {time.perf_counter() - debut:.1f} s")

for a, H in enumerate(H_VALEURS):
    for g, (j1, j2) in enumerate(GAMMES):
        m, e = s_reg[a, :, g].mean(), s_reg[a, :, g].std()
        print(f"H = {H} : s(1/2) sur [{j1},{j2}] = {m:.3f} +- {e:.3f}")
    i2 = np.argmin(np.abs(P_VALEURS - 2))
    print(f"H = {H} : s_j(1/2), j = 0..{J - 2} :",
          np.array2string(s_loc[a, :, :, i2].mean(axis=0), precision=3))

j = np.arange(J - 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
for a, (ax, H) in enumerate(zip(axes, H_VALEURS)):
    ax.axvspan(GAMMES[0][0], GAMMES[0][1] - 1, color="0.9", label="pentes de $[5,14]$")
    ax.axvline(GAMMES[1][1] - 1, color="0.5", linestyle=":", label="fin de $[5,11]$")
    for i, p in enumerate(P_VALEURS):
        m = s_loc[a, :, :, i].mean(axis=0)
        e = s_loc[a, :, :, i].std(axis=0)
        ax.plot(j, m, "o-", color=f"C{i}", ms=3, label=rf"$p = {p:g}$")
        ax.fill_between(j, m - e, m + e, color=f"C{i}", alpha=0.2)
    ax.axhline(H, color="k", linewidth=1, label=r"$s = H$")
    ax.set_ylim(H - 0.5, H + 0.3)
    ax.set_xlabel(r"$j$ (pente entre les niveaux $j$ et $j+1$)")
    ax.set_title(rf"fBm, $H = {H}$, $J = {J}$, {NB_TRAJ} trajectoires")
axes[0].set_ylabel(r"$\hat s_j(1/p) = \hat\zeta_j(p)/p$")
axes[0].legend(fontsize=8)
fig.savefig("figures/P_rugueux/p2_controle_pentes_locales.png", dpi=150, bbox_inches="tight")