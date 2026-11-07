"""Trajectoires de fBm pour plusieurs valeurs de H (méthode de Davies-Harte)."""
import numpy as np
import matplotlib.pyplot as plt

from regularite.simulation import fbm_davies_harte

n = 1000
valeurs_H = (0.2, 0.5, 0.8)
nb_trajectoires = 3
rng = np.random.default_rng(0)
t = np.linspace(0,1,n+1)

fig, axes = plt.subplots(len(valeurs_H), 1, figsize=(8, 7), sharex=True)
for ax, H in zip(axes, valeurs_H):
    for _ in range(nb_trajectoires):
        ax.plot(t, fbm_davies_harte(n,H,rng), linewidth=0.8)
    ax.set_title(f"fBm, $H = {H}$")
    ax.set_ylabel("$B_H(t)$")

axes[-1].set_xlabel("$t$")
fig.tight_layout()
fig.savefig("figures/illustrations/fbm.png", dpi=150, bbox_inches="tight")