"""Trace quelques trajectoires de mouvement brownien standard sur [0, 1]."""
import numpy as np
import matplotlib.pyplot as plt

from regularite.simulation import mvt_brownien

n = 1000
nb_trajectoires = 5
rng = np.random.default_rng(0)
t = np.linspace(0, 1, n+1)   # la grille 0, 1/n, 2/n, ..., 1

fig, ax = plt.subplots(figsize=(8, 4))
for _ in range(nb_trajectoires):
    B = mvt_brownien(n, rng)
    ax.plot(t, B, linewidth=0.8)

ax.set_xlabel("$t$")
ax.set_ylabel("$B_t$")
ax.set_title(ax.set_title(f"{nb_trajectoires} trajectoires browniennes, n = {n}"))
fig.savefig("figures/illustrations/brownien.png", dpi=150, bbox_inches="tight")