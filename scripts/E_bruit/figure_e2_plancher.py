"""E2, panneau de gauche : log2 S(j, 2) avec et sans bruit, et le plancher F(j, 2)."""

import numpy as np
import matplotlib.pyplot as plt

from regularite.lacunaires import coefficients_lacunaires
from regularite.bruit import bruiter, plancher, coude
from regularite.estimation import fonction_structure

J, ALPHA, BETA = 16, 0.5, 0.6
SIGMAS = (0.01, 0.05, 0.2)
rng = np.random.default_rng(0)
j = np.arange(J)

c = coefficients_lacunaires(J, ALPHA, BETA, rng)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(j, np.log2(fonction_structure(c, 2)), "k-", label="signal seul")
for i, sigma in enumerate(SIGMAS):
    eps = sigma / np.sqrt(2**J)
    y = bruiter(c, sigma, rng)
    ax.plot(j, np.log2(fonction_structure(y, 2)), "o-", color=f"C{i}", ms=3, label=rf"$\sigma = {sigma}$")
    ax.plot(j, np.log2(plancher(j, 2, eps)), ":", color=f"C{i}")
    j_eps = coude(eps, ALPHA, BETA)
    print(f"sigma = {sigma} : eps = {eps:.2e}, coude prédit j = {j_eps:.1f}")
ax.axvspan(5, 14, color="gray", alpha=0.1, label="plage de régression")
ax.set_xlabel("$j$")
ax.set_ylabel(r"$\log_2 S(j, 2)$")
ax.set_title("Série lacunaire bruitée : signal, observation, plancher")
ax.legend(fontsize=8)
fig.savefig("figures/E_bruit/e2_plancher.png", dpi=150, bbox_inches="tight")