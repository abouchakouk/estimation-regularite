"""P2, calcul Monte-Carlo : log2 S(j, p) par répétition pour un fBm, plusieurs J et H.

Écrit resultats/P_rugueux/p2.npz. Aucune figure : les scripts figure_p2_*.py relisent ce fichier.
"""

import time
import numpy as np

from regularite.simulation import fbm_davies_harte
from regularite.ondelettes import coefficients_linf, periodiser
from regularite.estimation import fonction_structure

GRAINE = 20261108
J_VALEURS = (8, 11, 16)
H_VALEURS = (0.06, 0.3)
R = 500
p_grille = np.arange(0.25, 5.01, 0.25)

graines = np.random.SeedSequence(GRAINE).spawn(len(J_VALEURS) * len(H_VALEURS))

resultats = {}
debut = time.perf_counter()
for a, J in enumerate(J_VALEURS):
    log_S = np.zeros((len(H_VALEURS), R, J, len(p_grille)))     # axes : H, répétition, j, p
    for b, H in enumerate(H_VALEURS):
        rng = np.random.default_rng(graines[a * len(H_VALEURS) + b])
        for r in range(R):
            c = coefficients_linf(periodiser(fbm_davies_harte(2 ** J, H, rng)))
            for i, p in enumerate(p_grille):
                log_S[b, r, :, i] = np.log2(fonction_structure(c, p))
    resultats[f"log_S_J{J}"] = log_S
    print(f"J = {J} : {time.perf_counter() - debut:.1f} s")

np.savez("resultats/P_rugueux/p2.npz", **resultats, p_grille=p_grille,
         J_valeurs=np.array(J_VALEURS), H_valeurs=np.array(H_VALEURS), R=R, graine=GRAINE)