# Journal de formation

Une entrée par séance : date, séance, compétence, étapes écrites seul / avec indices /
copiées, ce qui a bloqué (maths, code ou outil), écarts constatés à la réécriture de
mémoire.

Remarque : S1 et S2 ont été faites avec l'ancien format du programme (solution donnée
d'emblée, recopiée). Le format « maths puis code » s'applique à partir de S3.

---

## 2026-11-08 — S1 : pentes locales dans la bibliothèque

- **Compétence.** Indexation numpy et `np.diff` selon un axe ; test exact avec pytest.
- **Seul.** —
- **Avec indices.** —
- **Copiées.** `pentes_locales` (construction de `log_S`, `np.diff(..., axis=0)`) ; test
  exact $c_{j,k} = 2^{-jH}$ ; test du lemme 3.1 (moyenne pondérée des pentes locales).
- **Ce qui a bloqué.**
  - Code : `np.full`, `np.tile`, l'attribut `.shape`, l'opérateur `@` (produit
    matriciel, distinct de `*`) ; le broadcasting (qui rendait `np.tile` inutile dans le test).
- **Réécriture de mémoire.** Pas faite (ancien format).

## 2026-11-08 — S2 : contrôle des pentes locales d'un fBm rugueux

- **Compétence.** Figures à plusieurs panneaux ; générateur `np.random.default_rng` ;
  boucle Monte-Carlo.
- **Seul.** —
- **Avec indices.** —
- **Copiées.** Tout le script `scripts/P_rugueux/controle_p2_pentes_locales.py`.
- **Ce qui a bloqué.** Rien de signalé.
- **Prédiction avant lancement** (demandée par le nouveau programme) : non faite.
- **Réécriture de mémoire.** Pas faite (ancien format).

## 2026-11-08 — S3 : commenter le contrôle et écrire le protocole P2

- **Compétence.** Rédiger un protocole expérimental : question, facteurs, attendus
  chiffrés avec statut.
- **Seul.** Premier jet des sections « Question » et « Commentaire panneau par panneau »
  (dans le chat).
- **Avec indices.** Corrections de ce premier jet.
- **Copiées.** `notes/P_rugueux/p2.md`, rédigée par Claude puis relue.
- **Ce qui a bloqué.**
  - Maths : distinguer **biais** (déterministe, niveaux fins, $\delta_\kappa$) et
    **variance** (aléatoire, niveaux grossiers, peu de coefficients) ; « variance des
    coefficients » ($V_j$) et « variance de l'estimateur » ne sont pas la même chose.
  - Maths : origine de $\delta_\kappa$ : ni l'aléa, ni le bruit, ni le modèle, mais la
    résolution finie (filtre discret $\gamma_m$ court près de la résolution).
  - Maths : la borne $2^{-H(J-j_2)}$ de la note est une majoration non optimale ; la vitesse
    constatée est $2^{-(2H+1)(J-j_2)}$ ; la lenteur à $H$ petit vient de la constante
    ($\kappa_\infty = O(H)$).
  - Lecture de figure : deux panneaux de même forme ne sont pas « identiques » ; le
    résultat est dans les chiffres (étendue du plateau, taille du biais).
  - Bonne intuition : le décrochage ressemble à un bruit (pente vers $s = -\tfrac12$) ;
    formalisée en « bruit effectif $\sigma_{\text{eff}} \propto n^{-H}$ » [heuristique].
- **Réécriture de mémoire** (`pentes_locales` via `np.diff`, étape [nouveau] de S1) :
  écarts : _à compléter_.