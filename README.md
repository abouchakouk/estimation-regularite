# estimation-regularite

Simulation et estimation de la régularité de signaux multifractals, en vue de
l'étude de l'estimateur du spectre multifractal observé avec un bruit additif.

Ce dépôt accompagne mon projet de recherche. Il réécrit,
brique par brique et avec des tests, les expériences numériques de mes notes.

## Contenu

| Module (`src/regularite/`) | Rôle |
|---|---|
| `simulation.py` | mouvement brownien ; mouvement brownien fractionnaire par la méthode de Davies–Harte (plongement circulant, FFT) |
| `lacunaires.py` | séries d'ondelettes lacunaires et leur fonction d'échelle théorique |
| `cascades.py` | cascade log-normale de Mandelbrot, signal X(t) = μ([0, t]), fonction d'échelle de Molchan |
| `ondelettes.py` | transformée en ondelettes discrète (db4 périodisée) et passage à la normalisation L∞ |
| `estimation.py` | fonction de structure S(j, p), estimateur de ζ(p) par régression, estimation de p_c, construction géométrique du spectre |

- `tests/` : tests exacts (lois d'échelle connues, normalisation des coefficients)
  et tests statistiques (tolérances justifiées par le théorème central limite).
- `scripts/` : un script par figure ; chaque figure se régénère en une commande.
- `figures/` : les figures produites.

## Installation

```
conda create -n regul python=3.12 numpy scipy matplotlib pywavelets pytest
conda activate regul
pip install -e .
```

## Utilisation

```
pytest                                   # lance tous les tests
python scripts/figure_e1_lacunaire.py    # E1 : zeta estimée, série lacunaire
python scripts/figure_e1_cascade.py      # E1 : zeta estimée, cascade (20 réalisations)
```

## Conventions

- Coefficients en normalisation L∞ : c_{j,k} = 2^{j/2} d_{j,k} / √n, où les d_{j,k}
  sont les coefficients de la transformée discrète et n = 2^J le nombre de points.
- Le niveau j contient 2^j coefficients ; la régression porte par défaut sur j ∈ [5, 14].
- Toutes les simulations utilisent un générateur `numpy` à graine fixée, passé en
  argument : les figures sont reproductibles à l'identique.

Ayman Bouchakouk