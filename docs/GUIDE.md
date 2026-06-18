# Guide utilisateur

Ce guide t'amène d'une installation à une simulation exploitée, en partant
de zéro. Il est volontairement pratique et orienté exemple.

- Vue d'ensemble du projet → [`README.md`](../README.md)
- Architecture interne du code → [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Grammaire complète des fichiers de cas → `./build/run --list`

> Le solveur est **entièrement piloté par un fichier de cas `.ini`** :
> aucun C++ à écrire pour un nouveau cas.

---

## 1. Installer (une fois)

Prérequis : macOS 15+, Command Line Tools, CMake ≥ 3.24 (Apple Silicon).

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Vérifie que tout marche :

```sh
./build/run cases/sod.ini      # tube à choc de Sod (rapide)
```

---

## 2. Lancer un cas existant

Les cas livrés sont dans `cases/`. Deux façons de lancer :

```sh
# headless : calcule et écrit les sorties VTK
./build/run cases/dmr.ini

# temps réel : fenêtre Metal (si [render] live = true dans le cas)
./build/run cases/bubble.ini
```

Dans la fenêtre live : **espace** = pause, **q** ou fermeture = quitter.

### Les 4 commandes de `run`

| Commande | Effet |
|---|---|
| `run cas.ini` | lance la simulation |
| `run --check cas.ini` | parse + affiche la config effective, l'estimation de coût, et **signale les clés inconnues** (fautes de frappe) — sans calculer |
| `run --preview cas.ini` | écrit la **condition initiale** en `.vti` (visualisable avant de lancer) |
| `run --list` | aide-mémoire de la grammaire des cas |

Prends le réflexe de `--check` avant tout run : il attrape les coquilles et
te donne le pas de temps / le nombre d'étapes attendus.

---

## 3. Poser son propre cas en 10 minutes

On va construire un **choc qui percute une bulle légère**. Copie le modèle
commenté et édite-le :

```sh
cp cases/TEMPLATE.ini cases/moncas.ini
```

On remplit section par section (vérifie avec `--check` à chaque étape).

**a) Réglages globaux + domaine + grille**
```ini
backend = hybrid     # cpu (référence) | hybrid (GPU Metal)
t_end   = 0.4
cfl     = 0.4
mu      = 0          # > 0 -> Navier-Stokes (visqueux)

[domain]
x0 = 0
x1 = 2
y0 = 0
y1 = 1

[grid]
nx = 256             # cellules CARRÉES : nx/ny = (x1-x0)/(y1-y0)
ny = 128
```

**b) Bi-gaz + états primitifs nommés**
```ini
[species]            # optionnel : deux gamma pour un cas bi-gaz
gamma1 = 1.4         # air
gamma2 = 1.667       # gaz léger

[state.air]
rho = 1.4
p   = 1.0

[state.bulle]
rho = 0.2            # bulle légère
p   = 1.0
gas = 2             # rattache l'état au second gaz

[state.choc]
shock = air mach 1.5 +x   # post-choc Rankine-Hugoniot calculé pour toi
```

**c) Condition initiale (régions empilées : la dernière l'emporte)**
```ini
[ic]
default  = air
region.1 = circle 0.6 0.5 0.2 : bulle             # bulle au centre
region.2 = halfplane 1 0 0.3 speed auto : choc    # front à x=0.3, vitesse RH
```
Formes : `halfplane a b c [speed s]`, `band x|y lo hi`, `rect x0 x1 y0 y1`,
`circle cx cy r`, `sinex x0 amp lambda`. `speed auto` fait avancer le front
à la vitesse du choc Rankine-Hugoniot.

**d) Conditions aux limites (par côté)**
```ini
[bc]
left   = analytic    # le front mobile fournit l'inflow exact
right  = transmissive
bottom = reflective
top    = reflective
```
Types : `transmissive`, `reflective`, `noslip` (paroi visqueuse, `mu>0`),
`analytic` (réévalue les régions au temps t — BC exacte d'un choc mobile),
`inflow X`. Segmentable : `... if x < val else ...`. Ou `x|y = periodic`.

**e) AMR + sorties + vue live**
```ini
[amr]
enabled       = true
block         = 8
levels        = 3        # 3 niveaux -> finest = base/2^2
tag_threshold = 0.05     # raffine où |grad rho| dépasse ce seuil
regrid_every  = 4
subcycle      = true

[output]
frames = 20              # 20 images VTK espacées dans le temps
prefix = out/moncas      # -> out/moncas_0001.vthb …

[render]
live  = true
scale = 4                # pixels par cellule de base
grid  = true             # contours des patchs AMR
```

Vérifie puis lance :
```sh
./build/run --check cases/moncas.ini    # config OK ? coût ?
./build/run cases/moncas.ini
```

C'est tout — pas de code. Pour la grammaire exhaustive : `run --list`.

---

## 4. Lire le journal

Active le journal CSV dans le cas :
```ini
[diagnostics]
every = 50                    # une ligne tous les 50 pas de base (0 = off)
file  = out/moncas_log.csv    # défaut : <prefix>_log.csv
```

Colonnes :

| Colonne | Sens |
|---|---|
| `step, t, dt` | itération, temps physique, pas de temps |
| `res_mass, res_momx, res_momy, res_energy` | **résidus** (variation L1 par pas) — chutent vers 0 quand l'écoulement se stationnarise |
| `cells, patches` | nombre de cellules actives et de patchs AMR (suit le raffinement) |
| `rho_min/max, p_min/max` | extrema — surveille la **positivité** (rho, p > 0) |
| `mass` | masse totale — sa **dérive** mesure la conservation (≈ plancher fp32 sur domaine fermé ; non nulle si bords ouverts) |
| `kinetic_energy, total_energy` | bilans d'énergie |
| `enstrophy` | ∫ω²/2 — intensité tourbillonnaire (KH, RT…) |
| `species_mass` | masse du gaz 2 (cas bi-gaz) — doit se conserver |
| `wall_s, mcells_per_s` | temps mural et débit (Mcell-steps/s) |

À regarder : **résidus** (convergence), **dérive de masse** (conservation),
**extrema** (positivité/stabilité), **patches** (le raffinement suit-il les
structures), **mcells_per_s** (perf).

---

## 5. Exploiter les sorties

### ParaView (champ complet, tous niveaux AMR)
Les `.vthb` (vtkOverlappingAMR) s'ouvrent directement dans **ParaView** :
hiérarchie AMR complète, contours de patchs, toutes les variables
(`rho, u, v, p`, et `Y` en bi-gaz).

### Vue temps réel
`[render] live = true` (backend `hybrid`) : fenêtre Metal zéro-copie pendant
le calcul, échelle de couleur auto. Idéal pour itérer sur un cas.

### Vidéo / schlieren (post-traitement fourni)
`tools/schlieren_video.py` transforme les `.vthb` en vidéo qui « claque »
(schlieren numérique |∇ρ|, sans ParaView) :
```sh
python3 tools/schlieren_video.py --prefix out/moncas --full \
    --style light --cmap magma_r \
    --frames-dir out/moncas_frames --out out/moncas.mp4
```
Options utiles : `--amr-panel` (panneau densité + blocs AMR sous le
schlieren), `--annotate` + `--still last` (figure annotée), `--fps`,
`--freeze SEC` (figé final), `--start/--end` (sous-séquence). Dépendances :
`vtk`, `matplotlib`, `ffmpeg`.

### Courbes
`tools/plot_convergence.py`, `tools/plot_benchmark.py` (depuis les drivers
`convergence` / `benchmark`).

---

## 6. Réglages courants & dépannage

| Symptôme / besoin | Levier |
|---|---|
| Instable / NaN | baisser `cfl` (0.4 → 0.3) ; vérifier les extrema dans le journal |
| Trop lent | `backend = hybrid` (GPU) ; baisser `levels` ; `tag_threshold` plus haut |
| Pas assez de détail | `levels` plus haut ou `tag_threshold` plus bas |
| « AMR GPU slot pool exhausted » | trop de patchs : baisser `levels`, raffiner moins, ou monter `amr.max_patches` (cf. message) |
| Comparer CPU vs GPU | même cas en `backend = cpu` puis `hybrid` (résultats bit-identiques attendus) |
| Ordre élevé en régime lisse | `scheme = weno5` |
| Vérifier la config | `run --check` (signale les clés inconnues) |

Notes :
- **Cellules carrées** : garde `nx/ny = (x1-x0)/(y1-y0)`, sinon précision
  anisotrope (`--check` le signale).
- `grid.nx`, `grid.ny` doivent être **multiples de `amr.block`**.
- `#` **et** `;` démarrent un commentaire — une clé par ligne.

---

*Pour aller plus loin : [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) (interne),
[`ROADMAP.md`](../ROADMAP.md) (jalons et choix de conception).*
