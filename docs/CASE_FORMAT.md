# Référence du format de cas `.ini`

Référence exhaustive du fichier de cas (`CaseDef` + `AmrConfig`). Pour un
tutoriel pas-à-pas, voir [`docs/GUIDE.md`](GUIDE.md) ; pour l'aide-mémoire
rapide : `./build/run --list`.

Conventions du fichier : format INI ; `#` **et** `;` démarrent un
commentaire ; **une clé par ligne**. Les clés inconnues sont signalées par
`./build/run --check`.

---

## Clés de premier niveau (hors section)

| Clé | Défaut | Sens |
|---|---|---|
| `t_end` | 0.2 | temps physique final |
| `cfl` | 0.4 | nombre CFL |
| `mu` | 0 | viscosité dynamique ; `> 0` → Navier-Stokes (incompatible bi-gaz) |
| `backend` | `hybrid` | `cpu` (référence) \| `hybrid` (GPU Metal) |
| `scheme` | `muscl` | `muscl` (MUSCL-Hancock) \| `weno5` (WENO5 + RK3 ; mono-gaz) |
| `restart` | — | chemin d'un checkpoint pour reprendre |

---

## `[domain]` — domaine physique

| Clé | Sens |
|---|---|
| `x0, x1, y0, y1` | bornes du rectangle |

## `[grid]` — résolution de base

| Clé | Sens |
|---|---|
| `nx, ny` | cellules de base (**multiples de `amr.block`**) |

> Garder des **cellules carrées** : `nx/ny = (x1-x0)/(y1-y0)`, sinon
> précision anisotrope (`--check` propose la valeur).

## `[physics]` — sources

| Clé | Sens |
|---|---|
| `gravity = gx gy` | gravité (source splittée) ; les murs `reflective` deviennent hydrostatiques |

---

## `[species]` — bi-gaz (optionnel)

Active le modèle deux-gaz (transport de la fraction massique `Y` via
`phi = rho·Y`, fermeture `Gamma` quasi-conservative). Incompatible `mu > 0`.

| Clé | Défaut | Sens |
|---|---|---|
| `gamma1` | 1.4 | γ du gaz 1 (défaut des états) |
| `gamma2` | 1.4 | γ du gaz 2 (états avec `gas = 2`) |

## `[reaction]` — combustion (optionnel)

Source Arrhenius mono-étape sur la variable de progrès λ (implique
`species`, avec `gamma1 = gamma2 = gamma`). Strang-split autour de l'hydro.

| Clé | Sens |
|---|---|
| `A` | facteur pré-exponentiel |
| `Ea` | énergie d'activation |
| `q` | dégagement de chaleur |
| `Tign` | température d'ignition (cutoff) |
| `gamma` | γ commun (défaut 1.4) |

---

## `[state.NOM]` — états primitifs nommés

Un état par section `[state.<nom>]`. Référencé par les régions de l'IC, les
BC `inflow`, et `ic.default`.

| Clé | Défaut | Sens |
|---|---|---|
| `rho` | 1 | masse volumique |
| `u, v` | 0 | vitesse |
| `p` | 1 | pression |
| `gas` | 1 | 1 ou 2 (cas bi-gaz ; `gas = 2` nécessite `[species]`) |
| `shock` | — | **état dérivé** Rankine-Hugoniot (voir ci-dessous) |

**État post-choc dérivé** : `shock = <amont> mach <Ms> [+x|-x|+y|-y]`
calcule l'état post-choc R-H pour un choc de Mach `Ms` (> 1) se propageant
dans l'état `<amont>` (non dérivé), direction par défaut `+x`. Le solveur
mémorise la vitesse lab du front (utilisée par `speed auto`).

---

## `[ic]` — condition initiale

```ini
default = NOM            # état partout, avant les régions
region.N = <forme> : NOM # régions empilées (N=1..99, la DERNIÈRE l'emporte)
perturb.N = <perturbation>
```

### Formes de région

| Forme | Sémantique |
|---|---|
| `halfplane a b c [speed s]` | demi-plan `a·x + b·y < c` ; `speed s` fait avancer le front à la vitesse normale `s` ; `speed auto` = vitesse R-H de l'état (état `shock=`) |
| `band x lo hi` / `band y lo hi` | bande `lo < x < hi` (ou en y) |
| `rect x0 x1 y0 y1` | rectangle |
| `circle cx cy r` | disque de centre (cx,cy), rayon r |
| `sinex x0 amp lambda` | interface en cosinus : `x < x0 + amp·cos(2π y/lambda)` |

### Perturbations (appliquées après les régions)

| Forme | Effet sur la variable `var ∈ {u,v,rho,p}` |
|---|---|
| `var sin periods amp` | `+ amp·sin(2π·periods·(x-x0)/lx)` |
| `var erf x0 width amp` | `+ amp·erf((x-x0)/width)` |
| `var sing periods amp yc sigma` | sinus en x × enveloppe gaussienne en y (centre yc, σ) |
| `p hydro yref` | pression hydrostatique : `p += rho·gy·(y-yref)` |

---

## `[bc]` — conditions aux limites

```ini
x = periodic            # ou y = periodic (sinon, par côté)
left|right|bottom|top = <spec>
```

| Type | Effet |
|---|---|
| `transmissive` | gradient nul (sortie) |
| `reflective` | mur glissant (composante normale miroir ; hydrostatique sous gravité) |
| `noslip` | mur visqueux adhérent (les deux composantes miroir ; `mu > 0`) |
| `analytic` | réévalue la pile de régions au temps `t` dans les ghosts → **BC exacte d'un choc mobile** (haut du DMR) |
| `inflow NOM` | impose l'état primitif `NOM` |

**Côté segmenté** : `<specA> if x < val else <specB>` (ou `if y < val`) —
ex. le bas du DMR : `analytic if x < 0.1667 else reflective`.

---

## `[amr]` — raffinement adaptatif

| Clé | Défaut | Sens |
|---|---|---|
| `enabled` | true | si `false`, grille de base seule |
| `block` | 8 | taille de bloc (cellules grossières) |
| `levels` | 2 | nombre total de niveaux (base + raffinements) |
| `tag_threshold` | 0.08 | seuil de raffinement sur `|grad rho|` relatif |
| `tag_velocity` | 0 | seuil sur le saut de vitesse / vitesse du son (0 = off) |
| `regrid_every` | 4 | re-tag tous les K pas de base |
| `subcycle` | false | true = subcycling Berger-Colella (fin à dt/2) |
| `max_patches` | 0 | cap du pool de slots GPU (0 = auto ~1/8 du working set) |

## `[output]` — sorties VTK / checkpoint

| Clé | Défaut | Sens |
|---|---|---|
| `frames` | 4 | nombre d'images VTK espacées dans le temps |
| `every` | 0 | une image tous les K pas (prioritaire sur `frames` si > 0) |
| `prefix` | — | préfixe des fichiers (`<prefix>_0001.vthb`…) |
| `checkpoint` | — | préfixe de checkpoint |
| `max_steps` | 0 | arrêt après N pas (0 = illimité) |

## `[render]` — vue temps réel (backend `hybrid`)

| Clé | Défaut | Sens |
|---|---|---|
| `live` | false | ouvrir la fenêtre Metal |
| `scale` | 4 | pixels par cellule de base |
| `every` | 2 | rafraîchir tous les K pas de base |
| `grid` | true | dessiner les contours des patchs AMR |
| `rho_min, rho_max` | 0, 0 | plage de couleur (0/0 = auto-échelle) |

Contrôles fenêtre : **espace** = pause, **q** / fermeture = quitter.

## `[diagnostics]` — journal CSV

| Clé | Défaut | Sens |
|---|---|---|
| `every` | 0 | une ligne tous les K pas de base (0 = off) |
| `file` | `<prefix>_log.csv` | chemin du CSV |

Colonnes : `step, t, dt, res_{mass,momx,momy,energy}, cells, patches,
rho_min/max, p_min/max, mass, kinetic_energy, total_energy, enstrophy,
species_mass, wall_s, mcells_per_s` (détail dans [`GUIDE.md`](GUIDE.md#4-lire-le-journal)).

---

## Exemple minimal (Sod)

```ini
backend = cpu
t_end = 0.2
[domain]
x0 = 0
x1 = 1
y0 = 0
y1 = 0.25
[grid]
nx = 128          # multiples de amr.block (8)
ny = 32
[state.gauche]
rho = 1
p = 1
[state.droite]
rho = 0.125
p = 0.1
[ic]
default = droite
region.1 = halfplane 1 0 0.5 : gauche
[bc]
left = transmissive
right = transmissive
y = periodic
[output]
frames = 1
prefix = out/sod
```

Voir `cases/` pour des exemples complets (DMR, bulle, RM, détonation,
déflagration, Blasius…) et `cases/TEMPLATE.ini` (modèle commenté).
