# machmallow 🔥🧁

*Soft on the outside, supersonic on the inside.*

Solveur CFD 2D compressible (Euler, puis Navier-Stokes) avec AMR
block-structured hybride CPU/GPU, écrit from scratch en C++/Metal pour
Apple Silicon.

## Caractéristiques

- **Schéma** : MUSCL-Hancock ordre 2 + solveur de Riemann HLLC ;
  flux visqueux centrés optionnels (Navier-Stokes compressible, Pr = 0.72)
- **AMR** : hiérarchie de patchs block-structured (style AMReX) à
  profondeur arbitraire (`amr.levels`) — le CPU gère le regridding, le
  GPU calcule les flux (tous les niveaux partagent un pool de slots) ;
  subcycling Berger-Colella récursif (ghosts interpolés en temps,
  refluxing par paire de niveaux, nesting garanti au regrid)
- **GPU** : Metal via metal-cpp, shaders compilés au runtime (pas de Xcode requis),
  buffers partagés zéro-copie (mémoire unifiée)
- **Précision** : float32
- **Sorties** : VTK (vtkOverlappingAMR) pour ParaView

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/machmallow
```

Prérequis : macOS 15+, Command Line Tools, CMake ≥ 3.24.

## Lancer un cas

```sh
./build/run cases/dmr.ini      # DMR, AMR hybride subcyclé, sorties .vthb
./build/run cases/sod.ini      # tube à choc de Sod
./build/run cases/shear.ini    # couche de cisaillement visqueuse (NS)
```

**Le solveur est entièrement piloté par le fichier de cas** — aucun
C++ par cas. Le fichier déclare le domaine, des états primitifs nommés,
la condition initiale par régions géométriques (demi-plans — y compris
fronts de choc *mobiles* —, bandes, rectangles, cercles) avec
perturbations, et les conditions aux limites par côté (`transmissive`,
`reflective`, `inflow`, segmentables, périodiques — et `analytic` qui
réévalue les régions au temps t dans les ghosts : la BC exacte du DMR
en une ligne). Plus backend, viscosité, AMR (`levels`, tagging,
subcycling) et sorties.

Pour créer un cas : copier `cases/TEMPLATE.ini` (commenté), puis
`./build/run --check moncas.ini` (config effective + warnings de clés
inconnues) et `./build/run --list` (grammaire). L'équivalence du système
déclaratif avec les anciens presets C++ est verrouillée par
`casedef_test` (Sod au L1 historique exact, ghosts DMR identiques
cellule pour cellule). Les exécutables `sod1d`…`mlgpu_amr` restent les
harnais de validation.

## Feuille de route

- [x] Phase 0 — socle : CMake, metal-cpp, benchmark saxpy
- [x] Phase 1 — solveur 1D CPU (Sod, HLLC + MUSCL-Hancock, ordre observé 0.9 vs exact)
- [x] Phase 2 — 2D uniforme CPU + writer VTK (Sod diagonal ordre 0.97, DMR 480×120 OK)
- [x] Phase 3 — port GPU du solveur 2D (écart fp32 ~1e-5 vs CPU, 298 Mcell/s à 2880×720, ~10× CPU)
- [x] Phase 4 — AMR 2 niveaux CPU (L1 = 1.05× l'uniforme fin pour 63% du travail, conservation au plancher fp32)
- [x] Phase 5 — AMR hybride CPU/GPU (DMR 1/512 : 150 Mcell/s, 30% du travail uniforme, 4.4× l'AMR CPU)
- [x] Phase 6 — profiling et consolidation (ghost fill par bandes + CPU parallèle via GCD)
- [x] Phase 7 — subcycling Berger-Colella (1.67× sur DMR 1/512) + termes visqueux NS (ordre 2.33 vs erf, parité GPU)

## Performances (Apple M4, 10 cœurs GPU, 16 GB)

Double Mach Reflection AMR 2 niveaux, t = 0.2, CFL 0.4, `dmr_amr` :

| Résolution fine | Pas | Temps | Débit | Travail vs uniforme |
|---|---|---|---|---|
| 1/256 (coarse 512×128) | 2706 | ~1.8–2.8 s | 86–135 Mcell/s | 34 % |
| 1/512 (coarse 1024×256) | 5624 | ~9.7 s | ~180 Mcell/s | 30 % |

Répartition d'un pas hybride (1/512) : GPU ~80 % (calcul + 1 sync/pas),
ghost fill ~10 %, regrid ~6 %, reflux + restriction ~4 %. Taille de bloc
optimale : 8 cellules grossières (sweep 4/8/16). Attention à la variance
run-à-run sur Apple Silicon (gouverneur de fréquence GPU) : ±30 % sur les
petits cas.

Jalons intermédiaires : solveur 2D uniforme GPU ~300 Mcell/s (≈10× le CPU
mono-thread) ; AMR hybride ≈4× l'AMR CPU mono-thread à résolution égale.

## Validation

Tubes à choc de Sod 1D/2D (vs solution exacte de Riemann), Double Mach
Reflection (Woodward & Colella 1984), couche de cisaillement visqueuse
(vs profil erf exact).

## Et après ?

Les extensions envisagées (AMR multi-niveaux, WENO, NS sur AMR, rendu
temps réel, CI…) sont détaillées dans [ROADMAP.md](ROADMAP.md).
