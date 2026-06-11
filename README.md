# machmallow 🔥🧁

*Soft on the outside, supersonic on the inside.*

Solveur CFD 2D compressible (Euler, puis Navier-Stokes) avec AMR
block-structured hybride CPU/GPU, écrit from scratch en C++/Metal pour
Apple Silicon.

## Caractéristiques

- **Schéma** : MUSCL-Hancock ordre 2 + solveur de Riemann HLLC
- **AMR** : hiérarchie de patchs block-structured (style AMReX) —
  le CPU gère le regridding, le GPU calcule les flux
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

## Feuille de route

- [x] Phase 0 — socle : CMake, metal-cpp, benchmark saxpy
- [x] Phase 1 — solveur 1D CPU (Sod, HLLC + MUSCL-Hancock, ordre observé 0.9 vs exact)
- [x] Phase 2 — 2D uniforme CPU + writer VTK (Sod diagonal ordre 0.97, DMR 480×120 OK)
- [x] Phase 3 — port GPU du solveur 2D (écart fp32 ~1e-5 vs CPU, 298 Mcell/s à 2880×720, ~10× CPU)
- [x] Phase 4 — AMR 2 niveaux CPU (L1 = 1.05× l'uniforme fin pour 63% du travail, conservation au plancher fp32)
- [x] Phase 5 — AMR hybride CPU/GPU (DMR 1/512 : 150 Mcell/s, 30% du travail uniforme, 4.4× l'AMR CPU)
- [x] Phase 6 — profiling et consolidation (ghost fill par bandes + CPU parallèle via GCD)
- [ ] Phase 7 — subcycling, termes visqueux (Navier-Stokes)

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
Reflection (Woodward & Colella 1984).
