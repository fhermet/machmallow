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
- [ ] Phase 6 — profiling et consolidation
- [ ] Phase 7 — subcycling, termes visqueux (Navier-Stokes)

## Validation

Tubes à choc de Sod 1D/2D (vs solution exacte de Riemann), Double Mach
Reflection (Woodward & Colella 1984).
