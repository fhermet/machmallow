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
- [ ] Phase 1 — solveur 1D CPU (Sod, HLLC + MUSCL-Hancock)
- [ ] Phase 2 — 2D uniforme CPU + writer VTK
- [ ] Phase 3 — port GPU du solveur 2D
- [ ] Phase 4 — AMR 2 niveaux CPU (tagging, ghost fill, refluxing)
- [ ] Phase 5 — AMR hybride CPU/GPU, Double Mach Reflection
- [ ] Phase 6 — profiling et consolidation
- [ ] Phase 7 — subcycling, termes visqueux (Navier-Stokes)

## Validation

Tubes à choc de Sod 1D/2D (vs solution exacte de Riemann), Double Mach
Reflection (Woodward & Colella 1984).
