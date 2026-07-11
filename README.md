# machmallow 🔥🧁

*Soft on the outside, supersonic on the inside.*

A 2D compressible CFD solver (Euler and Navier–Stokes) with block-structured
hybrid CPU/GPU AMR, written from scratch in C++/Metal for Apple Silicon.

## Features

- **Schemes**: 2nd-order MUSCL-Hancock with an HLLC Riemann solver; optional
  WENO5 + RK3 (single-gas). Optional centered viscous fluxes for the
  compressible Navier–Stokes equations (Pr = 0.72), including adiabatic
  no-slip walls.
- **AMR**: block-structured patch hierarchy (AMReX-style) of arbitrary depth
  (`amr.levels`). The CPU handles regridding while the GPU computes fluxes on
  every level (all levels share one slot pool). Recursive Berger–Colella
  subcycling (time-interpolated ghosts, pairwise reflux, guaranteed nesting on
  regrid).
- **Immersed solids**: static geometric bodies (circle, rectangle, triangle,
  half-plane, …) masked on the Cartesian grid, treated as reflective slip
  walls — or viscous no-slip walls when viscosity is on. AMR auto-refines the
  body boundary.
- **Multi-physics**: optional two-gas model (mass-fraction transport,
  quasi-conservative Gamma closure) and single-step Arrhenius reaction for
  detonation/deflagration.
- **GPU**: Metal via [metal-cpp](https://developer.apple.com/metal/cpp/),
  shaders compiled at runtime (no Xcode required), zero-copy shared buffers
  (unified memory).
- **Precision**: float32.
- **Output**: VTK (`vtkOverlappingAMR`) for ParaView, plus an optional
  real-time Metal render window.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

Requirements: macOS 15+, Command Line Tools, CMake ≥ 3.24. No external
dependencies — metal-cpp is vendored under `third_party/`.

## Running a case

```sh
./build/run cases/dmr.ini              # Double Mach Reflection (hybrid subcycled AMR)
./build/run cases/sod.ini              # Sod shock tube
./build/run cases/cylinder_bowshock.ini # Mach 2 bow shock on an immersed cylinder
./build/run cases/shear.ini            # viscous shear layer (Navier–Stokes)
```

**The solver is entirely driven by the case file** — no per-case C++. A case
declares the domain, named primitive states, the initial condition as stacked
geometric regions (half-planes — including *moving* shock fronts —, bands,
rectangles, circles) with perturbations, immersed solids, and per-side
boundary conditions (`transmissive`, `reflective`, `noslip`, `inflow`,
`reservoir`, `backpressure`, segmentable, periodic — plus `analytic`, which
re-evaluates the regions at time *t* in the ghost cells: the exact moving-shock
BC in one line). Plus backend, viscosity, reactions, AMR (`levels`, tagging,
subcycling) and output settings.

To create a case, copy the commented `cases/TEMPLATE.ini`, then use
`./build/run --check mycase.ini` (effective config + unknown-key warnings) and
`./build/run --list` (grammar reference). There are 22 ready-made cases in
`cases/` (Sod, DMR, shear/Blasius, cylinder, nozzle, detonation/deflagration,
Kelvin–Helmholtz, Rayleigh–Taylor, Richtmyer–Meshkov, shock–bubble, …).

Render the `.vthb` output to video with `tools/schlieren_video.py`
(schlieren, vorticity, density; ParaView also opens the files directly).

## Documentation

- [`docs/GUIDE.md`](docs/GUIDE.md) — user guide: set up a case in 10 min, read
  the log, work with the output.
- [`docs/CASE_FORMAT.md`](docs/CASE_FORMAT.md) — exhaustive `.ini` case-file
  reference.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — code architecture (layers,
  AMR, hybrid CPU/GPU; Mermaid diagrams).
- [`docs/NUMERICS.md`](docs/NUMERICS.md) — numerical methods (equations, HLLC,
  MUSCL/WENO5, Berger–Colella AMR).
- [`docs/DEVELOPER.md`](docs/DEVELOPER.md) — developer guide: contributing,
  validation discipline, conventions.
- [`docs/VALIDATION.md`](docs/VALIDATION.md) — verification & validation:
  order of accuracy, conservation, CPU/GPU lock-step, vs exact solutions and
  experiments (with numbers).

## Performance (Apple M4, 10-core GPU, 16 GB)

Double Mach Reflection, 2-level AMR, t = 0.2, CFL 0.4 (`dmr_amr`):

| Finest resolution | Steps | Time | Throughput | Work vs uniform |
|---|---|---|---|---|
| 1/256 (coarse 512×128) | 2706 | ~1.8–2.8 s | 86–135 Mcell/s | 34 % |
| 1/512 (coarse 1024×256) | 5624 | ~9.7 s | ~180 Mcell/s | 30 % |

Breakdown of a hybrid step (1/512): GPU ~80 % (compute + 1 sync/step), ghost
fill ~10 %, regrid ~6 %, reflux + restriction ~4 %. Optimal block size: 8
coarse cells. Expect run-to-run variance on Apple Silicon (GPU frequency
governor): ±30 % on small cases.

Reference points: uniform 2D GPU solver ~300 Mcell/s (≈10× single-thread CPU);
hybrid AMR ≈4× single-thread CPU AMR at equal resolution.

## Validation

Sod shock tubes 1D/2D (vs exact Riemann solution), Double Mach Reflection
(Woodward & Colella 1984), viscous shear layer (vs exact erf profile), Blasius
boundary layer, oblique-shock wedge, and immersed-body cases. The CPU
validation harnesses run in CI on every push (see `.github/workflows/ci.yml`).
Development follows a strict CPU/GPU lock-step discipline (the CPU path is the
reference oracle).

## Roadmap

Milestones and planned work (multi-level AMR, WENO, cut-cells, real-time
rendering, …) are tracked in [ROADMAP.md](ROADMAP.md).

## License

[MIT](LICENSE) © Florian Hermet. Vendored [metal-cpp](third_party/metal-cpp)
is licensed by Apache under its own terms (see
`third_party/metal-cpp/LICENSE.txt`).
