# User guide

This guide takes you from a fresh install to an exploited simulation, from
scratch. It is deliberately practical and example-driven.

- Project overview → [`README.md`](../README.md)
- Internal code architecture → [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Full case-file grammar → `./build/run --list`

> The solver is **entirely driven by a `.ini` case file**: no C++ to write for
> a new case.

---

## 1. Install (once)

Requirements: macOS 15+, Command Line Tools, CMake ≥ 3.24 (Apple Silicon).

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Check that everything works:

```sh
./build/run cases/sod.ini      # Sod shock tube (fast)
```

---

## 2. Run an existing case

The bundled cases live in `cases/`. Two ways to run:

```sh
# headless: compute and write the VTK output
./build/run cases/dmr.ini

# real-time: Metal window (if [render] live = true in the case)
./build/run cases/bubble.ini
```

In the live window: **space** = pause, **q** or close = quit.

### The 4 `run` commands

| Command | Effect |
|---|---|
| `run case.ini` | runs the simulation |
| `run --check case.ini` | parses + prints the effective config, the cost estimate, and **flags unknown keys** (typos) — without computing |
| `run --preview case.ini` | writes the **initial condition** as `.vti` (viewable before running) |
| `run --list` | case-grammar cheat sheet |

Make `--check` a reflex before every run: it catches typos and gives you the
expected time step / step count.

---

## 3. Author your own case in 10 minutes

Let's build a **shock hitting a light bubble**. Copy the commented template
and edit it:

```sh
cp cases/TEMPLATE.ini cases/mycase.ini
```

Fill it section by section (check with `--check` at each step).

**a) Global settings + domain + grid**
```ini
backend = hybrid     # cpu (reference) | hybrid (Metal GPU)
t_end   = 0.4
cfl     = 0.4
mu      = 0          # > 0 -> Navier-Stokes (viscous)

[domain]
x0 = 0
x1 = 2
y0 = 0
y1 = 1

[grid]
nx = 256             # SQUARE cells: nx/ny = (x1-x0)/(y1-y0)
ny = 128
```

**b) Two-gas + named primitive states**
```ini
[species]            # optional: two gammas for a two-gas case
gamma1 = 1.4         # air
gamma2 = 1.667       # light gas

[state.air]
rho = 1.4
p   = 1.0

[state.bubble]
rho = 0.2            # light bubble
p   = 1.0
gas = 2             # attach the state to the second gas

[state.shock]
shock = air mach 1.5 +x   # Rankine-Hugoniot post-shock computed for you
```

**c) Initial condition (stacked regions: the last one wins)**
```ini
[ic]
default  = air
region.1 = circle 0.6 0.5 0.2 : bubble            # bubble at the center
region.2 = halfplane 1 0 0.3 speed auto : shock   # front at x=0.3, RH speed
```
Shapes: `halfplane a b c [speed s]`, `band x|y lo hi`, `rect x0 x1 y0 y1`,
`circle cx cy r`, `sinex x0 amp lambda`. `speed auto` advances the front at
the Rankine-Hugoniot shock speed.

**d) Boundary conditions (per side)**
```ini
[bc]
left   = analytic    # the moving front provides the exact inflow
right  = transmissive
bottom = reflective
top    = reflective
```
Types: `transmissive`, `reflective`, `noslip` (viscous wall, `mu>0`),
`analytic` (re-evaluates the regions at time t — exact moving-shock BC),
`inflow X`. Segmentable: `... if x < val else ...`. Or `x|y = periodic`.

**e) AMR + output + live view**
```ini
[amr]
enabled       = true
block         = 8
levels        = 3        # 3 levels -> finest = base/2^2
tag_threshold = 0.05     # refine where |grad rho| exceeds this
regrid_every  = 4
subcycle      = true

[output]
frames = 20              # 20 VTK images spaced in time
prefix = out/mycase      # -> out/mycase_0001.vthb …

[render]
live  = true
scale = 4                # pixels per base cell
grid  = true             # AMR patch outlines
```

Check then run:
```sh
./build/run --check cases/mycase.ini    # config OK? cost?
./build/run cases/mycase.ini
```

That's all — no code. For the exhaustive grammar: `run --list`.

---

## 4. Reading the log

Enable the CSV log in the case:
```ini
[diagnostics]
every = 50                    # one line every 50 base steps (0 = off)
file  = out/mycase_log.csv    # default: <prefix>_log.csv
```

Columns:

| Column | Meaning |
|---|---|
| `step, t, dt` | iteration, physical time, time step |
| `res_mass, res_momx, res_momy, res_energy` | **residuals** (L1 change per step) — drop toward 0 as the flow reaches steady state |
| `cells, patches` | number of active cells and AMR patches (tracks refinement) |
| `rho_min/max, p_min/max` | extrema — watch **positivity** (rho, p > 0) |
| `mass` | total mass — its **drift** measures conservation (≈ fp32 floor on a closed domain; nonzero with open boundaries) |
| `kinetic_energy, total_energy` | energy budgets |
| `enstrophy` | ∫ω²/2 — vortical intensity (KH, RT…) |
| `species_mass` | mass of gas 2 (two-gas case) — must be conserved |
| `wall_s, mcells_per_s` | wall-clock time and throughput (Mcell-steps/s) |

What to watch: **residuals** (convergence), **mass drift** (conservation),
**extrema** (positivity/stability), **patches** (is refinement tracking the
structures), **mcells_per_s** (performance).

---

## 5. Working with the output

### ParaView (full field, all AMR levels)
The `.vthb` files (vtkOverlappingAMR) open directly in **ParaView**: full AMR
hierarchy, patch outlines, all variables (`rho, u, v, p`, and `Y` in two-gas).

### Real-time view
`[render] live = true` (backend `hybrid`): zero-copy Metal window during the
computation, auto color scale. Ideal for iterating on a case.

### Video / schlieren (bundled post-processing)
`tools/schlieren_video.py` turns the `.vthb` files into a punchy video
(numerical schlieren |∇ρ|, no ParaView):
```sh
python3 tools/schlieren_video.py --prefix out/mycase --full \
    --style light --cmap magma_r \
    --frames-dir out/mycase_frames --out out/mycase.mp4
```
Useful options: `--amr-panel` (density + AMR blocks panel under the
schlieren), `--annotate` + `--still last` (annotated figure), `--fps`,
`--freeze SEC` (final freeze), `--start/--end` (subsequence). Dependencies:
`vtk`, `matplotlib`, `ffmpeg`.

### Curves
`tools/plot_convergence.py`, `tools/plot_benchmark.py` (from the
`convergence` / `benchmark` drivers).

---

## 6. Common settings & troubleshooting

| Symptom / need | Lever |
|---|---|
| Unstable / NaN | lower `cfl` (0.4 → 0.3); check the extrema in the log |
| Too slow | `backend = hybrid` (GPU); lower `levels`; raise `tag_threshold` |
| Not enough detail | higher `levels` or lower `tag_threshold` |
| "AMR GPU slot pool exhausted" | too many patches: lower `levels`, refine less, or raise `amr.max_patches` (see the message) |
| Compare CPU vs GPU | same case in `backend = cpu` then `hybrid` (bit-identical results expected) |
| High order in smooth regions | `scheme = weno5` |
| Check the config | `run --check` (flags unknown keys) |

Notes:
- **Square cells**: keep `nx/ny = (x1-x0)/(y1-y0)`, otherwise anisotropic
  accuracy (`--check` flags it).
- `grid.nx`, `grid.ny` must be **multiples of `amr.block`**.
- `#` **and** `;` start a comment — one key per line.

---

*Going further: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) (internals),
[`ROADMAP.md`](../ROADMAP.md) (milestones and design choices).*
