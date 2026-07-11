# `.ini` case-file reference

Exhaustive reference for the case file (`CaseDef` + `AmrConfig`). For a
step-by-step tutorial see [`docs/GUIDE.md`](GUIDE.md); for a quick cheat
sheet: `./build/run --list`.

File conventions: INI format; `#` **and** `;` start a comment; **one key per
line**. Unknown keys are flagged by `./build/run --check`.

---

## Top-level keys (no section)

| Key | Default | Meaning |
|---|---|---|
| `t_end` | 0.2 | final physical time |
| `cfl` | 0.4 | CFL number |
| `mu` | 0 | dynamic viscosity; `> 0` → Navier–Stokes (incompatible with two-gas) |
| `backend` | `hybrid` | `cpu` (reference) \| `hybrid` (Metal GPU) |
| `scheme` | `muscl` | `muscl` (MUSCL-Hancock) \| `weno5` (WENO5 + RK3; single-gas) |
| `restart` | — | path to a checkpoint to resume from |

---

## `[domain]` — physical domain

| Key | Meaning |
|---|---|
| `x0, x1, y0, y1` | rectangle bounds |

## `[grid]` — base resolution

| Key | Meaning |
|---|---|
| `nx, ny` | base cells (**multiples of `amr.block`**) |

> Keep **square cells**: `nx/ny = (x1-x0)/(y1-y0)`, otherwise anisotropic
> accuracy (`--check` suggests the value).

## `[physics]` — sources

| Key | Meaning |
|---|---|
| `gravity = gx gy` | gravity (split source); `reflective` walls become hydrostatic |

---

## `[species]` — two-gas (optional)

Enables the two-gas model (mass-fraction `Y` transport via `phi = rho·Y`,
quasi-conservative `Gamma` closure). Incompatible with `mu > 0`.

| Key | Default | Meaning |
|---|---|---|
| `gamma1` | 1.4 | γ of gas 1 (default for states) |
| `gamma2` | 1.4 | γ of gas 2 (states with `gas = 2`) |

## `[reaction]` — combustion (optional)

Single-step Arrhenius source on the progress variable λ (implies `species`,
with `gamma1 = gamma2 = gamma`). Strang-split around the hydrodynamics.

| Key | Meaning |
|---|---|
| `A` | pre-exponential factor |
| `Ea` | activation energy |
| `q` | heat release |
| `Tign` | ignition temperature (cutoff) |
| `gamma` | shared γ (default 1.4) |

---

## `[state.NAME]` — named primitive states

One state per `[state.<name>]` section. Referenced by IC regions, `inflow`
BCs, and `ic.default`.

| Key | Default | Meaning |
|---|---|---|
| `rho` | 1 | density |
| `u, v` | 0 | velocity |
| `p` | 1 | pressure |
| `gas` | 1 | 1 or 2 (two-gas case; `gas = 2` requires `[species]`) |
| `shock` | — | **derived state** Rankine–Hugoniot (see below) |

**Derived post-shock state**: `shock = <upstream> mach <Ms> [+x|-x|+y|-y]`
computes the post-shock R-H state for a shock of Mach `Ms` (> 1) propagating
into the (non-derived) state `<upstream>`, default direction `+x`. The solver
records the lab-frame front speed (used by `speed auto`).

---

## `[ic]` — initial condition

```ini
default = NAME           # state everywhere, before the regions
region.N = <shape> : NAME # stacked regions (N=1..99, the LAST one wins)
perturb.N = <perturbation>
```

### Region shapes

| Shape | Semantics |
|---|---|
| `halfplane a b c [speed s]` | half-plane `a·x + b·y < c`; `speed s` advances the front at normal speed `s`; `speed auto` = R-H speed of the state (a `shock=` state) |
| `band x lo hi` / `band y lo hi` | band `lo < x < hi` (or in y) |
| `rect x0 x1 y0 y1` | rectangle |
| `circle cx cy r` | disk centered at (cx,cy), radius r |
| `sinex x0 amp lambda` | cosine interface: `x < x0 + amp·cos(2π y/lambda)` |

### Perturbations (applied after the regions)

| Shape | Effect on variable `var ∈ {u,v,rho,p}` |
|---|---|
| `var sin periods amp` | `+ amp·sin(2π·periods·(x-x0)/lx)` |
| `var erf x0 width amp` | `+ amp·erf((x-x0)/width)` |
| `var sing periods amp yc sigma` | sine in x × Gaussian envelope in y (center yc, σ) |
| `p hydro yref` | hydrostatic pressure: `p += rho·gy·(y-yref)` |

---

## `[solid]` — immersed bodies (optional)

```ini
region.N = <shape>       # solid block (N=1..99; same grammar as [ic],
                         # WITHOUT a state or motion)
```

Cells whose center falls inside a `[solid]` region are removed from the flow;
fluid↔solid faces become **reflective slip walls** (mirrored normal
velocity), or **no-slip (adherent)** walls as soon as `mu > 0` (the viscous
fluxes impose zero wall velocity — validated on Blasius, gate
`immersed_noslip`). The shapes are the IC region shapes — `rect`, `circle`,
`halfplane`, `band`, `sinex` — plus **`triangle x1 y1 x2 y2 x3 y3`** (three
vertices, any order), without the `: NAME` (a solid has no state) or `speed`
(static).

```ini
[solid]
region.1 = rect 0.7 1.01 -0.01 0.03   # block / aligned wall
region.2 = circle 0.5 0.5 0.1         # cylinder (staircased)
```

The **AMR automatically refines the body boundary** (fluid cells touching a
solid are tagged) in addition to shocks — useful to smooth the staircase;
standard `[amr]` rules (`levels`, `tag_threshold`, `subcycle`…). See
`cases/cylinder_bowshock.ini` and `cases/wc_step.ini`.

> **Current limits** (v1.6): `backend = cpu` **or `hybrid`** (GPU, verified
> CPU/GPU lock-step — arbitrary AMR depth and viscous no-slip on both),
> `scheme = muscl` single-gas. TODO: cut-cells (to smooth the staircase). See
> the [roadmap](../ROADMAP.md). Reference case: `cases/shock_wall.ini` (gates
> `immersed_case`, `immersed_amr`, `immersed_gpu`).

---

## `[bc]` — boundary conditions

```ini
x = periodic            # or y = periodic (otherwise, per side)
left|right|bottom|top = <spec>
```

| Type | Effect |
|---|---|
| `transmissive` | zero gradient (outflow) |
| `reflective` | slip wall (mirrored normal component; hydrostatic under gravity) |
| `noslip` | adherent viscous wall (both components mirrored; `mu > 0`) |
| `analytic` | re-evaluates the region stack at time `t` in the ghosts → **exact moving-shock BC** (top of the DMR) |
| `inflow NAME` | imposes the primitive state `NAME` |
| `reservoir NAME` | **stagnation-condition inlet**: `NAME` gives stagnation (ρ0, p0); static pressure comes from the interior, the state is isentropic and the normal velocity adjusts (M deduced from p0/p) → non-reflecting, stable feed (nozzle chamber) |
| `backpressure t0 p0 t1 p1 ...` | **scheduled outlet pressure**: subsonic outflow → imposes a static pressure following a **piecewise-linear schedule** (time-pressure pairs, ≥2, increasing in `t`; constant before `t0` and after the last node); supersonic outflow → transmissive. ρ,u,v extrapolated. **Plateaus** (`… t1 p t2 p …`) PAUSE the flow on each regime; drives a regime sweep (nozzle, `cases/nozzle.ini`) |

**Segmented side**: `<specA> if x < val else <specB>` (or `if y < val`) —
e.g. the bottom of the DMR: `analytic if x < 0.1667 else reflective`.

---

## `[amr]` — adaptive mesh refinement

| Key | Default | Meaning |
|---|---|---|
| `enabled` | true | if `false`, base grid only |
| `block` | 8 | block size (coarse cells) |
| `levels` | 2 | total number of levels (base + refinements) |
| `tag_threshold` | 0.08 | refinement threshold on relative `|grad rho|` |
| `tag_velocity` | 0 | threshold on velocity jump / sound speed (0 = off) |
| `regrid_every` | 4 | re-tag every K base steps |
| `subcycle` | false | true = Berger–Colella subcycling (fine at dt/2) |
| `max_patches` | 0 | GPU slot-pool cap (0 = auto ~1/8 of the working set) |

## `[output]` — VTK / checkpoint output

| Key | Default | Meaning |
|---|---|---|
| `frames` | 4 | number of VTK images spaced in time |
| `every` | 0 | one image every K steps (takes precedence over `frames` if > 0) |
| `prefix` | — | file prefix (`<prefix>_0001.vthb`…) |
| `checkpoint` | — | checkpoint prefix |
| `max_steps` | 0 | stop after N steps (0 = unlimited) |

## `[render]` — real-time view (backend `hybrid`)

| Key | Default | Meaning |
|---|---|---|
| `live` | false | open the Metal window |
| `scale` | 4 | pixels per base cell |
| `every` | 2 | refresh every K base steps |
| `grid` | true | draw the AMR patch outlines |
| `rho_min, rho_max` | 0, 0 | color range (0/0 = auto-scale) |

Window controls: **space** = pause, **q** / close = quit.

## `[diagnostics]` — CSV log

| Key | Default | Meaning |
|---|---|---|
| `every` | 0 | one line every K base steps (0 = off) |
| `file` | `<prefix>_log.csv` | CSV path |

Columns: `step, t, dt, res_{mass,momx,momy,energy}, cells, patches,
rho_min/max, p_min/max, mass, kinetic_energy, total_energy, enstrophy,
species_mass, wall_s, mcells_per_s` (details in [`GUIDE.md`](GUIDE.md#4-reading-the-log)).

---

## Minimal example (Sod)

```ini
backend = cpu
t_end = 0.2
[domain]
x0 = 0
x1 = 1
y0 = 0
y1 = 0.25
[grid]
nx = 128          # multiples of amr.block (8)
ny = 32
[state.left]
rho = 1
p = 1
[state.right]
rho = 0.125
p = 0.1
[ic]
default = right
region.1 = halfplane 1 0 0.5 : left
[bc]
left = transmissive
right = transmissive
y = periodic
[output]
frames = 1
prefix = out/sod
```

See `cases/` for complete examples (DMR, bubble, RM, detonation,
deflagration, Blasius…) and `cases/TEMPLATE.ini` (commented template).
