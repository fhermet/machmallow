# machmallow — vision and roadmap

## Vision

machmallow is a **personal compressible-CFD laboratory on Apple Silicon**,
with four inseparable facets:

1. **Understand by building** — every numerical method is implemented from
   scratch, readable end to end (a mini-AMReX);
2. **Prove with references** — nothing lands without a quantitative validation
   gate (exact solutions, theory, conservation);
3. **Exploit and show** — study the physics of compressible instabilities with
   a native Mac experience (zero-copy, real-time);
4. **Usable as an industrial tool** — simple and practical: a case file is
   enough, pre-flight catches errors before the run, the log tracks
   convergence, the output is directly exploitable. Target usage: set up a
   case and get a reliable result without reading the code.

Guiding rule: **one addition = a case file that drives it + a gate that locks
it in + a UX on par with the rest.** Everything goes through the declarative
system and CI.

## Achieved (v1.0 — foundations, complete)

2D float32 Euler/Navier–Stokes solver, MUSCL-Hancock + HLLC with positivity,
multi-level block-structured hybrid CPU/GPU AMR (Berger-Colella subcycling,
refluxing, nesting, periodic), gravity, 100% declarative cases (regions,
moving RH fronts, analytic BCs), complete tooling (pre-flight, preview, CSV
log with residuals, 2-level checkpoint, macOS CI) and validation: Sod/Toro
T2-T5 vs exact Riemann, acoustics (TVD order 4/3), isentropic vortex (2.35),
Sedov (0.490), viscous erf layer, free fall, conservation at the fp32 floor,
CPU/GPU lock-steps everywhere. Details: git log.

Off-milestone addition — **no-slip walls**: declarative `noslip` BC (mirror
with both momentum components reversed, adiabatic wall), validated by the
**Blasius boundary layer** on a flat plate (driver `blasius` +
`cases/blasius.ini`): profile RMS 1.4%, δ99 −2%, Cf +7% vs theory at
Re_x ≈ 2700. Lesson: a clean compressible Blasius requires a pinned free
stream on top (ZPG) and a tall domain (~14 δ99) — a transmissive top lets the
layer's displacement accelerate the flow (Ue +16%) and thin the BL; you also
need a high enough Re_x (δ ≪ x) for the asymptotic regime. A true
characteristic non-reflecting BC remains in the backlog.

## Milestones

### "Industrial tool" thread *(transverse, a little at each milestone)*
- [x] ~~User guide (set up a case in 10 minutes, read the log, exploit the
  output)~~ — done: `docs/GUIDE.md` (step-by-step tutorial, shock/bubble
  example validated by `--check`). Code architecture doc split out into
  `docs/ARCHITECTURE.md` (Mermaid diagrams).
- [~] Bundled post-processing: started — `tools/plot_convergence.py` (order
  curves), `tools/plot_benchmark.py` (GPU gain vs size), and
  `tools/schlieren_video.py`: plotting fields in **numerical schlieren**
  (|∇ρ|) from the `.vthb` files, composited from the finest AMR level without
  ParaView (bilinear cell→point numpy compositing, ~1 s/frame,
  anti-staircase), black/white background palettes, auto tracking camera
  (DMR), annotated pedagogical overlay (LaTeX style) and MP4 export via
  ffmpeg. Frame-per-step output on the solver side via `[output] every = K`.
  Remaining: log plotting. Without depending on ParaView for daily use.
  **V&V figure dossier** (`vv/`): `vv/generate.py` runs the V&V drivers and
  turns their output into committed comparison figures (computed vs
  exact/theory) + one illustrated fiche per case (`vv/cases/`) — now covering
  **every CI case** (17 fiches): order of accuracy, MMS, 0D reactor, WENO5
  suite, multi-species, analytic (Toro/Sedov/RT), conservation, Sod (1D/2D/AMR),
  CJ detonation, Blasius, oblique-shock wedge, immersed boundaries, double Mach
  reflection, Haas–Sturtevant shock–bubble (vs experiment), and the AMR/GPU
  infrastructure.
- [~] Systematically actionable error messages (file:line, suggested fix) —
  started: GPU slot-pool exhaustion (AmrGpuML) now reported by a clear error
  (patch count, KB/patch, pool MB, and the levers: `amr.levels`,
  `tag_threshold`, `amr.max_patches`) instead of an abort by an `assert`
  disabled under `-DNDEBUG`. Configurable pool cap (`amr.max_patches`), by
  default sized to the device memory (~1/8 of the working set). Remaining:
  generalize this style to all gates. Point probes; multi-level checkpoint
  (resume any run).
- [x] ~~**Verification by manufactured solutions (MMS)**~~ — done. The
  convergence study (`convergence`) covered the smooth EULER order (entropy
  wave, isentropic vortex); no gate verified the order of the VISCOUS
  operator. MMS fills that in: a smooth periodic steady manufactured solution
  (sinusoidal ρ, u, v, p), source term S = div(F_Euler) − div(F_viscous)
  computed **in double** by 4th-order finite differences (h=1e-3) on the
  analytic fields ("exact" to ~1e-12 relative to the scheme under test),
  injected into the single-grid step `step2D(mu)`. We let it relax to steady
  state and measure the order of the L1 error (density) vs h over N=16→128.
  - [x] driver `mms.cpp` + manufactured source (4th-order FD, double)
  - [x] measurement via steady-solution error (fp32-robust — the truncation
    measurement `(U−U0)/dt` was fp32-noisy)
  - [x] **viscous operator (mu=0.01), BOTH schemes: MUSCL 2.10, WENO5 1.97**
    (gate >1.8, PASS) — verifies full Navier–Stokes at order 2; the central
    viscous flux (shared by both schemes) caps the order at 2, as expected.
  - [x] **gravity source** (split, MUSCL): order 2.10 (PASS). Since density
    has no gravity source, its order 2 confirms that the gravity coupling
    (momentum/energy → velocity/pressure → density via the fluxes) is
    consistent (a sign bug or a work-term bug would break the steady state and
    the order).
  - [x] steady inviscid (mu=0) *informative*: ~1 for both schemes (MUSCL 1.08,
    WENO5 1.01). Without physical viscosity, the steady-state error is set by
    the **numerical viscosity** of the face fluxes (1st order here) — this is
    NOT the transient design order (MUSCL ~2, WENO5 ~4-5), which is verified by
    `convergence` (advection of an exact solution). Not gated.
  - [x] **reaction source**: no MMS — already verified more rigorously by
    `reactor` (function `react()` vs exact analytic solutions: isothermal err
    8e-8, adiabatic equilibrium) and the reaction↔hydro Strang coupling by
    `detonation` (CJ to 0.8%).
  - [x] driver `mms` added to the CI CPU suite (Euler+viscous ×2 schemes +
    gravity)

### v1.1 — See and measure *(demo + lab)*
The project gains its eyes and its scientific instrumentation.
- [x] ~~Real-time Metal rendering~~ — done: `[render] live = true` —
  programmatic Cocoa window (no Xcode) + fragment shader sampling the
  SIMULATION buffers per pixel (descending the hierarchy from finest to base
  via the block→slot tables, zero copy), viridis colormap, patch outlines,
  pause (space) and clean quit (q/close). ~15% overhead on KH, results
  bit-identical to the headless run. Clean degradation in headless (CI).
  Extension: schlieren mode, chosen field.
- [x] ~~RT growth rate vs linear theory~~ — done: Fourier amplitude of the
  seeded mode at the interface (immune to the harmonics that pollute the
  global energy), LSQ fit over the linear window → **σ = 1.440 vs √(Agk) =
  1.447 (0.5%)**, gate ±15% in `analytic_suite`. Documented finding: KH as a
  vortex sheet is NOT gateable (ill-posed, σ ∝ k — the finest resolved scale
  wins; the y-uniform seed projects poorly onto the localized eigenmode) — a
  clean KH gate would require a tanh profile + Michalke's eigenvalues
  (backlog).
- **v1.1 exit: REACHED** — live demo (KH/RT/bubble) + RT growth gate in CI.

### v1.2 — Mixture physics *(lab)*
From "different densities" to "different gases".
- [x] ~~Two-gas core (uniform CPU grid)~~ — done: φ = ρY conservative (HLLC
  mass flux × upwind Y — Y exactly constant where uniform), Γ = 1/(γ−1)
  advected quasi-conservatively by the contact velocity S* (the EOS reads this
  Γ: consistent E/Γ mixing weights), two-γ HLLC, exact Riemann generalized to
  per-side γ. `species_suite`: interface advection |p−1| 1.0% (startup) /
  0.6% (sustained), two-γ Sod to 1.2e-3 of exact, species mass at the fp32
  floor. PRIMITIVE reconstruction (ρ,u,v,p)+Γ with Γ faces advanced by the
  half-step (the E/Γ desync was half the residual); the remainder comes from
  the E mixing of the HLLC star states — definitive fix = Abgrall-Karni
  double-flux (optional, later).
- [x] ~~Multi-species AMR plumbing (CPU)~~ — done: φ/Γ in AmrML (sibling
  ghosts + θ-blended prolongation, restriction, φ refluxing with
  back-out/fine-apply, species dt); single-gas bit-unchanged. Gate: two-γ Sod
  on 3-level subcycled AMR, L1 3.1e-3 vs exact, species mass 4.4e-5 (relative
  fp32 floor — φ carries ~9× less mass than ρ). Bug caught: physical scalar
  ghosts overwrote all 4 sides (including interior ones) → per-side masked
  version.
- [x] ~~Multi-species GPU~~ — done: two-γ Metal kernels mirroring `step2DY`
  (primitive + half-dt advected face Γ reconstruction, per-side-γ HLLC,
  upwinded φ flux, quasi-conservative Γ transport), scalars (φ, Γ) as a float2
  per cell in the slot pool, full AmrGpuML plumbing (θ-blended ghosts,
  restriction, φ reflux). Gate: two-γ Sod 3 levels GPU in full CPU lock-step —
  L1 2.62e-3 vs exact (= CPU to 1e-7), species mass 2.0e-6, CPU/GPU
  discrepancy 1.6e-4 (fp32), identical patches.
- [x] ~~Two-gas CaseDef + showcase cases~~ — done: `[species]` section
  (gamma1/gamma2), `gas = 2` per state, RH states computed with the upstream
  gas's γ, inflow/analytic closed on the right Γ, new `sinex` region (cosine
  interface). `bubble.ini` becomes the real He bubble (γ 1.667 in air, Haas &
  Sturtevant) and `rm.ini` adds air/SF6 Richtmyer-Meshkov (γ 1.09, single
  periodic mode). The two-gas cases go through AmrML/AmrGpuML at any depth (the
  2-level classes lack the species fields).
- [x] ~~Two-gas diagnostics & output~~ — done: `species_mass` column in the
  CSV log (conserved at the fp32 floor on a closed boundary), `Y` field in the
  .vti/.vthb and pressure closed on the local Γ (the old output closed on
  γ=1.4 everywhere: p wrong by ~20% in helium).
- [x] ~~Quantitative Haas & Sturtevant gate~~ — done: `hs_suite` replays the
  experiment (Mach 1.22 shock in air, He+28% air cylinder, γ 1.645, ρ/ρ_air
  0.182) and measures the interface velocities on the axis (Y=0.5 crossings,
  LSQ slopes over the regime windows measured by the experiment). Results vs
  H&S 1987: V_upstream +6.7%, V_downstream +5.6%, jet −0.7% (gates ±10%;
  Quirk & Karni 1996: +4.7/+0.7/−1.3%). bubble.ini aligned with the
  experiment (Mach 1.22, contaminated He). Lesson: the upstream interface
  ACCELERATES toward the jet — the experiment's "initial velocity" is measured
  just after the shock passes, not over a late window.
- [x] ~~RM case fixed~~ — the mushroom is nonlinear (ka > 1): shock close to
  the interface, t_end 8 (final ka 1.75, roll-up), left BC transmissive (an
  analytic BC re-injects the reflected shock). Bug fixed along the way: the
  base scalar ghosts were transmissive even on periodic axes (now wrapped).
- [ ] Optional Abgrall-Karni double-flux if the 0.6% bother us (v1.2 backlog).
- **v1.2 exit: REACHED** — He bubble (Haas & Sturtevant) and air/SF6
  Richtmyer-Meshkov driven by case file, interface gate (no p oscillation) +
  quantitative characteristic-velocity gate (`hs_suite`) in CI.

### v1.3 — High order *(pedago + lab)*
Lift the TVD limit quantified by the analytic suite (order 4/3 at smooth
extrema).
- [x] ~~NG = 3~~ — done: 3 ghost layers everywhere (WENO5 stencil), all suites
  bit-identical, checkpoint v2.
- [x] ~~WENO5 core + RK3-SSP (uniform grid, CPU)~~ — done: FD-WENO5 Jiang-Shu,
  local Lax-Friedrichs component splitting, conserved face fluxes for the
  upcoming refluxing. `weno_suite`: entropy wave at order 3.02 (= RK3 cap
  exactly, 4.84 on the coarse pair where spatial dominates, spatial-only ~4
  pre-asymptotically), vortex order 3.55 and dissipation 5.5× lower than MUSCL
  at 64², bounded Sod with no over/undershoot (contact +19% vs HLLC — LLF
  diffuses the contact, expected).
- [x] ~~WENO5 in the AMR (CPU)~~ — done: `cfg.weno`, 3 synchronous RK stages
  per level with ghosts refilled at stage times (θ = (k+c)/n), fluxes
  accumulated with weights (1/6, 1/6, 2/3) consumed by the existing refluxing.
  Brutal gate: an all-refined periodic 2-level hierarchy = uniform grid
  **bit-for-bit** (0 diff over 40 steps). 3-level Sod: L1 1.47× MUSCL on the
  same hierarchy (2nd-order prolongation folds the WENO stencils at the c-f
  interfaces — known; the high-order benefit is in the smooth interior),
  conservation 2.6e-6.
- [x] ~~WENO5 Metal kernels + `scheme = weno5`~~ — done: WENO flux kernels
  (integrated RK accumulation, zeroed at stage 0) + RK update (u⁰ capture at
  stage 0), 3 GPU round-trips per level step (CPU ghosts between stages). Gate:
  3-level Sod GPU in full CPU lock-step — L1 identical to 5 figures
  (4.5784e-3), discrepancy 4e-4, identical patches. `scheme = weno5` in the
  .ini (single-gas, inviscid, ML classes at any depth); a 4-level WENO KH runs
  live. Lesson: the RK combination a·u⁰+b·(q+dtL) doesn't telescope as cleanly
  in fp32 as the incremental MUSCL update — mass drift ~1e-5 on a closed
  domain (vs ~1e-8), that's the formulation floor, not a leak (the weighted
  reflux is conservative).
- [x] ~~LLF → HLLC faces~~ — the user was right: Lax-Friedrichs splitting
  dissipates ∝ |u|+c on ALL waves, including the shear (∝ |u|) that HLLC
  resolves near-exactly — the WENO KH was VISIBLY more diffused than MUSCL
  despite the formal order. Fix: WENO5 reconstruction of the primitive face
  states + HLLC. Re-measurements: Sod now beats MUSCL (1.29e-3 vs 1.46e-3),
  vortex 6× less dissipated, KH +14-19% enstrophy retained during roll-up.
  Lessons: (a) the measured 2D order (~2.2) reveals the dim-by-dim face
  quadrature that LLF masked — the value is the CONSTANT; (b) the entropy
  wave's e64 = RK3 temporal floor (identical across variants to 5 figures) —
  each order measurement must know which ceiling it is hitting.
- [x] ~~Viscous WENO5 (Navier–Stokes)~~ — done: the Stokes + Fourier flux
  (2nd-order central differences, factored operator shared with MUSCL) adds to
  the HLLC face fluxes at each RK stage, CPU + Metal + AMR. `scheme = weno5`
  now accepts `mu > 0`. Gates: erf shear layer vs exact solution at order 2.15
  (`weno_suite`), CPU/GPU parity 4.4e-4 on a refined 2-level hierarchy
  (`mlgpu_amr` gate 5); a 3-level viscous KH runs end to end. Note: with a
  smooth u=0/v the viscous operator is identical to MUSCL, so the residual
  discrepancy (~1.5×) is just the RK3-vs-Hancock temporal constant — the proof
  of correctness is the order-2 vs exact.
- [x] ~~Two-gas WENO5 (crosses v1.2 + v1.3)~~ — done: WENO5 reconstruction of
  the face states (ρ,u,v,p) + Y + Γ, per-side-γ HLLC, upwinded species flux,
  quasi-conservative Γ on the contact velocity; method-of-lines so no
  half-step term (face-p and face-Γ synchronous by construction). CPU (core +
  AMR) and GPU (Metal kernels + pool). `scheme = weno5` now accepts
  `[species]`. Gates: uniform two-γ Sod L1 1.49e-3, 3-level AMR L1 4.09e-3
  (between MUSCL 3.1e-3 and single-gas WENO 4.27e-3), GPU in full CPU lock-step
  (L1 = CPU to 4 figures, species mass 4.4e-6). Bug caught: the wave kernel was
  missing in `enableWenoSpecies` (crash of `gpu.maxStableDtAll`), invisible
  because the gate drove `dt` from the CPU → the gate now also exercises the
  GPU reduction.
- **v1.3 exit: REACHED** — vortex at order ≥3 and 6× less dissipated than
  MUSCL, KH/RM interfaces visibly finer at equal resolution; WENO5 + HLLC
  (Euler, Navier–Stokes AND two-gas) from the case file (`scheme = weno5`) all
  the way to the GPU, in CPU lock-step.

### v1.4 — The third dimension *(demo + pedago)*
The big undertaking, run like the multi-level one: reference CPU → validation
gates → GPU.
- [ ] 3D extension of the core (Grid, scheme, AMR, pool, CaseDef, rendering).
- **Exit**: a 3D AMR case ~100M effective cells on the M4, visualized in real
  time.

### v1.5 — Reactive flows (combustion) *(lab + pedago)*
Natural extension of the multi-species work (v1.2): add a reaction source term
+ heat release. Stays strictly 2D (orthogonal to the dimension). Deliberately
simple model — **single-step Arrhenius reaction** + progress variable λ +
heat of reaction q ("reactive Euler"); NO detailed multi-species chemistry
(out of scope, CHEMKIN-style). The source term reuses the earlier gravity one
(split source) and the variable-Γ EOS.
- [x] ~~Stiff reaction integrator (per-cell ODE)~~ — done: `Reaction.hpp`
  (single-step Arrhenius) + `react()` adaptive subcycled RK4. Energy slaved to
  the progress (e = e0 + q·Δλ exact), so conservative by construction and with
  no clamp inconsistency. Driver `reactor` (0D reactor), gates: exact
  isothermal (err 8e-8), adiabatic equilibrium (T = T0+(γ-1)q exact, energy to
  5e-7), stiff (A=1e4, coarse dt → bounded and converged: the subcycling
  handles the stiffness). Remaining: Strang splitting to COUPLE to the flow
  (the stiffness risk itself is lifted).
- [x] ~~Strang splitting + heat/λ coupling (1D)~~ — done:
  R(dt/2)·A(dt)·R(dt/2), A = `step2DY` reused with γ1=γ2 (λ carried by φ=ρλ,
  constant Γ), R = `reactGrid` (constant-volume per-cell reaction: heat into
  E, λ advanced). Driver `detonation`.
- [x] ~~1D Chapman-Jouguet detonation~~ — done: exact D_CJ solved numerically
  (Rankine-Hugoniot with heat + CJ tangency; strong limit → √(2(γ²−1)q)
  verified). Closed tube (reflecting wall, hot ignition): the overdriven
  detonation **relaxes toward CJ** via the Taylor rarefaction — speed measured
  +5.6% → +2.5% → +1.4% over successive windows, **+1.3% established** vs
  D_CJ=4.68. Lessons: (a) a transmissive boundary = infinite reservoir →
  permanent overdrive, you need a reflecting wall; (b) Ea too large / A too
  small → the reaction zone decouples from the shock (detonability failure) —
  you need a fast enough, weakly T-stiff reaction (Ea=8, ~10-cell zone).
- [x] ~~AMR (CPU): reaction in the multi-level hierarchy~~ — done:
  `reactLevel_` brackets each level's hyperbolic step (per-level Strang, a
  local source so no interaction with reflux/restrict), `cfg.react` (implies
  species, γ1=γ2). Gate: CJ detonation on 3-level AMR, refinement (tagged on
  the density jump) follows the moving front, D_CJ preserved to **+0.8%** (vs
  +1.3% uniform).
- [x] ~~GPU: reaction on the GPU (per-cell source kernel)~~ — done: Metal
  kernel `react`/`react_pool` (adaptive subcycled RK4 per thread, energy
  slaved to λ), `Euler2DGpu::encodeReact` + per-level reaction in AmrGpuML
  (Strang around the hyperbolic step). Gate: CJ detonation on 3-level GPU AMR =
  **+0.8% (identical to the CPU to the figure)** — perfect lock-step.
- [x] ~~Declarative `[reaction]` + 2D detonation cell~~ — done: `[reaction]`
  section (A, Ea, q, Tign, gamma) in CaseDef; the λ progress reuses the species
  scalar (`gas = 2` = ignited zone, λ=1). `cases/detonation.ini`: 2D channel
  detonation (closed tube, transverse perturbation, AMR-refined) that runs live
  via `run`. The CJ physics is locked in by the `detonation` driver (0.8% on
  CPU and GPU AMR).
- **Staged validation** (no-slip methodology): 0D reactor ✓ (exact isothermal,
  adiabatic equilibrium, stiffness) → 1D CJ detonation ✓ (D_CJ to 1.3%
  uniform, +0.8% on 3-level AMR) → the 2D detonation cell remains.
- [x] ~~Deflagration (viscous reactive path)~~ — done: `step2DY` carries the
  Stokes+Fourier flux (mu > 0, shared operator) + viscous dt + `species + mu`
  allowed. The flame propagates by CONDUCTION (driver `deflagration` — manual
  study, out of CI since dt ~ dx²/ν over a long propagation; +
  `cases/deflagration.ini`): subsonic (Mach 0.17), 3× faster than with
  numerical diffusion alone (mu=0). Small q so that compression alone cannot
  ignite (T_burnt > Tign by a little) → conduction leads. Lesson: T_burnt =
  1+(γ-1)q must exceed Tign or the flame dies; Zeldovich's √(α) is fragile
  (numerical-diffusion floor at mu=0, quench at high mu on a thin margin) —
  gated as "conduction-driven" rather than strict scaling.
- [x] ~~Viscous branch of the GPU species kernels~~ — done: Stokes+Fourier
  flux in `flux_x_y`/`flux_y_y` (+ pool), `q` wired to the kernels. The
  deflagration runs on GPU (CPU lock-step to 1%); `cases/deflagration.ini`
  goes through the hybrid backend + live.
- **v1.5 exit: REACHED** — Chapman-Jouguet detonation at the theoretical speed
  (uniform +1.3%, AMR CPU/GPU +0.8% in lock-step), 2D detonation cell driven
  by case file and AMR-refined, live. Optional remaining: multi-step chemistry
  (out of scope), quantitative cellular structure.
- Effort: CPU core + CJ gate ~1-2 sessions; full GPU/AMR integration ≈ the
  size of the multi-species milestone.

### v1.6 — Immersed bodies *(lab + demo)*
Geometries in the flow without meshing the solid: a **solid mask** on the
Cartesian grid (ghost-cell / reflective-wall method — no cut cells, hence no
small-cell problem). Fluid↔solid faces get a reflective-wall flux (mirrored
normal velocity, slip). First brick laid:
- [x] **mask-aware `step2D`** — optional `const uint8_t* solid` parameter
  (default `nullptr` → unchanged path, `convergence`/`sod1d`/`mms` regressions
  intact): solid cells ignored (predictor/update/gravity), slopes mirror-
  reconstructed when touching a solid, fluid↔solid face flux = reflective wall
  (`mx→−mx` / `my→−my`). **Inviscid** MVP (no-slip with `mu>0` deferred; the
  viscous fluxes still ignore the mask).
- [x] **`immersed` gate** — a plane shock reflecting off an aligned immersed
  wall, two regimes: Ms=2 (subsonic post-shock) → **14.95 vs 15.0 exact
  (0.33%)**; Ms=3 (**supersonic** post-shock toward the wall, M1≈1.36) →
  **51.68 vs 51.67 (0.02%)**. Aligned face ⇒ exact verification. Added to the
  CI CPU suite.
- [x] **exact wall flux** (`wallPressure`/`wallFluxX/Y` in `Hllc`) — the
  mirror wall + HLLC **leaks in supersonic** (the PVRS estimate keeps
  `SL = uL − cL·q > 0` and upwinds the entire incoming flux: a supersonic body
  became ~transparent). Replaced by the exact wall-pressure flux
  `(0, p*, 0, 0)`, `p*` solved by Newton on `f_W(p*) = u_n` (Toro) — correct in
  sub- AND supersonic. This is what unblocks bow shocks (without it the arc
  formed then drained through the wall).
- [x] **visual demos** — `cases/wc_step.ini` (Woodward & Colella 1984 Mach 3
  step: bow shock, Mach reflection, triple point, slip line, corner
  rarefaction) and `cases/cylinder_bowshock.ini` (Mach 2 cylinder: staircased
  detached arc, shoulder rarefaction, wake). Stagnation ρ consistent (step
  6.27, cylinder 4.36).
- [x] **declarative mask** in the case file — `[solid]` section
  `region.N = rect|circle|halfplane|band|sinex …` (same geometric grammar as
  `[ic]`, without state or motion; `CaseDef::solidAt`). Threaded through the
  MUSCL step of the `Amr2` base grid; guard in `run.cpp` (backend cpu, muscl
  single-gas, refinement disabled → base grid only). `immersed_case` gate:
  `cases/shock_wall.ini` replays the shock reflection through the declarative
  path → **14.95 vs 15.0 exact (0.33%)**, added to the CI CPU suite.
- [x] **AMR integration** (`Amr2`, 2 levels) — per-patch mask
  (`buildPatchSolid_`), mask-aware patch step, **restriction** on fluid
  daughter cells only (solid coarse cells frozen), **refluxing** that never
  corrects a solid cell, **prolongation** as a constant staircase touching a
  solid, and **boundary tagging** of the body (fluid cells touching it →
  staircase refinement). All a no-op without `[solid]` (non-solid gates
  `dmr_amr`/`casedef_test`/`ml_amr` intact). `immersed_amr` gate: refined shock
  reflection (4 patches), wall pressure **14.98 / 15.00 subcycled** vs 15.0
  exact (0.14 / 0.03%) and consistent with the base grid (0.19%); tests
  single-rate AND subcycled. `cylinder_bowshock`/`wc_step` demos refined
  (boundary + shocks).
- [x] **GPU port** (`AmrGpu` hybrid, lock-step) — solid mask in the Metal
  kernels (`predictor`/`flux_x`/`flux_y`/`update` + `_pool` variants: solid
  cells frozen, mirror slopes, **exact wall flux `wallPressure` in MSL**),
  per-slot mask (`smaskP_`) + coarse mask (`Euler2DGpu`), and mask-aware CPU
  AMR chain (restriction, reflux, prolongation, tagging) identical to `Amr2`.
  `AmrGpuML` gets a zero mask (shared inviscid kernels). `immersed_gpu` gate:
  CPU↔GPU lock-step on a Mach 2 cylinder (single **5.9e-4** + subcycled
  **1.1e-3** vs gate 1e-2, identical patches). Demos in `backend=hybrid` (~5×:
  cylinder 56→10 s, step 116→23 s). Non-solids intact (`dmr_gpu` 9.8e-6,
  `mlgpu_amr`).
- [x] **quantitative validation — θ-β-M oblique shock**: immersed wedge (15°
  ramp, Mach 2.5) → measured shock angle **β = 38.3° vs 36.9° exact** (θ-β-M
  relation, staircase bias of 1.4° that decreases with resolution:
  6.1°→1.7°→1.4°). `immersed_wedge` gate (CI CPU) + `cases/wedge.ini` demo
  (reflection on the top wall, inlet-like). The cylinder's detached arc vs
  Billig's correlation remains (qualitative for now).
- [x] **solid multi-level** (`AmrML`/`AmrGpuML`, depth > 2).
- [x] **forces (drag / lift by ∫ wall pressure)** — `wallForce`
  (`solver/Forces.hpp`) integrates p·n over the fluid/solid faces. Validated in
  `immersed_wedge`: the wedge's wall C_p **2.447 vs 2.468 exact (0.8%, oblique
  shock)** and **lift of a symmetric cylinder = 0** (|F_y/F_x| < 1e-3). Viscous
  friction not included (pressure drag).
- [x] **viscous no-slip** (mask-aware viscous flux) — a solid neighbor becomes
  a no-slip ghost (both velocities reversed, ρ/p preserved → adiabatic); the
  wall shear makes the wall adherent (the convective flux provides the
  pressure). `immersed_noslip` gate: Blasius boundary layer on an **immersed
  plate** → profile **RMS 0.7%** vs f', Cf **3%** vs 0.664/√Re_x. **Ported to
  GPU** (mask-aware viscous flux in the Metal kernels): viscous CPU↔GPU
  lock-step verified (`immersed_gpu`, discrepancy 4e-4) — `mu > 0` + solid runs
  in `hybrid`.
- [x] **solid multi-level** (depth > 2) — mask threaded through `AmrML` (CPU,
  arbitrary depth): per-patch + base, restriction (fluid daughters), refluxing
  (solid cells spared), prolongation (constant staircase), boundary tagging —
  at every level, via a geometric query `solidAt(position)` (a level's parent
  can be a patch, not the base). `immersed_amr` gate extended: shock reflection
  at **3 levels** → wall 15.03 vs 15.0 exact (0.21%, 16 patches). No-op without
  `[solid]` (ml_amr / species / weno intact). **Ported to GPU** (`AmrGpuML`,
  per-slot pool mask + base mask, geometric queries for the plumbing): CPU↔GPU
  lock-step at **3 levels** verified (`immersed_gpu`, discrepancy 7.5e-4).
  Solids now run in `hybrid` at arbitrary depth.
- [~] cut-cells (remove the staircase) — the only remaining item of the facet.
  **Phase 1 done** (`feature/cutcell-geometry`, gate `cutcell_geom`): the EB
  moment engine `src/geometry/CutCell.hpp` — exact volume fractions, face
  apertures and the divergence-closed embedded-boundary face (area + normal)
  for analytic circle/half-plane (cylinder/wedge). Fluid area exact at every
  resolution; EB perimeter converges O(dx²) to the exact interface.
  **Phase 2 done** (`feature/cutcell-euler`, gate `cutcell_euler`): inviscid
  1st-order cut-cell FV (`src/solver/CutCell2D.hpp`) — aperture-weighted HLLC
  faces + exact slip-wall EB flux, hybrid **flux redistribution** so the step
  is set by the full cells. Validated: uniform-at-rest preserved (well-
  balanced), mass/energy conserved to the fp32 floor with full-cell dt, and a
  Mach-2 bow shock whose stagnation pressure matches the Rayleigh pitot value
  to 0.4 %. Known limit: tangent slivers with a starved neighbourhood fall back
  to a positivity floor — a proper fix is state redistribution / an extended
  merging neighbourhood (a later phase).
  **Phase 3 done** (`feature/cutcell-o2`, gate `cutcell_o2`): 2nd order —
  least-squares primitive gradients (Barth-Jespersen limited for shocks),
  reconstructed to the face centres and the EB centroid, advanced with SSP-RK2.
  Validated: least-squares gradients reproduce a linear field exactly on the
  irregular cut-cell stencils, and a smooth entropy blob sliding along a 45°
  wall converges at **order 2.1** (design order, unlimited). Conservation and
  the Mach-2 bow shock still pass with the 2nd-order path.
  **Phase 4 done** (`feature/cutcell-viscous`, gate `cutcell_viscous`): viscous
  Navier–Stokes on cut cells — aperture-weighted Stokes/Fourier face fluxes +
  a **no-slip embedded-boundary traction** (tangential shear over the normal
  distance to the wall, adiabatic). Validated on planar **Couette** with the
  lower wall carried by the EB: the steady profile matches the exact linear
  solution to **<1 %** (0.28 % at 96², converging). The wall-shear model is
  ~1st order (order ~1.4); a fluid-centroid / full-stress EB model would lift
  it.
  **Phase 5 done** (`feature/cutcell-amr`, gate `cutcell_amr`, standalone
  2-level demonstrator): (5a) with exact moments a coarse cut cell's fluid
  volume equals the sum of its four fine children's (7.7e-12) → volume-weighted
  restriction is conservative (integral preserved to 5e-16); (5b) a 2-level
  advance (base + a patch containing the body) with flux-register **refluxing**
  keeps the composite mass at the float32 floor — **9e-8 with reflux vs 1.1e-3
  without (12000× better)**.
  **Phase 5c (partial)** (`feature/cutcell-run`): cut cells are now reachable
  from the declarative case system — `solid_method = cutcell` in an INI runs
  the 2nd-order (viscous-capable) cut-cell solver on the base grid through
  `./build/run` (`cases/cylinder_cutcell.ini`: exact-boundary Mach-2 bow shock,
  one circle/half-plane [solid], cpu single-level).
  **Phase 5d (increment 1)** (`feature/cutcell-amr-prod`, gate
  `cutcell_amr_prod`): cut cells wired into the **production `Amr2`** (2 levels,
  single-rate, CPU) — per-patch analytic geometry built at `makePatch_`,
  aperture-weighted flux-recording advance (`cutCellStepFluxed`) on the base
  grid and every patch, **cut-aware reflux** (`dt/(κV)`·(Σ fine − coarse) with
  extensive fluxes, reducing to the uniform `(dt/dx)F` when κ=1) and
  **κ-weighted restriction**, all behind `cfg.cutCell` + `Amr2::momentFn` (the
  staircase path is untouched). Gate: composite mass drift **6.7e-8 with reflux
  vs 1.2e-4 without (1800× better)** for a body contained in one refined patch.
  **Phase 5e (increment 2)** (`feature/cutcell-amr-frd`, gate
  `cutcell_amr_prod`): **cross-patch flux redistribution**. `cutCellStepFluxed`
  is split into `cutCellDc` (conservative divergence + recorded fluxes) and
  `cutCellHybridD` (hybrid divergence + redistribution); the fine patches now
  advance as a **composite** — Dc ghosts are filled from same-level siblings,
  and the redistribution that lands in a patch's ghost cells is **scattered**
  into the sibling that owns them (reads ghosts, writes interiors, so it is
  order-independent). A body **spanning a 4×4 block of patches** (cut cells on
  the internal sibling seams) now conserves to **9.1e-8** (was ~1.3e-5 without
  the scatter).
  **Phase 5f (increment 3)** (`feature/cutcell-amr-regrid`, gate
  `cutcell_amr_prod` gate 3): **regrid-driven EB-band refinement**. `regrid()`
  now tags the embedded-boundary band (coarse cut cells 0<κ<1; the existing ±2
  dilation extends it into the surrounding fluid without over-refining the solid
  interior), so refinement follows the body automatically instead of a pinned
  patch. Gate: the body block is auto-refined and the EB-driven set conserves to
  **9.2e-8**. (Flow-tag-driven *dynamic* regrid rides the general `Amr2` float32
  conservation floor, ~few×1e-6 — a tiny/absent body drifts the same way — so it
  is not cut-cell specific.)
  **Phase 5g (increment 4)** (`feature/cutcell-amr-subcycle`, gate
  `cutcell_amr_prod` gate 4): **Berger-Colella subcycling**. The base grid takes
  one dtC, each patch two dtF = dtC/2 substeps with time-interpolated coarse
  ghosts (the θ = ½ blend of tⁿ/tⁿ⁺¹). The reflux is split — `cutRefluxBackout_`
  removes the single coarse flux (×dtC), `cutRefluxFineApply_` applies each
  substep's fine flux (×dtF, twice = dtC) — and the composite cross-patch FRD
  runs per substep. Gate: composite mass drift **2.1e-8** subcycled.
  **Phase 5h (increment 5)** (`feature/cutcell-amr-ml`, gate `cutcell_amr_ml`):
  cut cells ported to **`AmrML` — arbitrary depth + recursive subcycling**. Per-
  patch geometry at every level, composite cross-patch FRD per level (Dc ghosts
  from siblings + scatter), cut-aware `refluxBackOut_`/`refluxFineApply_` (the
  reflux into a cut cell passes the increment through the same hybrid
  redistribution as the advance), κ-restriction and EB-band tagging — all behind
  `cfg.cutCell` + `AmrML::momentFn` (staircase/WENO/species paths untouched).
  Gates use a body **large enough to straddle the coarse-fine seams at every
  level** (the demanding cross-level case): composite mass drift **9.7e-8
  (2-level), 6.0e-9 (2-level subcycled), 1.2e-7 (3-level), 4.1e-9 (3-level
  subcycled)** — the fp32 floor, i.e. cut cells are conservative *across*
  coarse-fine boundaries, not only when nested. Resolution note: on a very
  coarse base (few cells across the body) with 3 levels a degenerate patch
  topology can raise the drift to ~2e-5 — an under-resolution artifact
  (base 48 → 1.2e-7 vs base 32 → 2.2e-5), not a scheme defect.
  **Phase 5i (increment 6, GPU)** (`feature/cutcell-gpu`, gate `cutcell_gpu`):
  the 1st-order cut-cell operator ported to **Metal** (`shaders/cutcell.metal` +
  `CutCell2DGpu`) — aperture-weighted HLLC face fluxes, slip-wall EB flux,
  conservative divergence, hybrid divergence + **flux redistribution in GATHER
  form** (each cell collects what its neighbours shed, no atomics — the GPU
  idiom for the CPU scatter), positivity floor. Physics copied verbatim from
  `euler2d.metal` so the GPU matches the CPU oracle. Gate: GPU vs CPU
  `cutCellStepFluxed` in **lock-step over 100 steps → worst relative ρ diff
  2.98e-6** (fp32 reassociation of the gather-vs-scatter FRD sum). Also the
  **`_pool` kernels** (all patches batched in one dispatch, `gid.z → slot`, the
  layout AmrGpu uses) with a pooled `CutCell2DGpu` path: a pooled patch matches
  the plain single-grid run **bit-exact (0.0e+00)**, no cross-slot bleed. Plus a
  **phase-split** step (`dcPhase`/`hybridPhase`/`updatePhase`, single grid and
  pool) so a composite caller can interleave the CPU cross-patch passes (Dc
  ghost fill / redistribution scatter) between GPU phases — validated bit-exact
  vs the monolithic step.
  **Phase 5j (increment 7, GPU)** (`feature/cutcell-gpu-amr`, gate
  `cutcell_gpu_amr`): the hybrid **`AmrGpuCut`** class — two-level cut-cell AMR
  with the coarse level and the patch pool both living in shared Metal buffers
  (a `CutCell2DGpu` each). Per step the GPU advances the coarse grid (monolithic
  cut step) and the patches as a **composite** via the phase-split pool
  (`dcPhasePool` → CPU fills each patch's Dc ghosts from same-level siblings →
  `hybridPhasePool` → `updatePhasePool`); the CPU orchestrates cut-aware reflux
  (`cutRefluxBackout_`/`cutRefluxFineApply_`), κ-restriction, EB-band tagging and
  slot management in place (unified memory, zero copies). Key simplification: the
  gather-form FRD makes the CPU **D-scatter unnecessary** — exchanging the Dc
  ghosts alone reproduces the cross-patch coupling (a patch's interior D equals
  the monolithic single-grid divergence over the tiled region), mathematically
  identical to the Amr2 scatter path. Gate (body spanning a 4×4 patch block,
  reflective box): GPU vs CPU (`Amr2` cut) **lock-step over 200 steps → worst
  relative ρ diff 4.4e-6 single-rate, 3.6e-6 subcycled**, and the GPU composite
  mass conserves to **1.1e-7** (fp32 floor).
  **Phase 5k (increment 8, GPU)** (`feature/cutcell-gpu-amr`, gate `cutcell_gpu`
  gate 4): the **2nd-order** single-grid cut-cell operator on Metal —
  least-squares primitive gradients (Barth-Jespersen limited, `cc_grad`),
  reconstructed to the face centres (`cc_flux_x/y_o2`, constant at grid
  boundaries) and to the **EB centroid** (`cc_dc_o2`, offset stashed in the
  `CCMom` pads by `setGeometry`), advanced with **SSP-RK2** (`cc_rk2` + a saved
  tⁿ buffer); `CutCell2DGpu::enableO2`/`divO2`/`rk2Stage1`/`rk2Stage2`. Gate: GPU
  vs the CPU `stepCutCell` oracle (which already showed design order 2.1) in
  lock-step over 100 steps → worst relative ρ diff **1.0e-3** (the LSQ + limiter
  in fp32 vs the CPU double path).
  **Phase 5l (increment 9, GPU)** (`feature/cutcell-gpu-amr`, gate
  `cutcell_gpu_ml`): the hybrid **`AmrGpuMLCut`** class — **arbitrary-depth**
  cut-cell AMR on GPU (the GPU analogue of `AmrML` cut, as `AmrGpuCut` is to
  `Amr2`). The base and every level live in shared Metal buffers (a
  `CutCell2DGpu` for the base, one pooled `CutCell2DGpu` per level); the CPU
  drives the recursive Berger-Colella subcycling (`advanceTree_`), the
  per-level composite (`dcPhasePool` → CPU D^c-ghost exchange from siblings →
  `hybridPhasePool` → `updatePhasePool`), cut-aware reflux (the hybrid-
  redistribution `cutRefluxCorr_`, `refluxBackOut_`/`refluxFineApply_`),
  κ-restriction, EB-band tagging + nesting-correct regrid and per-level slot
  pools — all ported from `AmrML`, dropping WENO/species/react/solid (cut is
  inviscid single-gas). As in `AmrGpuCut`, the gather-form FRD makes the CPU
  D-scatter unnecessary. Gates (body straddling the coarse-fine seams at every
  level, reflective box): GPU vs CPU (`AmrML` cut) lock-step over 120 steps →
  worst relative ρ diff **3.6e-6 / 3.2e-6 / 3.4e-6 / 4.3e-6** (2-level, 2-level
  subcycled, 3-level, 3-level subcycled), GPU composite mass conserved to
  **1.0e-7 / 1.3e-8 / 1.2e-7 / 5.7e-9** — the fp32 floor across the hierarchy.
  This completes the cut × AMR × GPU matrix at 1st order. Remaining: 2nd order
  in the *AMR* composite — needs a 2nd-order CPU AMR-cut oracle first (the
  current `Amr2`/`AmrML` cut paths are 1st order); a further facet.

## Backlog (pulled into a milestone when it serves, never in the abstract)

Steady-state mode (local time stepping — would give the log residuals their
full meaning **and unblock the internal nozzle regimes**; cf. nozzle lesson);
**characteristic non-reflecting outflow/inflow (NSCBC)** — demanded by both
Blasius (ZPG top) AND the nozzle/jet (cf. lessons); Richardson tagging;
multi-level checkpoint; perf (GPU pipelining, GPU ghost fill, multi-level
syncs); point probes; ellipse/polygon regions; 4-quadrant 2D Riemann;
refinement ratio 4; additional sources (Coriolis, heating); Metal System Trace
(requires full Xcode).

## Non-goals (to stay focused)

- No MPI / multi-machine — one Mac is the project's scope.
- No turbulence models (RANS/LES) and no implicit solver.
- Not a production code at *scale* (unstructured meshes, all-round generality)
  — but an **industrial-grade UX** *is* an explicit goal (facet 4): ease of use
  over generality, code readability over extreme optimization.

## Lessons learned (design notes)

- **Physical BCs of edge patches: always at the fine level**
  (`fillPatchPhysical`). Prolongating the coarse ghosts at the domain edges
  breaks consistency as soon as a wave touches the boundary — a bug found in
  phase 7, two orders of magnitude on conservation.
- Float32 conservation gates are calibrated on the **measured rounding floor**
  (~1e-8/step per active patch), not on an ideal value; the discriminating
  test is the contrast with/without refluxing on a frozen mesh crossed by the
  waves.
- Apple Silicon benchmarks: ±30% variance on small cases (GPU frequency
  governor) — always best-of-N, and large cases are more reliable.
- Convergence order is measured **in a smooth regime AND on the right grid
  regime**: TVD limiters cap at 4/3 at smooth extrema (sine), the O(A²)
  nonlinear steepening caps the return-to-IC tests at large N, and the fp32
  floor caps everything below ~1e-6 — each gate must know which of the three
  it is measuring.
- Reflecting walls under gravity: extrapolate the pressure hydrostatically
  into the ghosts (the mirror reverses the gradient and pumps energy up to
  blow-up).
- Tagging: stencils must read the ghosts (never clamp at patch seams); the
  periodic wrap applies to ALL levels including the parent coordinates of the
  prolongation; the regrid cadence must be per-level (scale-invariant buffer).
- **The rendering shader reads the same memory as the solver**: the NG 2→3
  transition (v1.3) was verified bit-identical on the DATA path (headless
  output), but the live fragment shader hardcoded the ghost offset (`+2`, width
  `nx0+4`) → all displayed cases garbled (whatever the scheme) while the gates
  passed. The ghost offset is now passed as a uniform (`ng`). Lesson: a
  "bit-identical" check must cover ALL consumers of the memory, not just the
  solver — the zero-copy rendering is one.
- **C-D nozzle: regimes & steady convergence** (declarative `backpressure` BC
  added: staircase schedule `t0 p0 t1 p1 …`). The backpressure sweep and the
  per-regime cases revealed two STRUCTURAL limits (not bugs), each pointing to
  a backlog item: (a) convergence to a subsonic STEADY state is very slow in
  explicit (the throat M creeps up over tens of time units, and the true first
  critical — choking — is lower than the 1D prediction because of boundary-
  layer / immersed-boundary blockage) → **steady-state mode (local time
  stepping / pseudo-time)**; (b) in an open domain, the jet exhausting into a
  quiescent ambient creates a mixing layer + outflow reflections
  (`backpressure` transmissive in supersonic) that prevent a clean steady state
  and stiffen the time step → **characteristic non-reflecting outflow**. As a
  bonus: the transonic normal shock is intrinsically unsteady (λ shock train +
  separation), and the over/under-expanded regimes have a turbulent jet (KH)
  that never freezes — the useful "established" state there is the near-exit
  SHOCK structure, not a frozen field. Key diagnostic: plot the **Mach field +
  M=1 sonic line**, not the schlieren (which over-reacts to AMR noise and exit
  mixing — it made me wrongly conclude there was a shock).
- **Stiff inviscid KH = ill-posed** (σ ∝ k, no cutoff): the fine roll-up
  structure is amplified truncation noise, hence SCHEME-DEPENDENT. MUSCL and
  WENO5-HLLC diverge completely in the small billows (different patterns, not
  just +14-19% enstrophy on the primary roll), and refining gives MORE
  roll-ups, never convergence. To compare two schemes you need a physical
  cutoff — `mu > 0` or a resolved-thickness tanh shear layer (cf. Lecoanet
  2016) — then both converge to the same field, WENO5 at lower resolution. The
  flux splitting (LLF) makes it worse: it dissipates ∝ |u|+c on ALL waves, so
  it smooths contacts and shear that HLLC keeps sharp.
