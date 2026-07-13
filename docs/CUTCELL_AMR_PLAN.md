# Cut-cells in the production AMR — integration plan

The standalone cut-cell solver (`src/solver/CutCell2D.hpp`) and the 2-level
demonstrator (`cutcell_amr`) prove the physics: exact geometry, flux
redistribution, 2nd order, no-slip, and conservative coarse/fine coupling. This
note maps how to fold that into the production `Amr2` / `AmrML` so cut cells run
with real AMR from a case file. It is a **large, deliberate refactor** — the
reflux is *not* a drop-in swap.

## Where things live today
- Per-patch state: `Patch{ Grid grid; std::vector<uint8_t> solid; Fx, Fy; ... }`
  (`Amr2.hpp:54`, `AmrML.hpp:41`).
- Advance: `stepLevel_` / `stepSingleRate_` call `step2D(grid, dt, scratch, mu,
  gx, gy, solid)` per grid; fluxes land in `scratch.Fx/Fy` and are copied to
  `p.Fx/p.Fy` (`AmrML.hpp:725`, `Amr2.hpp:329`).
- Reflux: `refluxCoarse_/refluxFine_` apply `neighbor ± (dt/dx)·F` — **assumes
  uniform cells** (`Amr2.hpp:488,523`; `AmrML.hpp:1010,1059`).
- Restriction: `0.25·Σ children` (fluid-only average when a staircase mask is
  present) (`Amr2.hpp:567`, `AmrML.hpp:1110`).
- Prolongation: limited-linear, slope→0 next to solid (`Amr2.hpp:399`).
- Solid mask: `solidAt(x,y)` callback → per-grid `uint8` mask, rebuilt on
  regrid; threaded through step/restrict/reflux/prolong/tag.
- Note (`Amr2.hpp:75`): *refinement near solids is not yet supported* — with
  solids, only the base grid currently runs.

## The integration, step by step
1. **Per-grid cut geometry.** Add `cutcell::Geometry` to `Patch` and the base
   level; build it (from the analytic `[solid]`) in `makePatch_` and on regrid,
   alongside the existing `solid` mask. Exact moments make it consistent across
   levels (see `cutcell_amr` gate 1).
2. **Cut-cell advance.** Replace the per-grid `step2D` with a cut-cell step that
   fills `scratch.Fx/Fy` with the **aperture-weighted** face fluxes (so the
   flux register keeps working) and applies FRD + the positivity floor. Reuse
   `cutCellStepFluxed`, adapted to write into the AMR scratch layout.
3. **Cut-aware reflux (the hard part).** Generalise `refluxCoarse_/refluxFine_`
   to `correction = (dt/(κ_c V)) · (Σ aperture·F_fine − aperture·F_coarse)`.
   The standalone `cutcell_amr` gate 3 is the reference implementation. Watch
   the FRD/reflux interaction at the coarse-fine seam.
4. **Volume-weighted restriction.** `q_c = (Σ κ_f V_f q_f)/(Σ κ_f V_f)` — the
   staircase path already does a fluid-only average; swap the weights to κ.
5. **Prolongation.** Keep limited-linear first (works); optionally make it
   volume-fraction aware later for strict conservation near the body.
6. **Tagging.** Refine the fluid band around the EB (cells with 0<κ<1 and ±1
   layer) rather than every cell touching a solid — avoids over-refining a
   solid interior.

## Hardest points
- **Reflux must become cut-aware** (κ + apertures) — not the current
  `(dt/dx)·F`. This is the crux.
- **Geometry consistency across regrid**: only from the exact analytic moments;
  never numerical mask sampling, or coarse/fine volumes stop summing.
- **FRD × reflux at the seam**: redistribution near a coarse-fine boundary and
  the flux register must not double-correct.
- **Cut-aware prolongation/positivity** near thin slivers.

## Suggested first increment
`Amr2` (exactly 2 levels), **single-rate**, CPU, one circle/half-plane solid:
steps 1–4 above, validated by (a) composite conservation and (b) the Mach-2
cylinder matching the single-level cut-cell result. Then subcycling, then
`AmrML` (arbitrary depth), then GPU. Expect several debug iterations per step.
