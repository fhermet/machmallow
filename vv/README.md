# Verification & Validation — figures

Reproducible evidence that machmallow solves the equations right
(**verification**) and the right equations (**validation**). Every figure
below is generated from a driver's output by `vv/generate.py` and committed,
so it renders here without running anything.

> These are the highlights with pictures. The **full gate list** (20+
> quantitative PASS/FAIL gates, replayed in CI) is in
> [`docs/VALIDATION.md`](../docs/VALIDATION.md).

Regenerate everything:

```sh
cmake --build build -j
python3 vv/generate.py
```

Numbers below are from an Apple M4 (float32); they may vary by ~1 ULP across
machines. All studies **PASS** their CI gates.

---

## 1. Verification — order of accuracy

Smooth-regime convergence: the $L_1$ error must fall at the scheme's design
rate as the grid is refined (measured before the float32 roundoff floor
flattens the curve).

> **Numerical setup** — MUSCL-Hancock + HLLC **and** WENO5 + SSP-RK3, on
> **uniform grids** (no AMR), inviscid Euler, CFL 0.4. Problems: entropy wave
> (periodic, t=1), 2D isentropic vortex (10×10 periodic, t=2), Sod (t=0.2).
> Exact reference = the advected initial condition (smooth cases) / the exact
> Riemann solution (Sod). float32.

![Order of accuracy](figures/order_of_accuracy.png)

| Problem | Scheme | Observed order |
|---|---|---|
| entropy wave | MUSCL | 1.88 |
| entropy wave | WENO5 | 4.03 |
| isentropic vortex | MUSCL | 2.00 |
| isentropic vortex | WENO5 | 2.21 |

MUSCL converges at ~2; WENO5's formal order 5 is capped by the RK3 time
integration and the midpoint face flux, but it carries a **much smaller error
constant** (the vortex is ~6× less dissipated than MUSCL at equal
resolution). The viscous Navier–Stokes operator is separately verified at
order 2 by manufactured solutions (`mms`; see `docs/VALIDATION.md`).

---

## 2. Validation — Sod shock tube vs exact Riemann

The classic Riemann problem: density, velocity and pressure at t = 0.2
overlaid on the exact solution, for **both schemes**. Each captures the
rarefaction, the contact discontinuity and the shock without spurious
oscillations; WENO5 + HLLC resolves the contact slightly more sharply.

> **Numerical setup** — **MUSCL-Hancock and WENO5**, both with HLLC, on a
> uniform grid (**no AMR**), inviscid Euler, CFL 0.4, t = 0.2. Profile shown
> at N = 400 (from the `convergence` driver, same solver for both
> schemes). Reference = exact Riemann solution. float32.

![Sod shock tube — MUSCL & WENO5 vs exact](figures/sod.png)

Grid-convergence order (L1 density vs the exact solution): **≈0.90 for both
schemes** — as expected, discontinuities cap both at first order (the
high-order advantage of WENO5 shows on *smooth* flow; see §1). The `sod1d`
driver's independent 1D grid-convergence study gives MUSCL 0.90.

---

## 3. Validation — Blasius boundary layer vs similarity

A low-Mach viscous flow over a flat plate. At the measurement station
(Re_x = 2732) the steady velocity profile must collapse onto the Blasius
similarity solution $u/U_e = f'(\eta)$.

> **Numerical setup** — **MUSCL-Hancock and WENO5**, both with HLLC, on a
> **single uniform grid 320×256** (dx = dy ≈ 3.9e-3, **no AMR**), GPU
> (`hybrid` backend), Navier–Stokes μ = 8e-5, CFL 0.4, free stream U = 0.3
> (**M ≈ 0.25**). BCs: inflow (left), zero-gradient (right), **pinned free
> stream** on top (zero pressure gradient), and an **aligned bottom wall** —
> slip ahead of the leading edge (x < 0.15), **no-slip** on the plate. Marched
> to steady state. float32.

![Blasius profile vs similarity](figures/blasius.png)

MUSCL and WENO5 land essentially on top of each other here — expected for a
**smooth steady** boundary layer (the viscous flux operator is shared and
there are no discontinuities for WENO5 to sharpen). That agreement is itself
a useful cross-scheme consistency check; the gated metrics below are from the
MUSCL run.

The skin friction measured at several stations along the plate, against the
Blasius law $C_f = 0.664/\sqrt{Re_x}$:

![Skin friction vs Blasius](figures/blasius_cf.png)

| Quantity (station Re_x = 2732) | Result vs theory |
|---|---|
| profile RMS $(u/U_e - f')$ | 1.3618e-02 (gate 3e-2) |
| boundary-layer thickness $\delta_{99}$ | -2.0% |
| skin friction $C_f$ vs $0.664/\sqrt{Re_x}$ | 7.0% |

**Where the Cf bias comes from — a grid-convergence + Mach study.** The Cf is
biased high everywhere (~+5 % mid-plate, rising to +7 % at the leading edge
and +12 % near the outflow). A refinement study (reproduce with
`bash vv/blasius_study.sh`) pins down each cause:

![Cf bias vs wall-normal resolution](figures/blasius_refine.png)

- **Dominant cause — near-wall resolution.** Refining the wall-normal grid
  drives the mid-plate bias down monotonically (+8.1 % at ny=128 → +4.0 % at
  ny=1024). The estimator itself is *exact* for a Blasius profile (the
  near-wall velocity is linear to $O(\eta^4)$), so this is the finite-grid
  representation of the wall shear, not the wall-gradient formula.
  **Refining in y *alone* recovers the accuracy at ~half the cells** of
  isotropic refinement (anisotropic ny=512/nx=320 = 164k cells beats isotropic
  328k) — the error is purely wall-normal.
- **Compressibility — ruled out.** At fixed Re_x and resolution, lowering the
  Mach number from 0.25 to 0.06 (raising the stagnation pressure) leaves the
  bias unchanged (+5.4 → +5.5 %). The incompressible reference is fine.
- **Residual ~+4 % (grid-independent plateau)** — consistent with the weak
  **favorable pressure gradient** ($U_e$ drifts +2 % along the plate under the
  pinned top), which raises the wall shear above the ideal ZPG Blasius law; it
  is Mach- and grid-independent. (A virtual-origin shift is ruled out — the
  mid-plate ratio is already flat at the geometric origin.)
- **Local rises** — the leading edge (+7 %) has the thinnest BL (fewest cells)
  plus the slip→no-slip / LE singularity; the outflow (+12 %) is a
  transmissive-boundary artifact. δ99 (−2 %) is the discrete 0.99-crossing.

**Design note.** Industrial codes avoid this bias with wall-normal
*stretching* (body-fitted prism layers, y⁺ < 1) — not possible on our
uniform-Cartesian + ratio-2 block-AMR foundation. The on-design equivalent is
the **anisotropic uniform grid** above (fine y, coarse x). All metrics pass
their gates.

---

*Generated by [`vv/generate.py`](generate.py). Source data in
[`vv/data/`](data/). Full V&V gate list: [`docs/VALIDATION.md`](../docs/VALIDATION.md).*
