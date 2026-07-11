# Sod shock tube — *validation vs exact Riemann*

**Objective.** Reproduce the exact solution of the classic Riemann problem:
density, velocity and pressure at t = 0.2 must overlay the exact solution,
capturing the rarefaction, the contact discontinuity and the shock without
spurious oscillations.

## Numerical setup
> **MUSCL-Hancock and WENO5**, both with HLLC, on a uniform grid (**no AMR**),
> inviscid Euler, CFL 0.4, t = 0.2. Profile shown at N = 400 (from the
> `convergence` driver — same solver for both schemes). Reference = exact
> Riemann solution. float32.

## Results
![Sod shock tube — MUSCL & WENO5 vs exact](../figures/sod.png)

Grid-convergence order (L1 density vs exact): **≈0.90 for both schemes**. The
`sod1d` driver's independent 1D study gives MUSCL 0.90.

## Discussion
Both schemes match the exact solution. Discontinuities cap the convergence at
first order for both — the high-order advantage of WENO5 appears on *smooth*
flow (see the order-of-accuracy fiche), while here WENO5 + HLLC only resolves
the **contact** slightly more sharply.

---
*Part of the [V&V dossier](../README.md). Regenerate: `python3 vv/generate.py`. Source data: [`../data/`](../data/).*
