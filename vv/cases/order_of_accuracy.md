# Order of accuracy — *verification*

**Objective.** Confirm each scheme converges at its **design rate** in a
smooth regime: the $L_1$ error must fall like $h^p$ as the grid is refined
(measured before the float32 roundoff floor flattens the curve).

## Numerical setup
> MUSCL-Hancock + HLLC **and** WENO5 + SSP-RK3, on **uniform grids** (no AMR),
> inviscid Euler, CFL 0.4. Problems: entropy wave (periodic, t=1), 2D
> isentropic vortex (10×10 periodic, t=2), Sod (t=0.2). Reference = the
> advected initial condition (smooth) / the exact Riemann solution (Sod).
> float32. Driver: `convergence`.

## Results
![Order of accuracy](../figures/order_of_accuracy.png)

| Problem | Scheme | Observed order |
|---|---|---|
| entropy wave | MUSCL | 1.88 |
| entropy wave | WENO5 | 4.03 |
| isentropic vortex | MUSCL | 2.00 |
| isentropic vortex | WENO5 | 2.21 |

## Discussion
MUSCL converges at ~2. WENO5's formal order 5 is capped here by the RK3 time
integration and the midpoint (1-point) face flux, but it carries a **much
smaller error constant** — the isentropic vortex is ~6× less dissipated than
MUSCL at equal resolution. The viscous Navier–Stokes operator is verified
separately at order 2 by manufactured solutions (`mms`).

---
*Part of the [V&V dossier](../README.md). Regenerate: `python3 vv/generate.py`. Source data: [`../data/`](../data/).*
