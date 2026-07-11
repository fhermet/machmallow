# Multi-species two-gas — *validation vs exact*

**Objective.** Validate the two-gas core (per-cell γ transported with the
species mass fraction): (1) the **Abgrall material-interface** test — a ρ+γ
jump advected in a uniform p, u field must keep **pressure and velocity flat**
(the discriminating test that naive conservative multi-gas schemes fail with
spurious pressure oscillations at the interface); (2) a **two-gas Sod**
(γ 1.4 | 1.6) vs the generalized per-side exact Riemann solution, uniform and
on **3-level AMR**; (3) species-mass conservation at the float32 floor.

## Numerical setup
> MUSCL-Hancock + per-side HLLC with a **quasi-conservative γ transport**
> (`Muscl2DSpecies`), CFL 0.4. Interface: periodic, u = 0.5, one full period
> (N = 200). Two-gas Sod: transmissive, t = 0.2, N = 400 (uniform) and a
> 3-level subcycled AMR hierarchy. Driver: `species_suite`. float32.

## Results
![Abgrall interface — p and u flat across the ρ/γ jump](../figures/species.png)

| Gate | Test | Result |
|---|---|---|
| 1 | Abgrall interface, \|p−1\| sustained | 6.261e-03 (gate 1e-2); max\|u−0.5\| 4.266e-03 |
| 2 | two-gas Sod (uniform), L1(ρ) vs exact | 1.2187e-03 (gate 6e-3) |
| 3 | species mass, 200 steps | drift 7.335e-09 (gate 1e-5) |
| 4 | two-gas Sod on 3-level AMR | L1 2.6227e-03, species drift 1.995e-06 |

## Discussion
The interface stays crisp with **pressure and velocity flat to ~0.6 % / 0.4 %**
across the ρ/γ jump — the Abgrall condition. A tiny bounded wiggle remains
because the reconstruction is on conservative variables; primitive-variable
reconstruction is the documented next refinement, but the sustained oscillation
is already well under the gate and does not grow. The two-gas Sod matches the
generalized exact Riemann solution (each side keeping its own γ across the
contact) both on a uniform grid and through the refluxed 3-level AMR, and
species mass is conserved to the float32 floor. The same two-gas path is
re-exercised under WENO5 in the [WENO5 suite](weno.md) (gates 7–8).

---
*Part of the [V&V dossier](../README.md). Regenerate: `python3 vv/generate.py`. Source data: [`../data/`](../data/).*
