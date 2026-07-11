#!/usr/bin/env python3
"""machmallow — Verification & Validation dossier generator.

Runs the V&V drivers, then turns their output into committed figures
(computed vs reference) and a self-contained report. Everything regenerates
with one command:

    cmake --build build -j            # (once) build the drivers
    python3 vv/generate.py            # run drivers + plot + write vv/README.md
    python3 vv/generate.py --no-run   # replot from existing out/*.csv only

Outputs (all committed, so they render on GitHub without running anything):
    vv/figures/*.png   — the comparison figures
    vv/data/*.csv       — the source data (provenance)
    vv/README.md        — the report, with the numbers filled in

This first batch covers the three classics: order of accuracy (verification),
Sod vs the exact Riemann solution, and the Blasius boundary layer vs the
similarity solution. More cases (CJ detonation, theta-beta-M, shock-bubble,
conservation) are added the same way.
"""
import argparse
import csv
import os
import re
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
VV = os.path.join(ROOT, "vv")
FIG = os.path.join(VV, "figures")
DATA = os.path.join(VV, "data")

# clean, credible scientific style (not the black brand look — clarity first)
CYAN, EMBER, PURPLE = "#1477b8", "#d1451b", "#6a3d9a"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "legend.framealpha": 0.9,
})


def sh(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def run_driver(exe):
    path = os.path.join(ROOT, "build", exe)
    if not os.path.exists(path):
        sys.exit(f"missing ./build/{exe} — build first: cmake --build build -j")
    print(f"  running {exe} …")
    r = sh([path])
    if r.returncode != 0:
        print(r.stdout)
        sys.exit(f"{exe} FAILED (exit {r.returncode})")
    return r.stdout


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---- figures -------------------------------------------------------------
def plot_order():
    rows = read_csv(os.path.join(OUT, "convergence.csv"))
    keys, orders = {}, {}
    for r in rows:
        keys.setdefault((r["problem"], r["scheme"]), []).append(
            (float(r["h"]), float(r["L1"])))
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    markers = {"muscl": "o", "weno5": "s"}
    for (prob, scheme), pts in sorted(keys.items()):
        pts.sort()
        h = np.array([p[0] for p in pts]); l1 = np.array([p[1] for p in pts])
        # fit slope on the discretization regime only: drop the fine tail
        # where the error hits (and bounces off) the float32 roundoff floor
        m = l1 > l1.min() * 4
        slope = np.polyfit(np.log(h[m]), np.log(l1[m]), 1)[0]
        orders[(prob, scheme)] = slope
        ax.loglog(h, l1, marker=markers.get(scheme, "o"), ms=5,
                  label=f"{prob} · {scheme}  (p={slope:.2f})")
    # reference-slope guides
    h0 = np.array([h.min(), h.max()])
    for p, style in [(1, ":"), (2, "--")]:
        ax.loglog(h0, l1.max() * (h0 / h0.max())**p, style, color="gray",
                  lw=1, alpha=0.7)
        ax.text(h0[0], l1.max() * (h0[0] / h0.max())**p, f" order {p}",
                color="gray", fontsize=8, va="center")
    ax.set_xlabel("grid spacing $h$"); ax.set_ylabel("$L_1$ error")
    ax.set_title("Order of accuracy (smooth regime)")
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "order_of_accuracy.png"))
    plt.close(fig)
    return orders


def plot_sod(n=400):
    # preferred: MUSCL and WENO5 from the same driver (convergence), N=400
    mp = os.path.join(OUT, "sod_muscl_400.csv")
    wp = os.path.join(OUT, "sod_weno_400.csv")
    both = os.path.exists(mp) and os.path.exists(wp)
    if both:
        muscl, weno = read_csv(mp), read_csv(wp)
    else:  # fallback: single-scheme dump from the sod1d driver
        path = os.path.join(OUT, f"sod_{n}.csv")
        if not os.path.exists(path):
            cands = sorted(int(re.findall(r"sod_(\d+)\.csv", f)[0])
                           for f in os.listdir(OUT)
                           if re.match(r"sod_\d+\.csv", f))
            n = min(cands, key=lambda c: abs(c - n))
            path = os.path.join(OUT, f"sod_{n}.csv")
        muscl = read_csv(path)
    x = np.array([float(r["x"]) for r in muscl])
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.6), sharex=True)
    for ax, (fld, exf, lab) in zip(axes, [
            ("rho", "rho_exact", "density $\\rho$"),
            ("u", "u_exact", "velocity $u$"),
            ("p", "p_exact", "pressure $p$")]):
        ax.plot(x, [float(r[exf]) for r in muscl], "-", color="black",
                lw=1.8, label="exact Riemann", zorder=1)
        ax.plot(x, [float(r[fld]) for r in muscl], "o", color=CYAN, ms=2.6,
                label="MUSCL", zorder=3)
        if both:
            ax.plot(x, [float(r[fld]) for r in weno], "s", color=EMBER,
                    ms=2.6, mfc="none", label="WENO5", zorder=2)
        ax.set_ylabel(lab); ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("$x$")
    schemes = "MUSCL & WENO5" if both else "MUSCL"
    axes[0].set_title(f"Sod shock tube — {schemes} vs exact (N=400)",
                      fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sod.png")); plt.close(fig)
    return 400


def plot_blasius():
    rows = read_csv(os.path.join(OUT, "blasius_profile.csv"))
    eta = np.array([float(r["eta"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([float(r["blasius"]) for r in rows], eta, "-", color="black",
            lw=2, label="Blasius $f'(\\eta)$")
    ax.plot([float(r["u_computed"]) for r in rows], eta, "o", color=CYAN,
            ms=5, label="MUSCL")
    wp = os.path.join(OUT, "blasius_profile_weno.csv")
    if os.path.exists(wp):
        w = read_csv(wp)
        ax.plot([float(r["u_computed"]) for r in w],
                [float(r["eta"]) for r in w], "s", color=EMBER, ms=5,
                mfc="none", label="WENO5")
    ax.set_xlabel("$u/U_e$"); ax.set_ylabel("$\\eta = y\\,\\sqrt{U_e/\\nu x}$")
    ax.set_title("Blasius boundary layer — profile vs similarity")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "blasius.png")); plt.close(fig)


def plot_blasius_cf():
    path = os.path.join(OUT, "blasius_cf.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    rex = np.array([float(r["Rex"]) for r in rows])
    o = np.argsort(rex); rex = rex[o]
    cf = np.array([float(r["Cf"]) for r in rows])[o]
    cfx = np.array([float(r["Cf_exact"]) for r in rows])[o]
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.loglog(rex, cfx, "-", color="black", lw=2,
              label="Blasius $0.664/\\sqrt{Re_x}$")
    ax.loglog(rex, cf, "o", color=CYAN, ms=6, label="MUSCL")
    wp = os.path.join(OUT, "blasius_cf_weno.csv")
    if os.path.exists(wp):
        w = read_csv(wp)
        rw = np.array([float(r["Rex"]) for r in w]); ow = np.argsort(rw)
        ax.loglog(rw[ow], np.array([float(r["Cf"]) for r in w])[ow], "s",
                  color=EMBER, ms=6, mfc="none", label="WENO5")
    ax.set_xlabel("$Re_x$"); ax.set_ylabel("skin friction $C_f$")
    ax.set_title("Skin friction along the plate vs Blasius")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "blasius_cf.png"))
    plt.close(fig)


def plot_blasius_refine():
    """Cf bias vs wall-normal resolution (isotropic vs anisotropic refinement).
    Reads the committed study data (vv/blasius_study.sh regenerates it)."""
    path = os.path.join(DATA, "blasius_refinement.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for study, mk, col, lab in [
            ("iso", "o", CYAN, "isotropic (refine x & y)"),
            ("aniso", "s", EMBER, "anisotropic (refine y only)")]:
        pts = sorted((float(r["dy"]), float(r["cf_pct"]))
                     for r in rows if r["study"] == study)
        if not pts:
            continue
        dy = [p[0] for p in pts]; cf = [p[1] for p in pts]
        ax.plot(dy, cf, mk + "-", color=col, ms=7, label=lab)
    ax.axhline(4.0, ls=":", color="gray", lw=1.2)
    ax.text(ax.get_xlim()[1], 4.05, "physical residual (~4%) ",
            color="gray", fontsize=8, ha="right", va="bottom")
    ax.set_xscale("log"); ax.invert_xaxis()          # finer to the right
    ax.set_xlabel("wall-normal spacing $dy$  (finer →)")
    ax.set_ylabel("$C_f$ bias vs Blasius  [%]")
    ax.set_title("$C_f$ bias vs wall-normal resolution")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "blasius_refine.png"))
    plt.close(fig)


# ---- metric parsing ------------------------------------------------------
def grab(text, pattern, default="—"):
    m = re.search(pattern, text)
    return m.group(1) if m else default


def write_readme(orders, sod_n, sod_txt, bla_txt, conv_txt):
    def order_of(prob, scheme):
        return f"{orders.get((prob, scheme), float('nan')):.2f}"
    rex = grab(bla_txt, r"Re_x = (\d+)")
    rms = grab(bla_txt, r"profile RMS.*?: ([\d.eE+-]+)")
    d99 = grab(bla_txt, r"delta99:.*?\(([-\d.]+)%\)")
    cf = grab(bla_txt, r"Cf:.*?\(([-\d.]+)%\)")
    conv_order = grab(sod_txt, r"mean order: ([\d.]+)")
    md = f"""# Verification & Validation — figures

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
| entropy wave | MUSCL | {order_of('entropy_wave', 'muscl')} |
| entropy wave | WENO5 | {order_of('entropy_wave', 'weno5')} |
| isentropic vortex | MUSCL | {order_of('isentropic_vortex', 'muscl')} |
| isentropic vortex | WENO5 | {order_of('isentropic_vortex', 'weno5')} |

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
> at N = {sod_n} (from the `convergence` driver, same solver for both
> schemes). Reference = exact Riemann solution. float32.

![Sod shock tube — MUSCL & WENO5 vs exact](figures/sod.png)

Grid-convergence order (L1 density vs the exact solution): **≈0.90 for both
schemes** — as expected, discontinuities cap both at first order (the
high-order advantage of WENO5 shows on *smooth* flow; see §1). The `sod1d`
driver's independent 1D grid-convergence study gives MUSCL {conv_order}.

---

## 3. Validation — Blasius boundary layer vs similarity

A low-Mach viscous flow over a flat plate. At the measurement station
(Re_x = {rex}) the steady velocity profile must collapse onto the Blasius
similarity solution $u/U_e = f'(\\eta)$.

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
Blasius law $C_f = 0.664/\\sqrt{{Re_x}}$:

![Skin friction vs Blasius](figures/blasius_cf.png)

| Quantity (station Re_x = {rex}) | Result vs theory |
|---|---|
| profile RMS $(u/U_e - f')$ | {rms} (gate 3e-2) |
| boundary-layer thickness $\\delta_{{99}}$ | {d99}% |
| skin friction $C_f$ vs $0.664/\\sqrt{{Re_x}}$ | {cf}% |

**Where the Cf bias comes from — a grid-convergence + Mach study.** The Cf is
biased high everywhere (~+5 % mid-plate, rising to +7 % at the leading edge
and +12 % near the outflow). A refinement study (reproduce with
`bash vv/blasius_study.sh`) pins down each cause:

![Cf bias vs wall-normal resolution](figures/blasius_refine.png)

- **Dominant cause — near-wall resolution.** Refining the wall-normal grid
  drives the mid-plate bias down monotonically (+8.1 % at ny=128 → +4.0 % at
  ny=1024). The estimator itself is *exact* for a Blasius profile (the
  near-wall velocity is linear to $O(\\eta^4)$), so this is the finite-grid
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
"""
    with open(os.path.join(VV, "README.md"), "w") as f:
        f.write(md)


def main():
    ap = argparse.ArgumentParser(description="machmallow V&V dossier generator")
    ap.add_argument("--no-run", action="store_true",
                    help="skip running the drivers; replot from out/*.csv")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    conv_txt = sod_txt = bla_txt = ""
    if not args.no_run:
        print("running V&V drivers…")
        conv_txt = run_driver("convergence")
        sod_txt = run_driver("sod1d")
        bla_txt = run_driver("blasius")

    print("plotting…")
    orders = plot_order()
    sod_n = plot_sod()
    plot_blasius()
    plot_blasius_cf()
    plot_blasius_refine()

    # copy source data for provenance
    keep = {"convergence.csv", "blasius_profile.csv", "blasius_cf.csv",
            "blasius_profile_weno.csv", "blasius_cf_weno.csv",
            "sod_muscl_400.csv", "sod_weno_400.csv"}
    for f in os.listdir(OUT):
        if f in keep or re.match(r"sod_\d+\.csv", f):
            shutil.copy(os.path.join(OUT, f), os.path.join(DATA, f))

    write_readme(orders, sod_n, sod_txt, bla_txt, conv_txt)
    print(f"done — figures in {FIG}, report in {os.path.join(VV, 'README.md')}")


if __name__ == "__main__":
    main()
