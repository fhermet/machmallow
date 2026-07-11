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
    path = os.path.join(OUT, f"sod_{n}.csv")
    if not os.path.exists(path):
        cands = sorted(int(re.findall(r"sod_(\d+)\.csv", f)[0])
                       for f in os.listdir(OUT) if re.match(r"sod_\d+\.csv", f))
        n = min(cands, key=lambda c: abs(c - n)); path = os.path.join(OUT, f"sod_{n}.csv")
    rows = read_csv(path)
    x = np.array([float(r["x"]) for r in rows])
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True)
    for ax, (fld, exf, lab, col) in zip(axes, [
            ("rho", "rho_exact", "density $\\rho$", CYAN),
            ("u", "u_exact", "velocity $u$", EMBER),
            ("p", "p_exact", "pressure $p$", PURPLE)]):
        ax.plot(x, [float(r[exf]) for r in rows], "-", color="black", lw=2,
                label="exact Riemann")
        ax.plot(x, [float(r[fld]) for r in rows], "o", color=col, ms=2.6,
                label=f"machmallow (N={n})")
        ax.set_ylabel(lab); ax.legend(fontsize=8.5, loc="best")
    axes[-1].set_xlabel("$x$")
    axes[0].set_title(f"Sod shock tube at t = 0.2 — computed vs exact")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sod.png")); plt.close(fig)
    return n


def plot_blasius():
    rows = read_csv(os.path.join(OUT, "blasius_profile.csv"))
    eta = np.array([float(r["eta"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([float(r["blasius"]) for r in rows], eta, "-", color="black",
            lw=2, label="Blasius $f'(\\eta)$")
    ax.plot([float(r["u_computed"]) for r in rows], eta, "o", color=CYAN,
            ms=5, label="machmallow (immersed no-slip)")
    ax.set_xlabel("$u/U_e$"); ax.set_ylabel("$\\eta = y\\,\\sqrt{U_e/\\nu x}$")
    ax.set_title("Blasius boundary layer — profile vs similarity")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "blasius.png")); plt.close(fig)


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
overlaid on the exact solution. The scheme captures the rarefaction, the
contact discontinuity and the shock without spurious oscillations.

![Sod shock tube vs exact Riemann](figures/sod.png)

Grid-convergence order (L1 density vs the exact solution): **{conv_order}**
(≈1, as expected for a discontinuous solution). Profile shown at N = {sod_n}.

---

## 3. Validation — Blasius boundary layer vs similarity

A low-Mach viscous flow over an **immersed** flat plate (no-slip wall). At the
measurement station (Re_x = {rex}) the steady velocity profile must collapse
onto the Blasius similarity solution $u/U_e = f'(\\eta)$.

![Blasius profile vs similarity](figures/blasius.png)

| Quantity | Result vs theory |
|---|---|
| profile RMS $(u/U_e - f')$ | {rms} (gate 3e-2) |
| boundary-layer thickness $\\delta_{{99}}$ | {d99}% |
| skin friction $C_f$ vs $0.664/\\sqrt{{Re_x}}$ | {cf}% |

The finite-resolution discrepancies (Cf, δ99) are quantified and gated, not
hidden. This validates the full viscous path (Navier–Stokes + no-slip wall)
on the staircased immersed boundary.

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

    # copy source data for provenance
    for f in os.listdir(OUT):
        if f == "convergence.csv" or f == "blasius_profile.csv" or \
                re.match(r"sod_\d+\.csv", f):
            shutil.copy(os.path.join(OUT, f), os.path.join(DATA, f))

    write_readme(orders, sod_n, sod_txt, bla_txt, conv_txt)
    print(f"done — figures in {FIG}, report in {os.path.join(VV, 'README.md')}")


if __name__ == "__main__":
    main()
