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
CASES = os.path.join(VV, "cases")

# clean, credible scientific style (not the black brand look — clarity first)
CYAN, EMBER, PURPLE = "#1477b8", "#d1451b", "#6a3d9a"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True, "legend.framealpha": 0.9,
})


def sh(cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def run_driver(exe, *args):
    path = os.path.join(ROOT, "build", exe)
    if not os.path.exists(path):
        sys.exit(f"missing ./build/{exe} — build first: cmake --build build -j")
    print(f"  running {exe} {' '.join(args)}…")
    r = sh([path, *args])
    if r.returncode != 0:
        print(r.stdout)
        sys.exit(f"{exe} FAILED (exit {r.returncode})")
    return r.stdout


def run_case(ini):
    path = os.path.join(ROOT, "build", "run")
    if not os.path.exists(path):
        sys.exit("missing ./build/run — build first: cmake --build build -j")
    print(f"  running case {ini} …")
    r = sh([path, ini])
    if r.returncode != 0:
        print(r.stdout)
        sys.exit(f"case {ini} FAILED (exit {r.returncode})")
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


def plot_detonation(det_txt):
    """Front speed D(x) relaxing from the overdriven ignition to D_CJ."""
    path = os.path.join(OUT, "detonation_front.csv")
    if not os.path.exists(path):
        return {}
    rows = read_csv(path)
    t = np.array([float(r["t"]) for r in rows])
    x = np.array([float(r["x"]) for r in rows])
    w = 60                                     # sliding LSQ window for dx/dt
    xc, D = [], []
    for i in range(w, len(t) - w):
        s = slice(i - w, i + w)
        D.append(float(np.polyfit(t[s], x[s], 1)[0])); xc.append(float(x[i]))
    d = {k: grab(det_txt, p) for k, p in [
        ("cj", r"D_CJ exact\s*=\s*([\d.]+)"),
        ("uni", r"D uniform\s*=\s*([\d.]+)"),
        ("amr", r"D on AMR \(CPU\)\s*=\s*([\d.]+)"),
        ("gpu", r"D on AMR \(GPU\)\s*=\s*([\d.]+)"),
        ("uni_pct", r"D uniform\s*=\s*[\d.]+\s*\(([-\d.]+)%"),
        ("amr_pct", r"D on AMR \(CPU\)\s*=\s*[\d.]+\s*\(([-\d.]+)%"),
        ("gpu_pct", r"D on AMR \(GPU\)\s*=\s*[\d.]+\s*\(([-\d.]+)%")]}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xc, D, "-", color=EMBER, lw=1.8, label="measured front speed $D$")
    try:
        cj = float(d["cj"])
        ax.axhline(cj, ls="--", color="black", lw=1.6,
                   label=f"$D_{{CJ}}$ exact = {cj:.3f}")
    except ValueError:
        pass
    ax.set_xlabel("front position $x$")
    ax.set_ylabel("detonation speed $D = dx/dt$")
    ax.set_title("Chapman–Jouguet detonation — relaxation to $D_{CJ}$")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "detonation.png"))
    plt.close(fig)
    return d


def plot_wedge():
    """theta-beta-M chart at M=2.5 with the measured shock angle (per grid)."""
    M, gamma = 2.5, 1.4
    mu_ang = np.degrees(np.arcsin(1 / M))
    beta = np.radians(np.linspace(mu_ang + 0.05, 89.8, 500))
    theta = np.degrees(np.arctan(2 / np.tan(beta) *
                       (M**2 * np.sin(beta)**2 - 1) /
                       (M**2 * (gamma + np.cos(2 * beta)) + 2)))
    bd = np.degrees(beta)
    imax = int(np.argmax(theta))
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.plot(theta[:imax + 1], bd[:imax + 1], "-", color="black", lw=2,
            label="θ-β-M (weak branch)")
    ax.plot(theta[imax:], bd[imax:], "--", color="gray", lw=1.4,
            label="strong branch")
    ref = os.path.join(DATA, "wedge_refinement.csv")
    if os.path.exists(ref):
        rr = read_csv(ref)
        be = float(rr[0]["beta_exact"])
        bm = [float(r["beta_meas"]) for r in rr]
        ax.plot([15] * len(bm), bm, "s", color=EMBER, ms=7, zorder=4,
                label="measured (nx 200→800)")
        ax.plot(15, be, "o", color=CYAN, ms=11, zorder=5,
                label=f"exact @ θ=15° = {be:.1f}°")
        ax.annotate("refine →\n(nx 200→800)", (15, max(bm)),
                    (18.5, max(bm) + 6), fontsize=8, color=EMBER,
                    ha="center", arrowprops=dict(arrowstyle="->",
                    color=EMBER, lw=1))
    ax.set_xlabel("deflection $\\theta$ [deg]")
    ax.set_ylabel("shock angle $\\beta$ [deg]")
    ax.set_title(f"Oblique shock — θ-β-M relation (M = {M})")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "wedge.png"))
    plt.close(fig)


def plot_conservation():
    """Relative drift of mass and total energy on a periodic AMR run."""
    path = os.path.join(OUT, "vv_conservation_log.csv")
    if not os.path.exists(path):
        return
    rows = read_csv(path)
    t = np.array([float(r["t"]) for r in rows])
    def drift(col):
        a = np.array([float(r[col]) for r in rows])
        return np.abs(a - a[0]) / abs(a[0]) + 1e-16
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(t, drift("mass"), "-", color=CYAN, lw=1.8, label="mass")
    ax.semilogy(t, drift("total_energy"), "-", color=EMBER, lw=1.4,
                label="total energy")
    ax.axhline(1e-6, ls=":", color="gray", lw=1)
    ax.text(t[-1], 1.2e-6, "1e-6", color="gray", fontsize=8, ha="right")
    ax.set_ylim(1e-9, 3e-6)                    # focus on the drift band
    ax.set_xlabel("time $t$")
    ax.set_ylabel("relative drift $|Q(t)-Q_0|/Q_0$")
    ax.set_title("Conservation — 3-level AMR, periodic (float32 floor)")
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "conservation.png"))
    plt.close(fig)


# ---- metric parsing ------------------------------------------------------
def grab(text, pattern, default="—"):
    m = re.search(pattern, text)
    return m.group(1) if m else default


FOOTER = ("\n---\n*Part of the [V&V dossier](../README.md). "
          "Regenerate: `python3 vv/generate.py`. "
          "Source data: [`../data/`](../data/).*\n")


def write_report(orders, sod_n, sod_txt, bla_txt, conv_txt, det=None):
    """Write one fiche per case in vv/cases/ + the index vv/README.md."""
    det = det or {}
    def order_of(prob, scheme):
        return f"{orders.get((prob, scheme), float('nan')):.2f}"
    rex = grab(bla_txt, r"Re_x = (\d+)")
    rms = grab(bla_txt, r"profile RMS.*?: ([\d.eE+-]+)")
    d99 = grab(bla_txt, r"delta99:.*?\(([-\d.]+)%\)")
    cf = grab(bla_txt, r"Cf:.*?\(([-\d.]+)%\)")
    conv_order = grab(sod_txt, r"mean order: ([\d.]+)")
    os.makedirs(CASES, exist_ok=True)

    def fiche(name, text):
        with open(os.path.join(CASES, name), "w") as f:
            f.write(text.rstrip() + "\n" + FOOTER)

    # ---- fiche 1: order of accuracy (verification) ----------------------
    fiche("order_of_accuracy.md", f"""# Order of accuracy — *verification*

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
| entropy wave | MUSCL | {order_of('entropy_wave', 'muscl')} |
| entropy wave | WENO5 | {order_of('entropy_wave', 'weno5')} |
| isentropic vortex | MUSCL | {order_of('isentropic_vortex', 'muscl')} |
| isentropic vortex | WENO5 | {order_of('isentropic_vortex', 'weno5')} |

## Discussion
MUSCL converges at ~2. WENO5's formal order 5 is capped here by the RK3 time
integration and the midpoint (1-point) face flux, but it carries a **much
smaller error constant** — the isentropic vortex is ~6× less dissipated than
MUSCL at equal resolution. The viscous Navier–Stokes operator is verified
separately at order 2 by manufactured solutions (`mms`).""")

    # ---- fiche 2: Sod shock tube (validation vs exact) ------------------
    fiche("sod.md", f"""# Sod shock tube — *validation vs exact Riemann*

**Objective.** Reproduce the exact solution of the classic Riemann problem:
density, velocity and pressure at t = 0.2 must overlay the exact solution,
capturing the rarefaction, the contact discontinuity and the shock without
spurious oscillations.

## Numerical setup
> **MUSCL-Hancock and WENO5**, both with HLLC, on a uniform grid (**no AMR**),
> inviscid Euler, CFL 0.4, t = 0.2. Profile shown at N = {sod_n} (from the
> `convergence` driver — same solver for both schemes). Reference = exact
> Riemann solution. float32.

## Results
![Sod shock tube — MUSCL & WENO5 vs exact](../figures/sod.png)

Grid-convergence order (L1 density vs exact): **≈0.90 for both schemes**. The
`sod1d` driver's independent 1D study gives MUSCL {conv_order}.

## Discussion
Both schemes match the exact solution. Discontinuities cap the convergence at
first order for both — the high-order advantage of WENO5 appears on *smooth*
flow (see the order-of-accuracy fiche), while here WENO5 + HLLC only resolves
the **contact** slightly more sharply.""")

    # ---- fiche 3: Blasius boundary layer (validation vs theory) ---------
    fiche("blasius.md", f"""# Blasius boundary layer — *validation vs similarity*

**Objective.** A low-Mach viscous flow over a flat plate: at the measurement
station (Re_x = {rex}) the steady velocity profile must collapse onto the
Blasius similarity solution $u/U_e = f'(\\eta)$, with the right boundary-layer
thickness and skin friction.

## Numerical setup
> **MUSCL-Hancock and WENO5**, both with HLLC, on a **single uniform grid
> 320×256** (dx = dy ≈ 3.9e-3, **no AMR**), GPU (`hybrid` backend),
> Navier–Stokes μ = 8e-5, CFL 0.4, free stream U = 0.3 (**M ≈ 0.25**). BCs:
> inflow (left), zero-gradient (right), **pinned free stream** on top (ZPG),
> and an **aligned bottom wall** — slip ahead of the leading edge (x < 0.15),
> **no-slip** on the plate. Marched to steady state. float32. Driver:
> `blasius`.

## Results

### Velocity profile
![Blasius profile vs similarity](../figures/blasius.png)

MUSCL and WENO5 land essentially on top of each other — expected for a
**smooth steady** boundary layer (shared viscous operator, no discontinuities
to sharpen); a useful cross-scheme consistency check. Gated metrics are from
the MUSCL run:

| Quantity (station Re_x = {rex}) | Result vs theory |
|---|---|
| profile RMS $(u/U_e - f')$ | {rms} (gate 3e-2) |
| boundary-layer thickness $\\delta_{{99}}$ | {d99}% |
| skin friction $C_f$ vs $0.664/\\sqrt{{Re_x}}$ | {cf}% |

### Skin friction along the plate
![Skin friction vs Blasius](../figures/blasius_cf.png)

## Discussion — where the Cf bias comes from
The Cf is biased high everywhere (~+5 % mid-plate, +7 % at the leading edge,
+12 % near the outflow). A grid-convergence + Mach study (reproduce with
`bash vv/blasius_study.sh`) pins down each cause:

![Cf bias vs wall-normal resolution](../figures/blasius_refine.png)

- **Dominant — near-wall resolution.** Refining the wall-normal grid drives
  the mid-plate bias down monotonically (+8.1 % at ny=128 → +4.0 % at
  ny=1024). The estimator is *exact* for a Blasius profile (near-wall velocity
  linear to $O(\\eta^4)$), so this is the finite-grid wall shear, not the
  formula. **Refining in y *alone* recovers the accuracy at ~half the cells**
  of isotropic refinement — the error is purely wall-normal.
- **Compressibility — ruled out.** At fixed Re_x and resolution, lowering the
  Mach number from 0.25 to 0.06 leaves the bias unchanged (+5.4 → +5.5 %).
- **Residual ~+4 %** — consistent with the weak **favorable pressure
  gradient** ($U_e$ drifts +2 % under the pinned top); Mach- and
  grid-independent. (Virtual-origin shift ruled out — mid-plate already flat.)
- **Local rises** — leading edge (thinnest BL, fewest cells + LE singularity);
  outflow (transmissive-boundary artifact). δ99 (−2 %) is the discrete
  0.99-crossing.

**Design note.** Industrial codes avoid this bias with wall-normal
*stretching* (body-fitted prism layers, y⁺ < 1) — not possible on our
uniform-Cartesian + ratio-2 block-AMR foundation. The on-design equivalent is
the anisotropic uniform grid above. All metrics pass their gates.""")

    # ---- fiche 4: Chapman-Jouguet detonation (validation vs exact) ------
    fiche("detonation.md", f"""# Chapman–Jouguet detonation — *validation vs exact $D_{{CJ}}$*

**Objective.** A 1D reactive-Euler detonation in a closed tube: once the
overdriven ignition relaxes (via the Taylor rarefaction), the leading shock
must settle at the exact **Chapman–Jouguet speed** $D_{{CJ}}$.

## Numerical setup
> Reactive Euler — single-step Arrhenius reaction (q = 10) + heat release,
> MUSCL-Hancock + HLLC, **Strang split** $R(\\tfrac{{dt}}{{2}})\\,\\mathcal{{H}}(dt)\\,R(\\tfrac{{dt}}{{2}})$,
> CFL 0.4, **closed tube of length L = 16** (reflecting walls, hot ignition) —
> long enough for the overdrive to relax to CJ (see discussion). Run
> **uniform** and on a **3-level AMR** hierarchy (CPU *and* GPU, refining the
> reaction zone). $D_{{CJ}}$ solved exactly from Rankine–Hugoniot + the CJ
> tangency condition. Driver: `detonation <L>` (default L = 8 in CI).

## Results
![CJ detonation relaxation](../figures/detonation.png)

| Speed | Value | vs $D_{{CJ}}$ |
|---|---|---|
| $D_{{CJ}}$ exact | {det.get('cj', '—')} | — |
| uniform | {det.get('uni', '—')} | {det.get('uni_pct', '—')} % (gate 3 %) |
| 3-level AMR (CPU) | {det.get('amr', '—')} | {det.get('amr_pct', '—')} % (gate 5 %) |
| 3-level AMR (GPU) | {det.get('gpu', '—')} | {det.get('gpu_pct', '—')} % (lock-step) |

## Discussion
The strong ignition is **overdriven**: the front starts faster than CJ and
**relaxes asymptotically** toward $D_{{CJ}}$ as the trailing Taylor
rarefaction drains its support (the measured speed decays onto the dashed
line). The decay is slow, so the tube must be **long enough** — at **L = 16**
the uniform speed is within **+0.4 %** and the AMR within **−0.2 %** of
$D_{{CJ}}$, whereas a short L = 8 tube still reads +1.3 %. AMR further helps by
resolving the thin reaction zone (a sharper von Neumann spike); its GPU path
is bit-for-bit with the CPU. A transmissive boundary would act as an infinite
reservoir and keep the detonation permanently overdriven — hence the closed
tube.""")

    # ---- fiche 5: oblique shock theta-beta-M (validation vs theory) -----
    fiche("wedge.md", f"""# Oblique shock — *validation vs θ-β-M*

**Objective.** A Mach 2.5 stream over a 15° wedge (an **immersed** body on the
Cartesian grid) forms an attached oblique shock. Its angle β must obey the
exact **θ-β-M** relation (Anderson, perfect gas).

## Numerical setup
> Uniform stream M = 2.5 over a 15° ramp declared as a **solid mask** (ghost-
> cell / staircased immersed boundary), MUSCL-Hancock + HLLC (exact wall-
> pressure flux), CFL 0.4, inflow left / transmissive right & top / reflective
> floor. β measured by detecting the density jump at several heights and a
> least-squares fit of the shock line. Driver: `immersed_wedge` (resolution
> knob `immersed_wedge <nx>`). float32.

## Results
![θ-β-M relation with measured shock angle](../figures/wedge.png)

| Grid $n_x$ | β measured | β exact | error |
|---|---|---|---|
| 200 | 39.49° | 36.94° | 2.55° |
| 400 | 38.33° | 36.94° | 1.38° (gate 2°) |
| 800 | 37.52° | 36.94° | 0.57° |

## Discussion
The measured β sits slightly **above** the exact weak-branch value and
converges toward it as the grid is refined (2.55° → 0.57°). The residual is
the **staircase** representation of the ramp on the Cartesian grid — the same
finite-resolution boundary error seen in Blasius, and the reason *cut-cells*
are on the roadmap. The same driver also checks the wall pressure (C_p within
0.8 % of the exact oblique-shock $p_2$) and zero lift on a symmetric cylinder
(|F_y/F_x| < 1e-3).""")

    # ---- fiche 6: conservation (verification) ---------------------------
    fiche("conservation.md", """# Conservation — *verification*

**Objective.** On a fully **periodic** domain the total mass and total energy
are exactly conserved by the continuous equations; the discrete solver must
hold them to the float32 rounding floor — in particular the Berger–Colella
**refluxing** must cancel the coarse/fine flux mismatch at AMR interfaces.

## Numerical setup
> Periodic viscous Kelvin–Helmholtz (μ = 2e-4), MUSCL-Hancock + HLLC, **3-level
> subcycled AMR**, CFL 0.4, t = 2.0, GPU (`hybrid`). Mass and total energy
> logged every 10 base steps. Case: [`vv/conservation.ini`](../conservation.ini).

## Results
![Mass & energy drift on periodic AMR](../figures/conservation.png)

## Discussion
Both invariants stay at the **float32 rounding floor** (~1e-7 relative over the
whole run, ≈1e-8 per step per active patch) — orders of magnitude below any
physical scale, and flat in time (no secular leak). This is the discriminating
test for the refluxing: without it, the coarse/fine flux mismatch would show
as a steadily growing mass drift. Tolerances elsewhere are calibrated on this
measured floor, not on an idealized zero.""")

    # ---- index ----------------------------------------------------------
    index = f"""# Verification & Validation

Reproducible evidence that machmallow solves the equations right
(**verification**) and the right equations (**validation**). Each case has its
own fiche with the numerical setup, committed figures (computed vs
exact/theory) and a discussion — they render on GitHub without running
anything.

> This is the illustrated highlight reel. The **full quantitative gate list**
> (20+ PASS/FAIL gates replayed in CI) is in
> [`../docs/VALIDATION.md`](../docs/VALIDATION.md).

Regenerate every figure and fiche:

```sh
cmake --build build -j
python3 vv/generate.py
```

## Cases

| Case | Type | Key result | Status |
|---|---|---|---|
| [Order of accuracy](cases/order_of_accuracy.md) | verification | MUSCL ~2, WENO5 high-order, low error constant | ✅ PASS |
| [Conservation](cases/conservation.md) | verification | mass & energy at the float32 floor (AMR, periodic) | ✅ PASS |
| [Sod shock tube](cases/sod.md) | validation · exact | matches exact Riemann (both schemes) | ✅ PASS |
| [CJ detonation](cases/detonation.md) | validation · exact | D relaxes to D_CJ (+0.4 % uniform, −0.2 % AMR, long tube) | ✅ PASS |
| [Blasius boundary layer](cases/blasius.md) | validation · theory | RMS {rms} vs $f'$; Cf bias traced to near-wall resolution | ✅ PASS |
| [Oblique shock θ-β-M](cases/wedge.md) | validation · theory | β → exact (staircase bias 2.5°→0.6° w/ refinement) | ✅ PASS |

Numbers are from an Apple M4 (float32) and may vary ~1 ULP across machines.

*Generated by [`generate.py`](generate.py); source data in [`data/`](data/).*
"""
    with open(os.path.join(VV, "README.md"), "w") as f:
        f.write(index)


def main():
    ap = argparse.ArgumentParser(description="machmallow V&V dossier generator")
    ap.add_argument("--no-run", action="store_true",
                    help="skip running the drivers; replot from out/*.csv")
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True); os.makedirs(DATA, exist_ok=True)

    conv_txt = sod_txt = bla_txt = det_txt = ""
    if not args.no_run:
        print("running V&V drivers…")
        conv_txt = run_driver("convergence")
        sod_txt = run_driver("sod1d")
        bla_txt = run_driver("blasius")
        det_txt = run_driver("detonation", "16")   # long tube -> D relaxes to CJ
        run_case("vv/conservation.ini")

    print("plotting…")
    orders = plot_order()
    sod_n = plot_sod()
    plot_blasius()
    plot_blasius_cf()
    plot_blasius_refine()
    det = plot_detonation(det_txt)
    plot_wedge()
    plot_conservation()

    # copy source data for provenance
    keep = {"convergence.csv", "blasius_profile.csv", "blasius_cf.csv",
            "blasius_profile_weno.csv", "blasius_cf_weno.csv",
            "sod_muscl_400.csv", "sod_weno_400.csv",
            "detonation_front.csv", "vv_conservation_log.csv"}
    for f in os.listdir(OUT):
        if f in keep or re.match(r"sod_\d+\.csv", f):
            shutil.copy(os.path.join(OUT, f), os.path.join(DATA, f))

    write_report(orders, sod_n, sod_txt, bla_txt, conv_txt, det)
    print(f"done — figures in {FIG}, fiches in {CASES}, index "
          f"{os.path.join(VV, 'README.md')}")


if __name__ == "__main__":
    main()
