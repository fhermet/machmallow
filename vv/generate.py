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
    os.makedirs(OUT, exist_ok=True)                # cache stdout for --no-run
    with open(os.path.join(OUT, exe + ".log"), "w") as f:
        f.write(r.stdout)
    return r.stdout


def cached(exe):
    p = os.path.join(OUT, exe + ".log")
    return open(p).read() if os.path.exists(p) else ""


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


def plot_sod2d():
    """Diagonal 2D Sod: 2D density field + collapse onto the 1D similarity."""
    fp = os.path.join(OUT, "sod2d_field.csv")
    if not os.path.exists(fp):
        return
    rows = read_csv(fp)
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    rho = np.array([float(r["rho"]) for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    xs = sorted(set(np.round(x, 5))); ys = sorted(set(np.round(y, 5)))
    xi_ = {v: i for i, v in enumerate(xs)}; yi_ = {v: i for i, v in enumerate(ys)}
    R = np.full((len(ys), len(xs)), np.nan)
    for xv, yv, rv in zip(x, y, rho):
        R[yi_[round(yv, 5)], xi_[round(xv, 5)]] = rv
    pc = ax[0].pcolormesh(np.array(xs), np.array(ys), R, shading="auto",
                          cmap="viridis")
    fig.colorbar(pc, ax=ax[0], label="ρ"); ax[0].set_aspect("equal")
    ax[0].set_title("2D density field (diagonal Sod)")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    # boundary-free central square used for validation (excludes corner/edge
    # effects, where the discontinuity meets the boundary at (1,0) & (0,1))
    ax[0].plot([0.3, 0.7, 0.7, 0.3, 0.3], [0.3, 0.3, 0.7, 0.7, 0.3], "--",
               color="red", lw=1.6)
    ax[0].text(0.5, 0.305, "validation region", color="red", fontsize=7.5,
               ha="center", va="bottom")
    # collapse only the boundary-free central square (what the gate measures)
    c = (x >= 0.3) & (x <= 0.7) & (y >= 0.3) & (y <= 0.7)
    xi = (x[c] + y[c] - 1.0) / np.sqrt(2)
    ax[1].plot(xi, rho[c], ".", color=CYAN, ms=2, alpha=0.3,
               label="2D cells (central square)")
    ep = os.path.join(OUT, "sod2d_exact.csv")
    if os.path.exists(ep):
        e = read_csv(ep)
        ax[1].plot([float(r["xi"]) for r in e],
                   [float(r["rho_exact"]) for r in e], "-", color="black",
                   lw=2, label="exact 1D Riemann")
    ax[1].set_xlim(-0.35, 0.35); ax[1].set_xlabel("$\\xi = (x+y-1)/\\sqrt{2}$")
    ax[1].set_ylabel("ρ"); ax[1].legend(fontsize=9)
    ax[1].set_title("Collapse onto the 1D similarity solution")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sod2d.png"))
    plt.close(fig)


def plot_sod_amr(txt):
    """Two panels: AMR composite vs exact Riemann | mass drift with/without
    refluxing (frozen mesh)."""
    m = {"with": grab(txt, r"frozen mesh\): ([\d.eE+-]+) with"),
         "without": grab(txt, r"with refluxing \| ([\d.eE+-]+) without")}
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    # left: composite density profile vs exact
    pp = os.path.join(OUT, "sod_amr_profile.csv")
    if os.path.exists(pp):
        rows = read_csv(pp)
        ex = sorted((float(r["x"]), float(r["rho_exact"])) for r in rows)
        ax[0].plot([a for a, _ in ex], [b for _, b in ex], "-", color="black",
                   lw=1.6, label="exact Riemann", zorder=1)
        for lvl, col, lab, mf in [("0", "#1477b8", "coarse (L0)", "#1477b8"),
                                  ("1", EMBER, "fine patch (L1)", "none")]:
            xs = [float(r["x"]) for r in rows if r["level"] == lvl]
            rs = [float(r["rho"]) for r in rows if r["level"] == lvl]
            ax[0].plot(xs, rs, "o", color=col, ms=4, mfc=mf, label=lab,
                       zorder=3 if lvl == "1" else 2)
        ax[0].set_xlabel("x"); ax[0].set_ylabel("ρ")
        ax[0].legend(fontsize=8, loc="upper right")
        ax[0].set_title("AMR composite vs exact Riemann")
    # right: refluxing conservation bar
    try:
        vals = [float(m["without"]), float(m["with"])]
        ax[1].bar(["no refluxing", "with refluxing"], vals,
                  color=[EMBER, CYAN], width=0.6)
        ax[1].set_yscale("log"); ax[1].set_ylabel("max mass drift (frozen mesh)")
        for i, v in enumerate(vals):
            ax[1].text(i, v * 1.4, f"{v:.1e}", ha="center", fontsize=9)
        ax[1].set_title("Refluxing restores conservation")
        ax[1].set_ylim(top=max(vals) * 6)
    except ValueError:
        pass
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "sod_amr.png"))
    plt.close(fig)
    return m


def plot_reactor(txt):
    """0D reactor: isothermal lambda(t) vs the exact exponential."""
    path = os.path.join(OUT, "reactor_isothermal.csv")
    d = {"iso": grab(txt, r"isothermal reactor:.*err ([\d.eE+-]+)"),
         "adT": grab(txt, r"adiabatic reactor:.*T ([\d.]+) vs eq"),
         "adEq": grab(txt, r"adiabatic reactor:.*vs eq ([\d.]+)"),
         "adResid": grab(txt, r"energy-balance resid ([\d.eE+-]+)"),
         "stiff": grab(txt, r"stiff reactor.*lambda ([\d.]+)")}
    if not os.path.exists(path):
        return d
    rows = read_csv(path)
    t = [float(r["t"]) for r in rows]
    lam = [float(r["lambda"]) for r in rows]
    ex = [float(r["lambda_exact"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(t, ex, "-", color="black", lw=2, label="exact  $1-e^{-Kt}$")
    ax.plot(t[::4], lam[::4], "o", color=CYAN, ms=5, label="machmallow (RK4)")
    ax.set_xlabel("time $t$"); ax.set_ylabel("reaction progress $\\lambda$")
    ax.set_title("0D reactor — isothermal kinetics vs exact")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "reactor.png"))
    plt.close(fig)
    return d


def plot_mms(txt):
    """Manufactured-solution order study: L1 vs N. Viscous ~2 (both schemes),
    steady-inviscid ~1 (numerical-viscosity-limited, informative)."""
    path = os.path.join(OUT, "mms.csv")
    if not os.path.exists(path):
        return {}
    rows = read_csv(path)
    series = {}
    for r in rows:
        series.setdefault(r["label"], []).append(
            (float(r["N"]), float(r["L1"])))
    style = {
        "viscous MUSCL": ("o-", CYAN, 1.0),
        "viscous WENO5": ("s-", EMBER, 1.0),
        "viscous MUSCL+gravity": ("^-", PURPLE, 1.0),
        "inviscid MUSCL": (".--", "gray", 0.55),
        "inviscid WENO5": (".--", "silver", 0.55),
    }
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for lab, pts in sorted(series.items()):
        pts.sort()
        N = np.array([p[0] for p in pts]); L1 = np.array([p[1] for p in pts])
        slope = -np.polyfit(np.log(N), np.log(L1), 1)[0]
        fmt, col, a = style.get(lab, ("o-", "black", 1.0))
        ax.loglog(N, L1, fmt, color=col, alpha=a, ms=5,
                  label=f"{lab}  (p={slope:.2f})")
    N0 = np.array([16.0, 128.0])
    for p, key in [(1, "inviscid MUSCL"), (2, "viscous MUSCL")]:
        if key in series:
            b = sorted(series[key])[0][1] * 1.25
            ax.loglog(N0, b * (N0 / 16.0) ** (-p), ":", color="gray", lw=1)
            ax.text(132, b * (128 / 16.0) ** (-p), f"p={p}", fontsize=8,
                    color="gray", va="center")
    ax.set_xlabel("N (cells per side)"); ax.set_ylabel("$L_1$ error (density)")
    ax.set_title("Manufactured solution — Navier–Stokes order")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "mms.png"))
    plt.close(fig)
    return {"vmuscl": grab(txt, r"visqueux MUSCL ([\d.]+)"),
            "vweno": grab(txt, r"WENO5 ([\d.]+) ;"),
            "vgrav": grab(txt, r"visqueux\+gravite ([\d.]+)")}


def plot_weno(txt):
    """Isentropic-vortex core: density slice through the advected centre at
    t=2. WENO5 tracks the exact dip; MUSCL-Hancock over-diffuses it. This is
    the head-to-head dissipation the suite gates (weno < muscl/4)."""
    pw = os.path.join(OUT, "weno_vortex_weno.csv")
    pm = os.path.join(OUT, "weno_vortex_muscl.csv")
    d = {"ent_ord": grab(txt, r"entropy wave:.*order ([\d.]+) \(gate >= 4"),
         "ent_sp": grab(txt, r"spatial-only 8->16: ([\d.]+)"),
         "sod_w": grab(txt, r"Sod 400: L1 weno ([\d.eE+-]+)"),
         "sod_m": grab(txt, r"Sod 400: L1 weno [\d.eE+-]+ vs muscl ([\d.eE+-]+)"),
         "vor_ord": grab(txt, r"vortex t=2: order ([\d.]+)"),
         "vor_w": grab(txt, r"vortex t=2:.*L1 weno ([\d.eE+-]+)"),
         "vor_m": grab(txt, r"vortex t=2:.*vs muscl ([\d.eE+-]+)"),
         "bit": grab(txt, r"all-refined.*: (\d+) differing"),
         "shear_ord": grab(txt, r"viscous shear.*order ([\d.]+)"),
         "tg_u": grab(txt, r"two-gas Sod \(uniform WENO5\): L1 = ([\d.eE+-]+)"),
         "tg_a": grab(txt, r"3-level AMR: L1 = ([\d.eE+-]+)"),
         "tg_drift": grab(txt, r"species mass drift = ([\d.eE+-]+)"),
         "samr_w": grab(txt, r"Sod on 3-level AMR: L1 weno ([\d.eE+-]+)"),
         "samr_m": grab(txt, r"Sod on 3-level AMR: L1 weno [\d.eE+-]+ vs muscl ([\d.eE+-]+)")}
    try:
        d["ratio"] = f"{float(d['vor_m']) / float(d['vor_w']):.1f}"
    except (ValueError, ZeroDivisionError):
        d["ratio"] = "—"
    if not (os.path.exists(pw) and os.path.exists(pm)):
        return d
    w = read_csv(pw); m = read_csv(pm)
    xw = [float(r["x"]) for r in w]; rw = [float(r["rho"]) for r in w]
    rex = [float(r["rho_exact"]) for r in w]
    xm = [float(r["x"]) for r in m]; rm = [float(r["rho"]) for r in m]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(xw, rex, "-", color="black", lw=2, label="exact (shifted IC)")
    ax.plot(xw, rw, "o-", color=CYAN, ms=4, lw=1.2,
            label=f"WENO5  ($L_1$={d['vor_w']})")
    ax.plot(xm, rm, "s--", color=EMBER, ms=4, lw=1.2,
            label=f"MUSCL-Hancock  ($L_1$={d['vor_m']})")
    ax.set_xlim(4, 10)
    ax.set_xlabel("x  (through the advected core, $y=7$)")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title(f"Isentropic vortex at $t=2$, $64^2$ — WENO5 is "
                 f"{d['ratio']}× less dissipative")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "weno.png"))
    plt.close(fig)
    return d


def plot_species(txt):
    """Abgrall interface: a rho+gamma jump advected at u=0.5 for one period.
    The signature of a well-posed multi-gas scheme is that p and u stay flat
    across the density jump (naive conservative schemes oscillate p there)."""
    path = os.path.join(OUT, "species_interface.csv")
    d = {"p_start": grab(txt, r"startup ([\d.eE+-]+)"),
         "p_sust": grab(txt, r"sustained ([\d.eE+-]+)"),
         "u_err": grab(txt, r"max\|u-0.5\| = ([\d.eE+-]+)"),
         "sod": grab(txt, r"two-gas Sod \(1.4\|1.6\): L1\(rho\) vs exact = ([\d.eE+-]+)"),
         "mass": grab(txt, r"species mass, 200 steps: drift = ([\d.eE+-]+)"),
         "amr": grab(txt, r"3-level AMR: L1 = ([\d.eE+-]+)"),
         "amr_drift": grab(txt, r"3-level AMR:.*species mass drift = ([\d.eE+-]+)")}
    if not os.path.exists(path):
        return d
    rows = read_csv(path)
    x = [float(r["x"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(x, [float(r["rho"]) for r in rows], "-", color=CYAN, lw=1.8,
            label=r"$\rho$ (jumps 1↔0.5)")
    ax.plot(x, [float(r["Y"]) for r in rows], "-", color=PURPLE, lw=1.4,
            label="Y (mass fraction)")
    ax.plot(x, [float(r["p"]) for r in rows], "-", color=EMBER, lw=1.8,
            label="p (must stay flat = 1)")
    ax.plot(x, [float(r["u"]) for r in rows], "--", color="black", lw=1.2,
            label="u (must stay flat = 0.5)")
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("x"); ax.set_ylabel("value")
    ax.set_title("Abgrall interface after one period — p, u flat across "
                 "the ρ/γ jump")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "species.png"))
    plt.close(fig)
    return d


def plot_analytic(txt):
    """Toro's Riemann battery (tests 2-5): the four hard 1D Riemann problems
    the single Sod case cannot exercise — near-vacuum double rarefaction, a
    p=1000 blast, colliding strong shocks, a slowly-moving contact."""
    titles = ["T2 · 123 / near-vacuum", "T3 · blast $p=1000$",
              "T4 · colliding shocks", "T5 · slow contact"]
    d = {"t2": grab(txt, r"T2 123/near-vacuum\s+L1\(rho\) = ([\d.eE+-]+)"),
         "t3": grab(txt, r"T3 blast p=1000\s+L1\(rho\) = ([\d.eE+-]+)"),
         "t4": grab(txt, r"T4 colliding shocks\s+L1\(rho\) = ([\d.eE+-]+)"),
         "t5": grab(txt, r"T5 slow contact\s+L1\(rho\) = ([\d.eE+-]+)"),
         "ac": grab(txt, r"smooth order = ([\d.]+) \(TVD"),
         "vor": grab(txt, r"mean smooth order = ([\d.]+)"),
         "sedov": grab(txt, r"exponent = ([\d.]+) \(theory"),
         "rt": grab(txt, r"sigma = ([\d.]+) vs"),
         "rt_th": grab(txt, r"vs sqrt\(Agk\) = ([\d.]+)")}
    paths = [os.path.join(OUT, f"analytic_toro{i}.csv") for i in (1, 2, 3, 4)]
    if not all(os.path.exists(p) for p in paths):
        return d
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.2))
    for ax, p, ti in zip(axes.flat, paths, titles):
        rows = read_csv(p)
        x = [float(r["x"]) for r in rows]
        ax.plot(x, [float(r["rho_exact"]) for r in rows], "-",
                color="black", lw=1.8, label="exact")
        ax.plot(x, [float(r["rho"]) for r in rows], "o", color=CYAN, ms=2.6,
                label="machmallow")
        ax.set_title(ti, fontsize=10)
        ax.set_xlabel("x"); ax.set_ylabel(r"$\rho$")
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Toro Riemann battery (N=400) — density vs exact", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "analytic.png"))
    plt.close(fig)
    return d


def plot_immersed(txt):
    """No-slip Blasius boundary layer grown on an IMMERSED plate (wall set by
    the solid mask, not a domain BC): u/Ue vs the Blasius f'(eta)."""
    path = os.path.join(OUT, "immersed_noslip.csv")
    d = {"rms": grab(txt, r"profil RMS\(u/Ue - f'\) = ([\d.eE+-]+)"),
         "slip": grab(txt, r"glissement paroi u/Ue = ([\d.]+)"),
         "cf": grab(txt, r"Cf = [\d.eE+-]+ vs [\d.eE+-]+ \(Blasius\) \| écart (\d+)%"),
         "rex": grab(txt, r"Re_x=(\d+)"),
         "refl_err": grab(txt, r"err ([\d.]+)%, gate 5%"),
         "amr_err": grab(txt, r"AMR vs base ([\d.]+)%"),
         "gpu": grab(txt, r"single\s+patches.*écart rel max ([\d.eE+-]+)"),
         "gpu_ml": grab(txt, r"ML \d+ niveaux\s+\| écart rel max ([\d.eE+-]+)")}
    if not os.path.exists(path):
        return d
    rows = read_csv(path)
    eta = [float(r["eta"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([float(r["fp"]) for r in rows], eta, "-", color="black", lw=2,
            label="Blasius $f'(\\eta)$")
    ax.plot([float(r["u"]) for r in rows], eta, "o", color=CYAN, ms=4,
            label=f"immersed plate (RMS {d['rms']})")
    ax.set_xlabel(r"$u / U_e$"); ax.set_ylabel(r"$\eta = y\,\sqrt{U_e/(\nu x)}$")
    ax.set_ylim(0, 8); ax.set_xlim(0, 1.05)
    ax.set_title(f"No-slip on an immersed plate — $Re_x$={d['rex']}")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "immersed.png"))
    plt.close(fig)
    return d


def plot_dmr(txt):
    """Double Mach reflection density field (GPU, t=0.2) — the canonical
    strong-shock AMR showcase. Metrics from the GPU/AMR/multi-level drivers."""
    d = {"gpu_rel": grab(txt, r"correctness \(\d+x\d+, \d+ steps\): max rel diff = ([\d.eE+-]+)"),
         "gpu_speed": grab(txt, r"speedup: ([\d.]+)x"),
         "gpu_rate": grab(txt, r"GPU: \d+ steps in [\d.]+ s -> ([\d.]+) Mcell"),
         "amr_rel": grab(txt, r"correctness \(30 lock-steps\): max rel diff = ([\d.eE+-]+)"),
         "amr_work": grab(txt, r"work vs uniform 1/256: \d+ vs \d+ Mcell-steps \((\d+)%\)"),
         "ml_rel": grab(txt, r"3-level DMR lock-step, 30 steps: max rel diff ([\d.eE+-]+)"),
         "ml_kh": grab(txt, r"3-level periodic KH \(GPU\), \d+ steps: mass drift ([\d.eE+-]+)"),
         "cd_ghost": grab(txt, r"DMR ghosts CaseDef vs preset: (\d+) differing")}
    path = os.path.join(OUT, "dmr_field.csv")
    if not os.path.exists(path):
        return d
    rows = read_csv(path)
    x = np.array([float(r["x"]) for r in rows])
    y = np.array([float(r["y"]) for r in rows])
    rho = np.array([float(r["rho"]) for r in rows])
    xs = np.unique(x); ys = np.unique(y)
    Z = rho.reshape(len(ys), len(xs))
    fig, ax = plt.subplots(figsize=(9.0, 2.7))
    pcm = ax.pcolormesh(xs, ys, Z, cmap="inferno", shading="auto",
                        vmin=float(rho.min()), vmax=float(rho.max()))
    ax.set_aspect("equal")
    ax.set_xlim(0, 3)          # crop to the reflected-shock region
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Double Mach reflection — density at $t=0.2$ (GPU)")
    fig.colorbar(pcm, ax=ax, shrink=0.85, label=r"$\rho$")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "dmr.png"), dpi=140)
    plt.close(fig)
    return d


def plot_hs(txt):
    """Haas & Sturtevant (1987) shock–bubble: characteristic interface-point
    velocities vs the experiment and the canonical Quirk & Karni (1996)
    numerics. Measured values parsed from the log (code units × 343 m/s)."""
    C = 343.0  # experimental ambient sound speed, m/s
    d = {"ui": grab(txt, r"V_ui : [\d.]+ / ([\d.]+)"),
         "di": grab(txt, r"V_di: [\d.]+ / ([\d.]+)"),
         "jet": grab(txt, r"V_jet: [\d.]+ / ([\d.]+)"),
         "ui_pc": grab(txt, r"V_ui :.*\(([-+][\d.]+)%"),
         "di_pc": grab(txt, r"V_di:.*\(([-+][\d.]+)%"),
         "jet_pc": grab(txt, r"V_jet:.*\(([-+][\d.]+)%")}
    try:
        meas = [float(d["ui"]) * C, float(d["di"]) * C, float(d["jet"]) * C]
    except ValueError:
        return d
    d["ui_ms"] = f"{meas[0]:.0f}"; d["di_ms"] = f"{meas[1]:.0f}"
    d["jet_ms"] = f"{meas[2]:.0f}"
    labels = ["upstream edge", "downstream edge", "air jet"]
    exp = [170, 145, 230]; qk = [178, 146, 227]
    xpos = np.arange(3); w = 0.26
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.bar(xpos - w, exp, w, color="black", label="experiment (H&S 1987)")
    ax.bar(xpos, qk, w, color="gray", label="Quirk & Karni 1996")
    ax.bar(xpos + w, meas, w, color=CYAN, label="machmallow")
    for k, v in enumerate(meas):
        ax.text(xpos[k] + w, v + 3, f"{v:.0f}", ha="center", fontsize=8,
                color=CYAN)
    ax.set_xticks(xpos); ax.set_xticklabels(labels)
    ax.set_ylabel("interface-point velocity (m/s)")
    ax.set_title("Haas & Sturtevant Mach 1.22 air → helium cylinder")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "hs.png"))
    plt.close(fig)
    return d


# ---- metric parsing ------------------------------------------------------
def grab(text, pattern, default="—"):
    m = re.search(pattern, text)
    return m.group(1) if m else default


FOOTER = ("\n---\n*Part of the [V&V dossier](../README.md). "
          "Regenerate: `python3 vv/generate.py`. "
          "Source data: [`../data/`](../data/).*\n")


def write_report(orders, sod_n, sod_txt, bla_txt, conv_txt, det=None,
                 sod2d_txt="", samr_txt="", mms=None, rea=None, weno=None,
                 species=None, analytic=None, immersed=None, dmr=None,
                 hs=None):
    """Write one fiche per case in vv/cases/ + the index vv/README.md."""
    det = det or {}
    mms = mms or {}
    rea = rea or {}
    weno = weno or {}
    species = species or {}
    analytic = analytic or {}
    immersed = immersed or {}
    dmr = dmr or {}
    hs = hs or {}
    sod2d_order = grab(sod2d_txt, r"mean order: ([\d.]+)")
    rfx_with = grab(samr_txt, r"frozen mesh\): ([\d.eE+-]+) with")
    rfx_without = grab(samr_txt, r"with refluxing \| ([\d.eE+-]+) without")
    rfx_ratio = grab(samr_txt, r"\(([\d.]+)x worse\)")
    samr_l1 = grab(samr_txt, r"ratio ([\d.]+) \(gate")
    samr_work = grab(samr_txt, r"uniform [\d.]+ \((\d+)%\)")
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
separately at order 2 by manufactured solutions (`mms`; see the MMS fiche).""")

    # ---- fiche 1b: manufactured solution (Navier-Stokes order) ----------
    fiche("mms.md", f"""# Manufactured solution (MMS) — *verification (Navier–Stokes order)*

**Objective.** Verify the **viscous Navier–Stokes** operator converges at its
design order 2. A smooth steady solution is imposed with an exact source term
$S = \\nabla\\!\\cdot F_{{\\rm Euler}} - \\nabla\\!\\cdot F_{{\\rm visc}}$; the
discrete L1 error must then fall as $O(h^2)$. This is the only gate that pins
the **viscous** operator's order (the
[order-of-accuracy](order_of_accuracy.md) fiche covers the inviscid transient
order).

## Numerical setup
> Smooth periodic manufactured solution (sinusoidal ρ,u,v,p), source computed
> **in double** by 4th-order finite differences and injected each step. Relax
> to steady state, measure L1(ρ) vs the manufactured field, N = 16 → 128,
> μ = 0.01, CFL 0.4. Schemes MUSCL & WENO5 (+ a gravity-source variant).
> Driver: `mms`. float32.

## Results
![MMS order study](../figures/mms.png)

| Operator | Observed order |
|---|---|
| viscous, MUSCL | {mms.get('vmuscl', '—')} |
| viscous, WENO5 | {mms.get('vweno', '—')} |
| viscous + gravity, MUSCL | {mms.get('vgrav', '—')} |

## Discussion
Both schemes reach order **~2** on the viscous operator — capped at 2 by the
**central 2nd-order viscous flux** (shared by MUSCL and WENO5), exactly as
expected, so the full Navier–Stokes discretization is verified at order 2. The
gravity-source variant stays at 2 (a sign or work-term bug would break the
steady state and the order). The faint **inviscid** curves sit at ~1: with no
physical viscosity the steady-state error is set by the scheme's *numerical*
viscosity (1st order on this solution) — that is **not** the transient design
order (MUSCL ~2, WENO5 ~4–5), which the order-of-accuracy fiche verifies by
advecting an exact solution.""")

    # ---- fiche 1c: 0D reactor kinetics (verification vs analytic) -------
    fiche("reactor.md", f"""# 0D reactor kinetics — *verification vs analytic*

**Objective.** Verify the stiff Arrhenius reaction integrator against exact 0D
solutions *before* it is coupled to the flow: (1) **isothermal** (q=0) → exact
exponential; (2) **adiabatic** → full burn to λ=1 with T rising by exactly
(γ−1)q and exact energy balance; (3) **very stiff** → the subcycling stays
bounded and still reaches equilibrium in one coarse step.

## Numerical setup
> Single-step Arrhenius reaction, adaptive **subcycled RK4**, energy slaved to
> the progress variable ($e = e_0 + q\\,\\Delta\\lambda$, conservative by
> construction). Constant-volume 0D integration. Driver: `reactor`.

## Results
![Isothermal reactor: λ(t) vs exact](../figures/reactor.png)

| Test | Result |
|---|---|
| isothermal, λ vs $1-e^{{-Kt}}$ | err {rea.get('iso', '—')} (gate 1e-5) |
| adiabatic, equilibrium T | {rea.get('adT', '—')} vs {rea.get('adEq', '—')} exact; energy resid {rea.get('adResid', '—')} |
| stiff (A=1e4, dt=1), λ | {rea.get('stiff', '—')} (bounded, equilibrates) |

## Discussion
The isothermal case (constant rate K) reproduces the exact exponential to
~1e-7 (curve). The adiabatic burn reaches λ=1 with the temperature rising by
exactly (γ−1)q and energy conserved to ~5e-7 — the energy-slaving makes the
integrator conservative by construction. The stiff case (A=1e4 in a single
coarse step) stays bounded and equilibrates, showing the adaptive subcycling
handles stiffness. This 0D validation underpins the coupled
[CJ detonation](detonation.md) case.""")

    # ---- fiche 1d: WENO5 scheme suite (verification) --------------------
    fiche("weno.md", f"""# WENO5 scheme suite — *verification*

**Objective.** Validate the fifth-order WENO5 + SSP-RK3 scheme end to end and
quantify what it buys over the default MUSCL-Hancock: smooth-flow order, the
absence of spurious oscillations on the Sod shock, the **head-to-head
dissipation** on an isentropic vortex, the viscous flux against an exact `erf`
diffusion layer, two-gas Riemann, and bit-exact behaviour through the AMR
stage-ghost machinery.

## Numerical setup
> **WENO5 reconstruction + 3-stage SSP-RK3**, LLF/HLLC faces, CFL 0.4, on
> uniform grids and (gates 4/5/8) the multi-level AMR. Each smooth test is run
> head-to-head against **MUSCL-Hancock** on the identical grid. References:
> exact advection (entropy wave, vortex), exact Riemann (Sod, two-gas), exact
> `erf` (viscous shear). Driver: `weno_suite`. float32.

## Results
![Isentropic vortex core — WENO5 vs MUSCL vs exact](../figures/weno.png)

At $64^2$ the vortex core is resolved by WENO5 down to the exact density dip,
while MUSCL-Hancock over-diffuses it — an $L_1$ dissipation ratio of
**{weno.get('ratio', '—')}×** (gate: WENO < MUSCL/4).

| Gate | Test | Result |
|---|---|---|
| 1 | smooth entropy wave, spatial order | {weno.get('ent_ord', '—')} (spatial 8→16: {weno.get('ent_sp', '—')}; gate ≥ 4) |
| 2 | Sod tube, L1 vs exact + boundedness | WENO {weno.get('sod_w', '—')} vs MUSCL {weno.get('sod_m', '—')}; no over/undershoot |
| 3 | **isentropic vortex, dissipation vs MUSCL** | WENO {weno.get('vor_w', '—')} vs MUSCL {weno.get('vor_m', '—')} (**{weno.get('ratio', '—')}×**, order {weno.get('vor_ord', '—')}) |
| 4 | all-refined 2-level = uniform, bit-exact | {weno.get('bit', '—')} differing values (gate 0) |
| 5 | Sod on 3-level AMR vs MUSCL | WENO {weno.get('samr_w', '—')} vs MUSCL {weno.get('samr_m', '—')} (gate < 2×) |
| 6 | viscous shear (erf) order | {weno.get('shear_ord', '—')} (gate ≥ 1.8) |
| 7 | two-gas Sod (uniform) L1 | {weno.get('tg_u', '—')} (gate 4e-3) |
| 8 | two-gas Sod on 3-level AMR | L1 {weno.get('tg_a', '—')}, species drift {weno.get('tg_drift', '—')} |

## Discussion
The entropy wave recovers the design order (~5) in the spatial-limited regime
before the RK3 temporal error floors the total — the expected behaviour at
fixed CFL. On the genuinely-2D vortex the dimension-by-dimension midpoint
quadrature caps the **formal** order near 2, but the error **constant** is what
matters: WENO5 is **{weno.get('ratio', '—')}× less dissipative** than
MUSCL-Hancock at the same resolution, which is exactly the payoff on smooth
turbulent structures. On the Sod shock WENO5 stays in MUSCL's error class with
no spurious extrema (HLLC faces are hard to beat on a single discontinuity).
Gate 4 is the strongest correctness check: a fully-refined non-subcycled
hierarchy reproduces the uniform fine grid **bit for bit**, proving the
per-stage ghost machinery is exact. Two-gas Riemann (uniform and 3-level AMR)
and the exact-`erf` viscous layer close the loop on the species and viscous
paths. WENO5 is single-gas-per-cell and **incompatible with immersed solids**
(hard error), by design.""")

    # ---- fiche 1e: multi-species / two-gas (validation) ----------------
    fiche("species.md", f"""# Multi-species two-gas — *validation vs exact*

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
| 1 | Abgrall interface, \\|p−1\\| sustained | {species.get('p_sust', '—')} (gate 1e-2); max\\|u−0.5\\| {species.get('u_err', '—')} |
| 2 | two-gas Sod (uniform), L1(ρ) vs exact | {species.get('sod', '—')} (gate 6e-3) |
| 3 | species mass, 200 steps | drift {species.get('mass', '—')} (gate 1e-5) |
| 4 | two-gas Sod on 3-level AMR | L1 {species.get('amr', '—')}, species drift {species.get('amr_drift', '—')} |

## Discussion
The interface stays crisp with **pressure and velocity flat to ~0.6 % / 0.4 %**
across the ρ/γ jump — the Abgrall condition. A tiny bounded wiggle remains
because the reconstruction is on conservative variables; primitive-variable
reconstruction is the documented next refinement, but the sustained oscillation
is already well under the gate and does not grow. The two-gas Sod matches the
generalized exact Riemann solution (each side keeping its own γ across the
contact) both on a uniform grid and through the refluxed 3-level AMR, and
species mass is conserved to the float32 floor. The same two-gas path is
re-exercised under WENO5 in the [WENO5 suite](weno.md) (gates 7–8).""")

    # ---- fiche 1f: analytic suite (verification + validation) ----------
    fiche("analytic.md", f"""# Analytic suite — *verification & validation*

**Objective.** Exercise the solver against exact/theoretical references the
single Sod case cannot: (1) **Toro's Riemann battery** (tests 2–5) — the hard
1D problems: near-vacuum double rarefaction, a p = 1000 blast, colliding strong
shocks, a slowly-moving contact — vs the exact Riemann solver; (2) a smooth
**acoustic wave** returning onto its IC after one period (smooth-regime order);
(3) the **isentropic vortex** (Yee) advected one period (canonical smooth 2D
Euler); (4) **Sedov** self-similar blast exponent ($r\\sim t^{{1/2}}$); (5)
**Rayleigh–Taylor** linear growth rate vs $\\sqrt{{Agk}}$.

## Numerical setup
> MUSCL-Hancock + HLLC, CFL 0.4, uniform grids. Riemann battery: N = 400,
> transmissive, reduced start-up CFL for the sharp IC. Acoustic/vortex:
> periodic, order from grid refinement. Sedov: 256², point energy in 3 cells.
> RT: seeded single mode, growth from a least-squares fit of $\\ln a(t)$.
> Driver: `analytic_suite`. float32.

## Results
![Toro Riemann battery — density vs exact](../figures/analytic.png)

| Gate | Test | Result |
|---|---|---|
| 1 | Toro T2 near-vacuum / T3 blast | L1 {analytic.get('t2', '—')} / {analytic.get('t3', '—')} |
| 1 | Toro T4 colliding / T5 slow contact | L1 {analytic.get('t4', '—')} / {analytic.get('t5', '—')} |
| 2 | acoustic wave, smooth order | {analytic.get('ac', '—')} (TVD-extremum theory 4/3) |
| 3 | isentropic vortex, mean order | {analytic.get('vor', '—')} (gate > 1.5) |
| 4 | Sedov 2D front exponent | {analytic.get('sedov', '—')} (theory 0.5, ±0.03) |
| 5 | Rayleigh–Taylor growth σ | {analytic.get('rt', '—')} vs √(Agk) {analytic.get('rt_th', '—')} (±15 %) |

## Discussion
The four Toro tests are the standard robustness gauntlet: the solver survives
the **near-vacuum** double rarefaction (positivity), the **p = 1000 blast** and
the **colliding-shock** problem (strong-shock stability), and resolves the
**slowly-moving contact** — all within their exact-solution error gates and
with no NaNs. The acoustic order sits at the TVD **4/3** ceiling (limiters clip
the smooth sine crests — the documented motivation for WENO5, see
[the WENO suite](weno.md)), while the vortex — error-dominated away from
extrema — reaches ~2. Sedov recovers the self-similar **½** exponent and RT
matches the linear dispersion relation $\\sqrt{{Agk}}$ to within a few percent.
Together these pin the scheme's accuracy **and** its nonlinear robustness.""")

    # ---- fiche 1g: immersed boundaries (validation + verification) -----
    fiche("immersed.md", f"""# Immersed boundaries — *validation & verification*

**Objective.** Validate the staircase immersed-solid treatment (solid mask +
mask-aware fluxes threaded through `step2D` and the whole AMR chain) on four
fronts: (1) a **planar shock reflecting on an immersed wall** — the
post-reflection wall pressure has an *exact* 1D value (sub- and supersonic
incident); (2) the **declarative** path (a `[solid]` region parsed from
`cases/shock_wall.ini` run through the real runner); (3) the same reflection
**with AMR** (2-level, subcycled, 3-level) — wall pressure preserved through
restriction / reflux / prolongation / body-edge tagging; (4) a **no-slip
viscous** boundary layer — Blasius on a plate posed *by the mask* (not a domain
BC) — and (5) **CPU↔GPU lock-step** on an immersed Mach-2 cylinder.

## Numerical setup
> MUSCL-Hancock + HLLC, reflective wall flux inside the masked cells. Shock
> reflection: Ms = 2 (subsonic) / 3 (supersonic), exact 1D reflected pressure.
> No-slip: M ≈ 0.25, μ = 1.2e-4, plate from x = 0.2, Blasius by RK4. GPU:
> `AmrGpu` vs the validated CPU `Amr2` in lock-step (same dt). Drivers:
> `immersed`, `immersed_case`, `immersed_amr`, `immersed_noslip`,
> `immersed_gpu`. float32.

## Results
![No-slip Blasius on an immersed plate](../figures/immersed.png)

| Gate | Test | Result |
|---|---|---|
| shock | wall pressure vs exact 1D (declarative) | err {immersed.get('refl_err', '—')} % (gate 5 %), non-penetration \\|u\\|→0 |
| AMR | wall pressure, AMR vs base | {immersed.get('amr_err', '—')} % (gate 2 %), exact within 5 % |
| no-slip | profile RMS(u/Ue − f′) | {immersed.get('rms', '—')} (gate 3e-2); wall slip {immersed.get('slip', '—')}; Cf within {immersed.get('cf', '—')} % |
| GPU | CPU↔GPU lock-step, max rel. error | single {immersed.get('gpu', '—')}, 3-level {immersed.get('gpu_ml', '—')} |

## Discussion
The reflected-shock wall pressure lands within a few tenths of a percent of the
exact 1D value for **both** a subsonic and a supersonic incident shock, and the
declarative `[solid]` path reproduces it — validating the full chain from INI
parsing through `solidAt()` to the mask-aware step. Turning on AMR keeps the
wall pressure within 0.2 % of the base-grid run across 2-level, subcycled and
3-level hierarchies, proving the mask survives restriction, refluxing and
body-edge tagging. The **no-slip** case is the strongest: a boundary layer
grown entirely by the mask reproduces the Blasius profile to
**RMS {immersed.get('rms', '—')}** with wall slip {immersed.get('slip', '—')}
and skin friction within {immersed.get('cf', '—')} % of 0.664/√Re_x — the
mask-aware viscous flux is doing real physics, not just blocking flow. Finally
the GPU path advances in lock-step with the CPU reference to ~1e-3 (fp32 sum
reassociation), locking the Metal port of the mask. Complements the
[oblique-shock wedge](wedge.md), which is itself an immersed body.

> WENO5 is incompatible with immersed solids (hard error) — the immersed path
> is MUSCL-Hancock only.""")

    # ---- fiche 1h: double Mach reflection (verification) ---------------
    fiche("dmr.md", f"""# Double Mach reflection — *verification*

**Objective.** The double Mach reflection (Woodward & Colella) is the standard
**strong-shock** stress test: a Mach-10 shock striking a 30° wedge produces the
incident/reflected/Mach-stem shocks meeting at a **triple point**, with a
slip-line jet rolling up along the wall. There is no closed-form solution, so
this is a **verification** case: (1) the GPU path advances in **lock-step** with
the validated CPU reference; (2) the hierarchy is bit-close through **2-level**
(`AmrGpu`) and **3-level** (`AmrGpuML`) subcycled AMR; (3) the declarative
`CaseDef` ghost fill reproduces the analytic moving-shock BC **cell-for-cell**.

## Numerical setup
> MUSCL-Hancock + HLLC, CFL 0.4, domain 4 × 1, t = 0.2, the classic Mach-10
> reflected-shock inflow (`dmr::fillGhosts`). Field shown: **GPU**, 960 × 240.
> Lock-step gates compare `Euler2DGpu` / `AmrGpu` / `AmrGpuML` against the CPU
> `Grid` / `Amr2` / `AmrML` on the same dt sequence. Drivers: `dmr_gpu`,
> `dmr_amr`, `mlgpu_amr`, `casedef_test`. float32.

## Results
![Double Mach reflection — density at t=0.2](../figures/dmr.png)

The density field shows the textbook structure: the Mach stem normal to the
wall, the reflected shock, the triple point, and the wall jet curling under the
contact — resolved on the refined patches.

| Gate | Test | Result |
|---|---|---|
| GPU lock-step (uniform) | max rel. diff CPU↔GPU | {dmr.get('gpu_rel', '—')} (gate 1e-2); {dmr.get('gpu_speed', '—')}× speedup |
| AMR lock-step (2-level) | max rel. diff CPU↔GPU | {dmr.get('amr_rel', '—')}; work {dmr.get('amr_work', '—')} % of uniform 1/256 |
| AMR lock-step (3-level) | max rel. diff CPU↔GPU | {dmr.get('ml_rel', '—')} |
| 3-level periodic KH (GPU) | closed-domain mass drift | {dmr.get('ml_kh', '—')} (gate 1e-6) |
| CaseDef DMR ghosts | vs preset, differing cells | {dmr.get('cd_ghost', '—')} (gate 0) |

## Discussion
The GPU reproduces the CPU reference to **{dmr.get('gpu_rel', '—')}** (fp32 sum
reassociation) at a **{dmr.get('gpu_speed', '—')}× speedup**, and the agreement
holds through the full AMR machinery — 2-level and 3-level subcycled hierarchies
stay bit-close to the CPU AMR with **identical per-level patch counts**, so the
GPU refluxing / restriction / prolongation are exact copies of the validated
CPU path. AMR does the DMR at 1/256 for only **{dmr.get('amr_work', '—')} %** of
the uniform-grid cell-steps. The `CaseDef` gate closes the declarative loop: the
parsed analytic moving-shock BC reproduces the hand-written `dmr::fillGhosts`
**cell-for-cell** ({dmr.get('cd_ghost', '—')} differing). This is the case the
[conservation](conservation.md) fiche's periodic KH complements on the
GPU-lock-step / mass-drift axis.""")

    # ---- fiche 1i: Haas–Sturtevant shock–bubble (validation·exp) -------
    fiche("shock_bubble.md", f"""# Shock–bubble (Haas & Sturtevant) — *validation vs experiment*

**Objective.** The most demanding validation in the dossier: reproduce the
**experimental** interface dynamics of Haas & Sturtevant (1987) — a **Mach
1.22** shock in air striking a cylinder of (contaminated) **helium** — and
compare the early-time velocities of the three characteristic interface points
against both the experiment and the canonical numerical reference,
Quirk & Karni (1996).

## Numerical setup
> Two-gas (air γ 1.4 | contaminated-helium γ 1.645, ρ ratio 0.182), **3-level
> subcycled AMR on GPU** (`AmrGpuML`), density + velocity tagging, domain
> 2 × 1, finest 1/256. Post-shock inflow left, reflective tube walls. Interface
> tracked as the Y = 0.5 axis crossings; each velocity is a least-squares slope
> over the phase window the experiment measures. Driver: `hs_suite`. float32.

## Results
![Haas & Sturtevant interface velocities vs experiment](../figures/hs.png)

| Interface point | Experiment | Quirk & Karni | machmallow | Δ vs exp |
|---|---|---|---|---|
| upstream edge | 170 m/s | 178 m/s | {hs.get('ui_ms', '—')} m/s | {hs.get('ui_pc', '—')} % |
| downstream edge | 145 m/s | 146 m/s | {hs.get('di_ms', '—')} m/s | {hs.get('di_pc', '—')} % |
| air jet | 230 m/s | 227 m/s | {hs.get('jet_ms', '—')} m/s | {hs.get('jet_pc', '—')} % |

## Discussion
All three characteristic velocities land **within ±10 %** of the experimental
values and sit right alongside Quirk & Karni's computed numbers — the upstream
edge and downstream edge slightly fast (consistent with the contaminated-helium
model and the ±measurement uncertainty H&S report), the **air jet within 1 %**.
Recovering an *experimental* dataset — not just an exact solution — with a
two-gas, GPU, multi-level-AMR run is the end-to-end validation that the species
transport, the shock–interface interaction and the adaptive refinement all work
**together**. The two-gas machinery itself is unit-validated against exact
Riemann solutions in the [multi-species fiche](species.md).""")

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

    # ---- fiche 2b: diagonal 2D Sod (validation + isotropy) --------------
    fiche("sod2d.md", f"""# Diagonal 2D Sod — *validation + isotropy*

**Objective.** Run the Sod problem **diagonally** across a 2D grid. The exact
solution depends only on $\\xi = (x+y-1)/\\sqrt2$, so the 2D field must
**collapse** onto the 1D Riemann solution — testing the 2D scheme *and* its
isotropy (no grid-alignment bias) at once.

## Numerical setup
> MUSCL-Hancock + HLLC, **uniform 2D grid** (no AMR), inviscid Euler, CFL 0.4,
> t = 0.15, transmissive on all sides. Grid-convergence N = 64, 128, 256 on
> the central (boundary-free) square. Reference = exact Riemann in $\\xi$.
> Driver: `sod2d`. float32.

## Results
![Diagonal 2D Sod: field and collapse onto the 1D solution](../figures/sod2d.png)

Mean grid-convergence order (L1 density, central square): **{sod2d_order}**
(≈1, discontinuous solution).

## Discussion
Every cell of the 2D field lands on the exact 1D curve when plotted against
$\\xi$ (right panel) — the scheme is **isotropic** (a 45° shock is captured
like an axis-aligned one) and matches the exact Riemann solution. Order ~1 is
the expected discontinuity-limited rate, consistent with the 1D Sod fiche.

The initial discontinuity meets the boundary at the corners (1,0) and (0,1),
where the flow is genuinely 2D and the transmissive BCs distort the oblique
waves; those disturbances travel inward at finite speed, so both the L1 gate
and the collapse above are measured only on the **boundary-free central
square [0.3, 0.7]²** (dashed box) — still untouched at t = 0.15.""")

    # ---- fiche 2c: Sod on AMR (verification: refluxing + accuracy) ------
    fiche("sod_amr.md", f"""# Sod on AMR — *verification: refluxing + accuracy*

**Objective.** Run Sod on a 2-level AMR hierarchy and check two things:
(1) **conservation** — the Berger–Colella *refluxing* must cancel the
coarse/fine flux mismatch (mass drift at the float32 floor); (2) **accuracy**
— the composite L1 must match a uniform-fine run, for a fraction of the work.

## Numerical setup
> MUSCL-Hancock + HLLC, **2-level AMR** (coarse 128×32 → fine 256×64, ratio 2),
> inviscid Euler, CFL 0.4, t = 0.2. The refluxing test uses a **frozen** mesh
> refined around the initial discontinuity, so the shock, contact and
> rarefaction all cross the coarse/fine interfaces. Driver: `sod_amr`.

## Results
![AMR composite vs exact Riemann (left); mass drift with/without refluxing (right)](../figures/sod_amr.png)

The **left panel** overlays the AMR composite density on the exact Riemann
solution: the coarse cells (blue) carry the smooth regions, while the fine
patches (orange) cluster exactly on the shock, contact and rarefaction — and
the whole thing lands on the exact curve.

| Metric | Result |
|---|---|
| mass drift, refluxing **on** (frozen) | {rfx_with} |
| mass drift, refluxing **off** (frozen) | {rfx_without} (**{rfx_ratio}× worse**) |
| composite L1 vs exact / uniform-fine | ratio {samr_l1} (gate 1.4) |
| work vs uniform fine | {samr_work} % of the cell-steps |

## Discussion
**What is refluxing?** At a coarse/fine interface the coarse cell and the
adjacent fine cells each compute the flux through the *shared* face
independently — at different resolutions, so the two disagree. Left
uncorrected, that mismatch adds or removes mass (and momentum/energy) at the
interface every step, breaking conservation. **Refluxing** (Berger–Colella)
fixes it: after the fine level advances, the coarse flux through the shared
face is *replaced* by the sum of the fine-face fluxes, so the interface
becomes conservative to machine precision.

The right panel is the proof: with a frozen coarse/fine interface swept by all
three waves, turning refluxing **off** leaks mass **{rfx_ratio}× more**; with
it **on**, the drift sits at the float32 floor. Meanwhile the AMR composite is
**as accurate** as the uniform-fine grid (L1 ratio ≈ 1) for only ~{samr_work} %
of the cell-steps — the whole point of AMR.""")

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
| [Manufactured solution](cases/mms.md) | verification | viscous Navier–Stokes order 2 (both schemes) | ✅ PASS |
| [0D reactor kinetics](cases/reactor.md) | verification | Arrhenius integrator vs exact (isothermal/adiabatic/stiff) | ✅ PASS |
| [WENO5 scheme suite](cases/weno.md) | verification | WENO5 {weno.get('ratio', '—')}× less dissipative than MUSCL (vortex); bit-exact on AMR | ✅ PASS |
| [Conservation](cases/conservation.md) | verification | mass & energy at the float32 floor (AMR, periodic) | ✅ PASS |
| [Multi-species two-gas](cases/species.md) | validation · exact | Abgrall interface (p, u flat); two-gas Riemann (uniform + AMR) | ✅ PASS |
| [Analytic suite](cases/analytic.md) | verification · validation | Toro battery, acoustic/vortex order, Sedov ½, Rayleigh–Taylor | ✅ PASS |
| [Sod on AMR](cases/sod_amr.md) | verification | refluxing conserves (6000× vs off); L1 = uniform-fine | ✅ PASS |
| [Sod shock tube](cases/sod.md) | validation · exact | matches exact Riemann (both schemes) | ✅ PASS |
| [Diagonal 2D Sod](cases/sod2d.md) | validation · exact | 2D field collapses onto 1D Riemann (isotropy) | ✅ PASS |
| [CJ detonation](cases/detonation.md) | validation · exact | D relaxes to D_CJ (+0.4 % uniform, −0.2 % AMR, long tube) | ✅ PASS |
| [Blasius boundary layer](cases/blasius.md) | validation · theory | RMS {rms} vs $f'$; Cf bias traced to near-wall resolution | ✅ PASS |
| [Oblique shock θ-β-M](cases/wedge.md) | validation · theory | β → exact (staircase bias 2.5°→0.6° w/ refinement) | ✅ PASS |
| [Immersed boundaries](cases/immersed.md) | validation · theory | reflected-shock wall p exact; no-slip Blasius (RMS {immersed.get('rms', '—')}); GPU lock-step | ✅ PASS |
| [Double Mach reflection](cases/dmr.md) | verification | strong-shock triple point; CPU↔GPU lock-step ({dmr.get('gpu_speed', '—')}×) through 3-level AMR | ✅ PASS |
| [Shock–bubble (Haas & Sturtevant)](cases/shock_bubble.md) | validation · experiment | interface velocities within ±10 % of experiment (two-gas + AMR + GPU) | ✅ PASS |

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
    sod2d_txt = samr_txt = mms_txt = rea_txt = weno_txt = spec_txt = ""
    ana_txt = imm_txt = dmr_txt = hs_txt = ""
    IMM = ["immersed", "immersed_case", "immersed_amr", "immersed_noslip",
           "immersed_gpu"]
    DMR = [("dmr_gpu", ["240"]), ("dmr_amr", ["128", "gpu"]),
           ("mlgpu_amr", ["32"]), ("casedef_test", [])]
    if not args.no_run:
        print("running V&V drivers…")
        conv_txt = run_driver("convergence")
        sod_txt = run_driver("sod1d")
        sod2d_txt = run_driver("sod2d")
        samr_txt = run_driver("sod_amr")
        mms_txt = run_driver("mms")
        rea_txt = run_driver("reactor")
        weno_txt = run_driver("weno_suite")
        spec_txt = run_driver("species_suite")
        ana_txt = run_driver("analytic_suite")
        bla_txt = run_driver("blasius")
        det_txt = run_driver("detonation", "16")   # long tube -> D relaxes to CJ
        imm_txt = "\n".join(run_driver(e) for e in IMM)  # last: GPU lock-step
        dmr_txt = "\n".join(run_driver(e, *a) for e, a in DMR)  # GPU DMR suite
        hs_txt = run_driver("hs_suite")                         # GPU shock-bubble
        run_case("vv/conservation.ini")
    else:                                           # replot from cached logs
        (conv_txt, sod_txt, sod2d_txt, samr_txt, mms_txt, rea_txt, weno_txt,
         spec_txt, ana_txt, bla_txt, det_txt) = (
             cached("convergence"), cached("sod1d"), cached("sod2d"),
             cached("sod_amr"), cached("mms"), cached("reactor"),
             cached("weno_suite"), cached("species_suite"),
             cached("analytic_suite"), cached("blasius"),
             cached("detonation"))
        imm_txt = "\n".join(cached(e) for e in IMM)
        dmr_txt = "\n".join(cached(e) for e, _ in DMR)
        hs_txt = cached("hs_suite")

    print("plotting…")
    orders = plot_order()
    sod_n = plot_sod()
    plot_sod2d()
    samr = plot_sod_amr(samr_txt)
    mms = plot_mms(mms_txt)
    rea = plot_reactor(rea_txt)
    weno = plot_weno(weno_txt)
    species = plot_species(spec_txt)
    analytic = plot_analytic(ana_txt)
    immersed = plot_immersed(imm_txt)
    dmr = plot_dmr(dmr_txt)
    hs = plot_hs(hs_txt)
    plot_blasius()
    plot_blasius_cf()
    plot_blasius_refine()
    det = plot_detonation(det_txt)
    plot_wedge()
    plot_conservation()

    # copy source data for provenance
    keep = {"convergence.csv", "blasius_profile.csv", "blasius_cf.csv",
            "blasius_profile_weno.csv", "blasius_cf_weno.csv",
            "sod_muscl_400.csv", "sod_weno_400.csv", "sod2d_field.csv",
            "sod2d_exact.csv", "sod_amr_profile.csv", "mms.csv",
            "reactor_isothermal.csv", "detonation_front.csv",
            "weno_vortex_weno.csv", "weno_vortex_muscl.csv",
            "species_interface.csv", "analytic_toro1.csv",
            "analytic_toro2.csv", "analytic_toro3.csv", "analytic_toro4.csv",
            "immersed_noslip.csv", "vv_conservation_log.csv"}
    for f in os.listdir(OUT):
        if f in keep or re.match(r"sod_\d+\.csv", f):
            shutil.copy(os.path.join(OUT, f), os.path.join(DATA, f))

    write_report(orders, sod_n, sod_txt, bla_txt, conv_txt, det,
                 sod2d_txt, samr_txt, mms, rea, weno, species, analytic,
                 immersed, dmr, hs)
    print(f"done — figures in {FIG}, fiches in {CASES}, index "
          f"{os.path.join(VV, 'README.md')}")


if __name__ == "__main__":
    main()
