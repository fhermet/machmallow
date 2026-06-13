#!/usr/bin/env python3
"""Plot the convergence curves written by the `convergence` driver.

Reads out/convergence.csv (problem, scheme, N, h, L1) and draws one
log-log panel per problem with both schemes, reference-slope guides, and
the least-squares order fitted over the points that are still in the
discretization regime (before the float32 roundoff floor flattens the
curve). Saves out/convergence.png.

    ./build/convergence            # produce the CSV
    python3 tools/plot_convergence.py
"""
import csv
import math
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: write a file, never open a window
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

CSV = "out/convergence.csv"
PNG = "out/convergence.png"

# Human titles and the reference slopes worth drawing on each panel.
PANELS = {
    "entropy_wave": ("Onde d'entropie lisse (1D alignee)", [2, 5]),
    "isentropic_vortex": ("Vortex isentropique (2D lisse)", [2, 3]),
    "sod": ("Tube de Sod (avec discontinuites)", [1, 2]),
}
COLORS = {"muscl": "#c1272d", "weno5": "#0050a0"}
LABELS = {"muscl": "MUSCL-Hancock + HLLC", "weno5": "WENO5 + HLLC + RK3"}


def load(path):
    data = {}  # (problem, scheme) -> list of (h, L1) sorted by 1/h
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["problem"], row["scheme"])
            data.setdefault(key, []).append(
                (float(row["h"]), float(row["L1"])))
    for v in data.values():
        v.sort(key=lambda hL: -hL[0])  # coarse -> fine
    return data


def fit_preflood(hs, ls):
    """LSQ slope over points still decreasing by a clear factor (the
    rest is float32-roundoff-floored: error stops dropping or rises)."""
    keep = [0]
    for k in range(1, len(ls)):
        if ls[k] < 0.7 * ls[k - 1]:
            keep.append(k)
        else:
            break  # floor reached; later points only get worse
    if len(keep) < 2:
        return None, keep
    lx = [math.log(hs[k]) for k in keep]
    ly = [math.log(ls[k]) for k in keep]
    n = len(lx)
    sx, sy = sum(lx), sum(ly)
    sxx = sum(x * x for x in lx)
    sxy = sum(x * y for x, y in zip(lx, ly))
    return (n * sxy - sx * sy) / (n * sxx - sx * sx), keep


def main():
    if not os.path.exists(CSV):
        sys.exit(f"{CSV} not found — run ./build/convergence first")
    data = load(CSV)
    problems = [p for p in PANELS if any(k[0] == p for k in data)]
    fig, axes = plt.subplots(1, len(problems),
                             figsize=(5.2 * len(problems), 4.6))
    if len(problems) == 1:
        axes = [axes]

    for ax, prob in zip(axes, problems):
        title, refs = PANELS[prob]
        for scheme in ("muscl", "weno5"):
            pts = data.get((prob, scheme))
            if not pts:
                continue
            hs = [h for h, _ in pts]
            ls = [L for _, L in pts]
            slope, keep = fit_preflood(hs, ls)
            lbl = LABELS[scheme]
            if slope is not None:
                lbl += f"  (ordre {slope:.2f})"
            ax.loglog(hs, ls, "o-", color=COLORS[scheme], label=lbl,
                      zorder=3)
            # open markers on the float32-floored tail
            fl = [k for k in range(len(hs)) if k not in keep]
            if fl:
                ax.loglog([hs[k] for k in fl], [ls[k] for k in fl],
                          "o", mfc="white", color=COLORS[scheme],
                          zorder=4)
        # reference slope guides anchored at the coarsest MUSCL point
        anchor = data.get((prob, "muscl")) or next(iter(data.values()))
        h0, L0 = anchor[0]
        for p in refs:
            hh = [anchor[0][0], anchor[-1][0]]
            yy = [L0 * (h / h0) ** p for h in hh]
            ax.loglog(hh, yy, "--", color="0.6", lw=0.9, zorder=1)
            ax.text(hh[1], yy[1], f" ordre {p}", color="0.5",
                    fontsize=8, va="center")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("h (taille de maille)")
        ax.set_ylabel("erreur L1 sur rho")
        ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("machmallow — ordre de convergence des schemas "
                 "(float32)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PNG, dpi=130)
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
