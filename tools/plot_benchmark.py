#!/usr/bin/env python3
"""Plot the GPU/hybrid vs pure-CPU benchmark (out/benchmark.csv).

Left panel : throughput (Mcell-steps/s) vs total cells, CPU and hybrid
             for each case (uniform box, 3-level DMR).
Right panel: speedup (hybrid / CPU) vs total cells, with the break-even
             line at 1 — below it the GPU loses to launch/sync overhead.

    ./build/benchmark
    python3 tools/plot_benchmark.py

NB: the CPU path is single-threaded, so this is GPU vs ONE core.
"""
import csv
import os
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

CSV, PNG = "out/benchmark.csv", "out/benchmark.png"
CASE_LBL = {"uniform": "uniforme (1 niveau)", "dmr": "DMR (AMR 3 niveaux)"}
CASE_COL = {"uniform": "#0050a0", "dmr": "#c1272d"}


def main():
    if not os.path.exists(CSV):
        sys.exit(f"{CSV} not found — run ./build/benchmark first")
    # (case, backend) -> list of (cells, mcellps)
    d = {}
    for row in csv.DictReader(open(CSV)):
        d.setdefault((row["case"], row["backend"]), []).append(
            (int(row["cells"]), float(row["mcell_per_s"])))
    for v in d.values():
        v.sort()

    cases = [c for c in CASE_LBL if (c, "cpu") in d]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    for case in cases:
        col = CASE_COL[case]
        cpu = d[(case, "cpu")]
        hyb = d[(case, "hybrid")]
        cx, cy = zip(*cpu)
        hx, hy = zip(*hyb)
        ax1.loglog(cx, cy, "o--", color=col, mfc="white",
                   label=f"{CASE_LBL[case]} — CPU (1 cœur)")
        ax1.loglog(hx, hy, "o-", color=col,
                   label=f"{CASE_LBL[case]} — hybride GPU")
        # speedup at matching cell counts
        cpud = dict(cpu)
        sx = [c for c, _ in hyb if c in cpud]
        sy = [dict(hyb)[c] / cpud[c] for c in sx]
        ax2.semilogx(sx, sy, "o-", color=col, label=CASE_LBL[case])

    ax1.set_xlabel("cellules totales")
    ax1.set_ylabel("débit (Mcell-steps/s)")
    ax1.set_title("Débit CPU vs hybride")
    ax1.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax1.legend(fontsize=8, loc="upper left")

    ax2.axhline(1.0, color="0.5", ls="--", lw=1)
    ax2.text(ax2.get_xlim()[1], 1.0, " seuil (x1)", color="0.5",
             fontsize=8, va="bottom", ha="right")
    ax2.set_xlabel("cellules totales")
    ax2.set_ylabel("speedup  (hybride / CPU)")
    ax2.set_title("Apport du GPU vs taille de problème")
    ax2.grid(True, which="both", ls=":", lw=0.5, alpha=0.6)
    ax2.legend(fontsize=8, loc="upper left")

    fig.suptitle("machmallow — apport du GPU/hybride (M4, CPU 1 cœur)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(PNG, dpi=130)
    print(f"wrote {PNG}")


if __name__ == "__main__":
    main()
