#!/usr/bin/env python3
"""Monte une video PEDAGOGIQUE de la tuyere C-D a partir du split
vorticite/schlieren (nozzle_hero_split.mp4).

L'ecoulement se DEROULE puis S'ARRETE sur chaque regime caracteristique
avec un panneau de commentaire (nom du regime, NPR, physique). Carte de
titre en intro + labels permanents (haut = vorticite, bas = schlieren).

  python3 tools/nozzle_pedago.py --video nozzle_hero_split.mp4 \
      --out nozzle_pedago.mp4

Conception : matplotlib ne dessine QUE les surcouches statiques (intro,
labels, 1 image legendee figee par regime) ; les frames jouees sont
composees au numpy (rapide). Les repetitions (figes, intro) sont des
liens durs (os.link) -> pas de duplication disque."""
import argparse
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image

DPI = 200

# Regimes : (frame source 1-based la PLUS etablie du palier, titre, NPR,
# lignes de commentaire). Les frames suivent le schedule de nozzle.ini a
# 25 fps (palier choc droit ~45, surdetendu ~80, adapte ~110,
# sousdetendu ~142, disque de Mach ~180).
REGIMES = [
    (45, "Choc droit dans le divergent", "1,6",
     ["Le col est amorcé (M = 1) : débit figé.",
      "L'écoulement devient supersonique dans le divergent puis",
      "repasse subsonique à travers un CHOC DROIT, qui décolle",
      "la couche limite des parois."]),
    (80, "Surdétendu — chocs obliques", "3,5",
     ["La pression de sortie reste SUPÉRIEURE à l'ambiante :",
      "le jet est surdétendu. Des chocs obliques se forment en",
      "sortie pour le recomprimer, avec décollement et",
      "recirculation au bord de fuite."]),
    (110, "Adapté (régime de design)", "13",
     ["Pression de sortie = pression ambiante : le jet sort",
      "sans choc ni détente nette. Les cellules de choc sont",
      "à peine marquées — c'est le point de fonctionnement",
      "nominal de la tuyère."]),
    (142, "Sous-détendu — détentes", "22",
     ["Pression de sortie INFÉRIEURE à l'ambiante : le jet se",
      "détend en sortie (éventails de Prandtl-Meyer) et",
      "s'élargit. Les cellules de choc en losange (diamants)",
      "se succèdent le long du jet."]),
    (180, "Fortement sous-détendu — disque de Mach", "46",
     ["Forte sous-détente : un TONNEAU de choc se referme par",
      "un DISQUE DE MACH (choc droit dans le jet libre).",
      "Les nappes de cisaillement sont très écartées ;",
      "l'après-disque redevient subsonique."]),
]

TITLE = "Tuyère convergente-divergente — transitoire entre régimes"
SUBTITLE = "Balayage de la contre-pression   ·   NPR = p₀ / pₐ"
LEGEND = "Haut : vorticité (icefire)      Bas : schlieren ( |∇ρ| )"


def fig_for(W, H):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax


def save_exact(fig, out, transparent=False):
    fig.savefig(out, dpi=DPI, transparent=transparent)
    plt.close(fig)


def make_intro(W, H, out):
    s = H / 640.0
    fig, ax = fig_for(W, H)
    ax.add_patch(Rectangle((0, 0), 1, 1, color="black"))
    halo = [pe.withStroke(linewidth=2 * s, foreground="#000")]
    ax.text(0.5, 0.60, TITLE, ha="center", va="center", fontsize=13 * s,
            color="white", fontweight="bold", path_effects=halo)
    ax.text(0.5, 0.48, SUBTITLE, ha="center", va="center", fontsize=8.5 * s,
            color="#9fd3ff")
    ax.text(0.5, 0.34, LEGEND, ha="center", va="center", fontsize=7 * s,
            color="#c9c9c9")
    ax.text(0.5, 0.06, "machmallow · solveur CFD compressible GPU",
            ha="center", va="center", fontsize=5.5 * s, color="#6f6f6f")
    save_exact(fig, out)


def make_labels(W, H, out):
    """Surcouche RGBA transparente : labels de modalite permanents."""
    s = H / 640.0
    fig, ax = fig_for(W, H)
    halo = [pe.withStroke(linewidth=2.4 * s, foreground="black")]
    ax.text(0.012, 0.965, "VORTICITÉ", ha="left", va="top",
            fontsize=6.2 * s, color="white", fontweight="bold",
            path_effects=halo, alpha=0.92)
    ax.text(0.012, 0.035, "SCHLIEREN", ha="left", va="bottom",
            fontsize=6.2 * s, color="white", fontweight="bold",
            path_effects=halo, alpha=0.92)
    ax.text(0.988, 0.035, "machmallow", ha="right", va="bottom",
            fontsize=5.2 * s, color="white", fontweight="bold",
            path_effects=halo, alpha=0.62)
    save_exact(fig, out, transparent=True)


def make_caption(src_rgb, title, npr, lines, out):
    """Image legendee : frame source + bandeau bas + commentaire."""
    H, W = src_rgb.shape[:2]
    s = H / 640.0
    fig, ax = fig_for(W, H)
    ax.imshow(src_rgb, extent=(0, 1, 0, 1), aspect="auto", zorder=0)
    # bandeau bas semi-transparent (lower-third)
    bh = 0.345
    ax.add_patch(Rectangle((0, 0), 1, bh, color="black", alpha=0.72,
                           zorder=2))
    ax.add_patch(Rectangle((0, bh), 1, 0.006, color="#9fd3ff", alpha=0.85,
                           zorder=2))
    halo = [pe.withStroke(linewidth=1.8 * s, foreground="black")]
    ax.text(0.022, bh - 0.045, title, ha="left", va="top",
            fontsize=10.5 * s, color="white", fontweight="bold",
            path_effects=halo, zorder=3)
    ax.text(0.978, bh - 0.05, f"NPR ≈ {npr}", ha="right", va="top",
            fontsize=10 * s, color="#9fd3ff", fontweight="bold", zorder=3)
    y = bh - 0.155
    for ln in lines:
        ax.text(0.022, y, ln, ha="left", va="top", fontsize=6.4 * s,
                color="#e8e8e8", zorder=3)
        y -= 0.046
    save_exact(fig, out)


def composite(src_path, overlay_rgba, out_path):
    src = np.asarray(Image.open(src_path).convert("RGB"), np.float32)
    a = overlay_rgba[..., 3:4] / 255.0
    out = src * (1 - a) + overlay_rgba[..., :3] * a
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="nozzle_hero_split.mp4")
    ap.add_argument("--out", default="nozzle_pedago.mp4")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--intro", type=float, default=3.0, help="duree intro (s)")
    ap.add_argument("--pause", type=float, default=3.2, help="arret/regime (s)")
    ap.add_argument("--pause-last", type=float, default=4.0)
    ap.add_argument("--tail", type=float, default=1.2, help="fin apres dernier")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="pedago_")
    src = os.path.join(work, "src")
    seq = os.path.join(work, "seq")
    os.makedirs(src)
    os.makedirs(seq)

    # 1) extraire les frames du split
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i",
                    args.video, os.path.join(src, "s_%04d.png")], check=True)
    files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
    n = len(files)
    H, W = np.asarray(Image.open(os.path.join(src, files[0]))).shape[:2]
    print(f"source : {n} frames {W}x{H}")

    # 2) surcouches statiques
    intro_png = os.path.join(work, "intro.png")
    labels_png = os.path.join(work, "labels.png")
    make_intro(W, H, intro_png)
    make_labels(W, H, labels_png)
    labels_rgba = np.asarray(Image.open(labels_png).convert("RGBA"), np.float32)

    cap_png = []
    for k, (idx, title, npr, lines) in enumerate(REGIMES):
        f = os.path.join(src, f"s_{idx:04d}.png")
        srgb = np.asarray(Image.open(f).convert("RGB"))
        # poser d'abord les labels permanents sous le bandeau
        a = labels_rgba[..., 3:4] / 255.0
        srgb = (srgb * (1 - a) + labels_rgba[..., :3] * a).astype(np.uint8)
        out = os.path.join(work, f"cap_{k}.png")
        make_caption(srgb, title, npr, lines, out)
        cap_png.append(out)
    print(f"surcouches : intro + labels + {len(cap_png)} legendes")

    # 3) assembler la sequence de sortie (liens durs pour les repetitions)
    oi = 0

    def emit_file(path, count):
        nonlocal oi
        for _ in range(count):
            dst = os.path.join(seq, f"o_{oi:05d}.png")
            os.link(path, dst)
            oi += 1

    def emit_play(lo, hi):           # frames source [lo,hi] composees + labels
        nonlocal oi
        for i in range(lo, hi + 1):
            dst = os.path.join(seq, f"o_{oi:05d}.png")
            composite(os.path.join(src, f"s_{i:04d}.png"), labels_rgba, dst)
            oi += 1

    emit_file(intro_png, int(round(args.intro * args.fps)))
    prev = 1
    for k, (idx, *_rest) in enumerate(REGIMES):
        emit_play(prev, idx)         # derouler jusqu'au palier
        dur = args.pause_last if k == len(REGIMES) - 1 else args.pause
        emit_file(cap_png[k], int(round(dur * args.fps)))   # ARRET commente
        prev = idx + 1
    if prev <= n:
        emit_play(prev, n)
    emit_play(n, n)                  # garde la derniere
    emit_file(os.path.join(src, f"s_{n:04d}.png"), int(round(args.tail * args.fps)))

    # 4) encoder
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
           "-i", os.path.join(seq, "o_%05d.png"), "-c:v", "libx264",
           "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(args.fps), args.out]
    subprocess.run(cmd, check=True)
    print(f"video : {args.out}  ({oi} frames @ {args.fps} fps = "
          f"{oi / args.fps:.1f} s)")
    subprocess.run(["rm", "-rf", work])


if __name__ == "__main__":
    main()
