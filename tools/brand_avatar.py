#!/usr/bin/env python3
"""machmallow — brand asset generator (avatar, wordmark, banner).

Hand-drawn mascot: a cute marshmallow panicking before a detached bow shock,
in the channel palette (icefire on pure black). NOT a simulation — pure
vector drawing, so it regenerates instantly and stays 100% on-brand.

    python3 tools/brand_avatar.py            # all assets -> out/brand/
    python3 tools/brand_avatar.py --out DIR  # choose output dir

See BRAND.md for the spec. Title font is Space Grotesk if installed,
otherwise DejaVu Sans Bold (matplotlib's built-in fallback).
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, Polygon, Arc

# ---- brand palette (see BRAND.md) ---------------------------------------
BG       = "#000000"
CYAN     = "#29C5E6"
ACCENT   = "#9FD3FF"
EMBER    = "#F2622E"
GOLD     = "#F6C95B"
WHITEHOT = "#FFF2D0"
CREAM    = "#FBF3E4"
CREAM_D  = "#E7D9C2"
FACE     = "#2E2016"
ROSE     = "#F2622E"

TAGLINE = "a hybrid CPU/GPU AMR solver, coded on a Mac"


def title_font(size, weight="medium"):
    """Space Grotesk (chosen weight) if available, else DejaVu Sans Bold."""
    try:
        fp = fm.FontProperties(family="Space Grotesk", weight=weight)
        path = fm.findfont(fp, fallback_to_default=False)
        if "spacegrotesk" in os.path.basename(path).lower():
            return fm.FontProperties(fname=path, size=size)
    except Exception:
        pass
    return fm.FontProperties(family="DejaVu Sans",
                             weight="bold" if weight in ("medium", "bold")
                             else "normal", size=size)


def lerp(c1, c2, t):
    a = np.array(mc.to_rgb(c1)); b = np.array(mc.to_rgb(c2))
    return tuple(a + (b - a) * t)


def shock_color(t):
    """t=0 stagnation (white-hot/gold) -> t=1 flanks (cyan/accent)."""
    stops = [(0.0, WHITEHOT), (0.16, GOLD), (0.42, EMBER),
             (0.72, CYAN), (1.0, ACCENT)]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t <= t1:
            return lerp(c0, c1, (t - t0) / (t1 - t0))
    return ACCENT


def draw_scene(ax, cx, cy, R, stream_x0, lw_scale=1.0):
    """Draw the full mascot scene centered on (cx, cy) with cap radius R.
    stream_x0 = left x where incoming streamlines begin.
    lw_scale   = thickness multiplier for the incoming streamlines."""
    u = R / 0.120                       # unit scale (offsets tuned at R=0.120)
    half_w, half_h = R, 0.215 * u
    nose_x = cx - half_w

    # --- detached bow shock crescent, in front (left) of the body --------
    standoff = 0.105 * u
    vx = nose_x - standoff
    yt = np.linspace(-0.36 * u, 0.36 * u, 500)
    xs = vx + 1.55 / u * yt**2
    ys = cy + yt
    tnorm = np.abs(yt) / (0.36 * u)
    for lw, al in [(26, 0.05), (15, 0.09), (8, 0.20), (3.6, 0.95)]:
        for i in range(len(xs) - 1):
            ax.plot(xs[i:i+2], ys[i:i+2], color=shock_color(tnorm[i]),
                    lw=lw * u, alpha=al, solid_capstyle="round", zorder=3)

    # --- incoming supersonic streamlines, deflected by the shock ---------
    for y0 in np.linspace(cy - 0.32 * u, cy + 0.32 * u, 8):
        off = y0 - cy
        t = np.linspace(0, 1, 240)
        x = stream_x0 + (vx - 0.015 * u - stream_x0) * t
        prox = np.clip((x - (vx - 0.20 * u)) / (0.20 * u), 0, 1)
        amp = 0.135 * u * np.exp(-(off / (0.20 * u))**2)
        y = y0 + np.sign(off) * amp * prox**1.8
        a_near = 0.22 + 0.30 * t
        ax.plot(x, y, color=CYAN, lw=5.0 * u * lw_scale, alpha=0.07,
                solid_capstyle="round", zorder=1)
        for i in range(len(x) - 1):
            ax.plot(x[i:i+2], y[i:i+2], color=ACCENT, lw=2.0 * u * lw_scale,
                    alpha=float(a_near[i]), solid_capstyle="round", zorder=1)
        xi = 90
        ax.annotate("", xy=(x[xi+8], y[xi+8]), xytext=(x[xi], y[xi]),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT,
                                    lw=1.6, alpha=0.45), zorder=1)

    # --- marshmallow body (vertical stadium, pillowy) --------------------
    def capsule(fc, ec, lw, z, clip=None):
        p = FancyBboxPatch((cx - half_w, cy - half_h), 2*half_w, 2*half_h,
                           boxstyle=f"round,pad=0,rounding_size={half_w}",
                           fc=fc, ec=ec, lw=lw, zorder=z)
        if clip is not None: p.set_clip_path(clip)
        ax.add_patch(p); return p

    body = capsule(CREAM, "none", 0, 5)
    ax.add_patch(Ellipse((cx + 0.06*u, cy - 0.05*u), 0.20*u, 0.40*u,
                 fc=CREAM_D, ec="none", alpha=0.40, zorder=6, clip_path=body))
    ax.add_patch(Ellipse((cx - 0.045*u, cy + 0.10*u), 0.11*u, 0.20*u,
                 fc="#FFFFFF", ec="none", alpha=0.45, zorder=7, clip_path=body))
    ax.add_patch(Ellipse((nose_x + 0.015*u, cy), 0.07*u, 0.40*u, fc=GOLD,
                 ec="none", alpha=0.28, zorder=7, clip_path=body))
    capsule("none", CREAM_D, 1.4, 8)

    # --- panicked face ---------------------------------------------------
    ex, eye_y, dx = cx - 0.008*u, cy + 0.060*u, 0.052*u
    for s in (-1, 1):
        ax.add_patch(Ellipse((ex + s*dx, eye_y), 0.044*u, 0.058*u,
                     fc="#FFFFFF", ec=FACE, lw=1.2, zorder=9))
        ax.add_patch(Circle((ex + s*dx - 0.012*u, eye_y + 0.006*u), 0.0145*u,
                     fc=FACE, ec="none", zorder=10))
        ax.add_patch(Circle((ex + s*dx - 0.016*u, eye_y + 0.012*u), 0.0050*u,
                     fc="#FFFFFF", ec="none", zorder=11))
        ax.add_patch(Arc((ex + s*dx, eye_y + 0.052*u), 0.058*u, 0.040*u,
                     angle=0, theta1=20, theta2=160, color=FACE, lw=2.6,
                     zorder=11))
    for s in (-1, 1):
        ax.add_patch(Ellipse((ex + s*0.086*u, cy + 0.002*u), 0.028*u, 0.018*u,
                     fc=ROSE, ec="none", alpha=0.38, zorder=9))
    ax.add_patch(Ellipse((ex - 0.004*u, cy - 0.052*u), 0.028*u, 0.042*u,
                 fc=FACE, ec="none", zorder=9))
    ax.add_patch(Ellipse((ex - 0.004*u, cy - 0.060*u), 0.016*u, 0.018*u,
                 fc=ROSE, ec="none", alpha=0.55, zorder=10))
    sdx, sdy = cx + half_w - 0.012*u, cy + half_h - 0.018*u
    ax.add_patch(Circle((sdx, sdy), 0.018*u, fc=ACCENT, ec="none", zorder=12))
    ax.add_patch(Polygon([(sdx - 0.013*u, sdy + 0.004*u),
                          (sdx + 0.013*u, sdy + 0.004*u),
                          (sdx, sdy + 0.034*u)], closed=True, fc=ACCENT,
                          ec="none", zorder=12))
    ax.add_patch(Circle((sdx - 0.005*u, sdy + 0.004*u), 0.006*u, fc="#FFFFFF",
                 ec="none", zorder=13))


def _fig(w_in, h_in, xmax=1.0):
    fig = plt.figure(figsize=(w_in, h_in), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, xmax); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG); ax.axis("off")
    return fig, ax


def avatar(out):
    fig, ax = _fig(10.24, 10.24)
    draw_scene(ax, cx=0.605, cy=0.50, R=0.120, stream_x0=0.06)
    fig.savefig(out, facecolor=BG); plt.close(fig); print("wrote", out)


def wordmark(out):
    fig, ax = _fig(10.24, 10.24)
    draw_scene(ax, cx=0.605, cy=0.555, R=0.120, stream_x0=0.06)
    ax.text(0.5, 0.135, "machmallow", ha="center", va="center",
            color="#FFFFFF", fontproperties=title_font(64, "medium"))
    ax.text(0.5, 0.075, TAGLINE, ha="center", va="center", color=ACCENT,
            alpha=0.9, fontproperties=title_font(17, "regular"))
    fig.savefig(out, facecolor=BG); plt.close(fig); print("wrote", out)


def banner(out, debug=False):
    # 2560x1440. Devices crop the banner differently; the ONLY region
    # guaranteed visible everywhere (mobile is the tightest) is the central
    # 1546x423 px "safe zone". All critical content (wordmark + tagline)
    # must live inside it. With aspect=equal and xmax=16/9 (1 unit = 1440px):
    #   safe x in [0.352, 1.426], safe y in [0.353, 0.647].
    AR = 2560 / 1440
    SX0, SX1 = (AR - 1546/1440) / 2, (AR + 1546/1440) / 2   # 0.352 .. 1.426
    SY0, SY1 = (1 - 423/1440) / 2, (1 + 423/1440) / 2        # 0.353 .. 0.647
    fig, ax = _fig(25.60, 14.40, xmax=AR)

    # large mascot on the right, streamlines (thick) sweep the full width.
    # Keep the mascot+shock fully inside the safe zone so mobile shows it
    # whole (shock vertex ~x=1.01, mascot right edge ~1.40 < 1.426).
    draw_scene(ax, cx=1.27, cy=0.50, R=0.135, stream_x0=0.04, lw_scale=1.9)

    # big wordmark inside the safe zone, in the clear band left of the shock
    # (text spans ~x in [0.36, 0.96], shock vertex ~1.01 -> clears it).
    tx = 0.66
    for rw, rh, a in [(0.74, 0.32, 0.50), (0.58, 0.24, 0.50)]:
        ax.add_patch(Ellipse((tx, 0.50), rw, rh, fc=BG, ec="none",
                     alpha=a, zorder=15))
    ax.text(tx, 0.552, "machmallow", ha="center", va="center",
            color="#FFFFFF", fontproperties=title_font(104, "medium"),
            zorder=20)
    ax.text(tx, 0.455, TAGLINE, ha="center", va="center", color=ACCENT,
            alpha=0.92, fontproperties=title_font(30, "regular"), zorder=20)

    if debug:
        ax.add_patch(plt.Rectangle((SX0, SY0), SX1-SX0, SY1-SY0, fill=False,
                     ec="#00FF6A", lw=2, ls="--", zorder=30))
        ax.text(SX0+0.01, SY1-0.02, "safe 1546x423 (all devices)",
                color="#00FF6A", fontsize=11, va="top", zorder=30)
    fig.savefig(out, facecolor=BG); plt.close(fig); print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description="machmallow brand assets")
    ap.add_argument("--out", default="out/brand", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    avatar(os.path.join(args.out, "avatar.png"))
    wordmark(os.path.join(args.out, "wordmark.png"))
    banner(os.path.join(args.out, "banner.png"))


if __name__ == "__main__":
    main()
