#!/usr/bin/env python3
"""machmallow hero — turn a rendered frame sequence into a finished Short.

Takes a directory of rendered PNG frames (e.g. vorticity icefire frames from
tools/schlieren_video.py) OR a .vthb prefix (which it renders first), and
produces a publish-ready clip:

  * 9:16 vertical (1080x1920) for Shorts/Reels/TikTok — rotates a horizontal
    wake so the flow runs DOWN the tall frame and fills it; or 16:9 for
    long-form (--format wide).
  * SEAMLESS LOOP — detects the temporal period of a periodic flow (von
    Kármán, Kelvin-Helmholtz…) by image self-similarity and keeps the longest
    whole-period segment whose end frame matches the start (invisible seam).
    Period-cut beats ping-pong: reversed vortices look unphysical.
  * brand watermark — discreet "machmallow" wordmark, bottom-right.

Examples:
  # one-stop: render vorticity icefire + loop + watermark, vertical Short
  python3 tools/hero.py --prefix out/von_karman --start 120 \
      --mask-circle 1.2,0,0.1 --out out/brand/von_karman_short.mp4

  # from already-rendered frames
  python3 tools/hero.py --frames-dir /tmp/vk_frames --out short.mp4
"""
import argparse
import glob
import os
import subprocess
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))


# ---- branding ------------------------------------------------------------
def find_font(size, regular=False):
    """Space Grotesk if installed, else DejaVu Sans, else default.
    `regular` picks the lighter Regular weight (for subtitles)."""
    heavy = [
        os.path.expanduser("~/Library/Fonts/SpaceGrotesk-Medium.otf"),
        os.path.expanduser("~/Library/Fonts/SpaceGrotesk-Bold.otf"),
        "/Library/Fonts/SpaceGrotesk-Medium.otf",
        "/System/Library/Fonts/Supplemental/DejaVuSans-Bold.ttf",
        "/Library/Fonts/DejaVuSans-Bold.ttf",
    ]
    light = [
        os.path.expanduser("~/Library/Fonts/SpaceGrotesk-Regular.otf"),
        os.path.expanduser("~/Library/Fonts/SpaceGrotesk-Light.otf"),
        "/System/Library/Fonts/Supplemental/DejaVuSans.ttf",
    ]
    candidates = (light + heavy) if regular else heavy
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


ACCENT = (159, 211, 255)


def add_label(img, title, subtitle=None):
    """Professional two-line title at the top: bold keyword line + a smaller
    keyword-rich subtitle (SEO). Clean text with a soft halo, no box."""
    W, H = img.size
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    y = int(H * 0.040)
    tfont = find_font(int(H * 0.050))
    d.text((W // 2, y), title, font=tfont, anchor="ma",
           fill=(255, 255, 255, 245), stroke_width=max(3, H // 380),
           stroke_fill=(0, 0, 0, 235))
    if subtitle:
        sfont = find_font(int(H * 0.0225), regular=True)
        d.text((W // 2, y + int(H * 0.062)), subtitle, font=sfont,
               anchor="ma", fill=ACCENT + (235,), stroke_width=2,
               stroke_fill=(0, 0, 0, 210))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def add_flow_arrow(img, direction="down", opacity=0.92):
    """Arrow marking the flow direction, with a small 'flow' caption."""
    W, H = img.size
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    a = int(255 * opacity)
    col = ACCENT + (a,)
    lw = max(5, H // 220)
    if direction in ("down", "up"):
        x = int(W * 0.15)
        y1, y2 = int(H * 0.135), int(H * 0.225)
        if direction == "up":
            y1, y2 = y2, y1
        hw = int(H * 0.017)
        tip = (x, y2)
        base = y2 - hw * 1.7 if direction == "down" else y2 + hw * 1.7
        head = [(x - hw, base), (x + hw, base), tip]
        cap = (x, min(y1, y2) - int(H * 0.008))
        anchor = "mb"
    else:  # left / right
        y = int(H * 0.15)
        x1, x2 = int(W * 0.30), int(W * 0.70)
        if direction == "left":
            x1, x2 = x2, x1
        hw = int(W * 0.03)
        tip = (x2, y)
        base = x2 - hw * 1.7 if direction == "right" else x2 + hw * 1.7
        head = [(base, y - hw), (base, y + hw), tip]
        cap = (min(x1, x2), y - int(H * 0.02))
        anchor = "rb"
    # dark halo underlay then the coloured stroke
    if direction in ("down", "up"):
        d.line([(x, y1), (x, y2)], fill=(0, 0, 0, 200), width=lw + 6)
        d.line([(x, y1), (x, y2)], fill=col, width=lw)
    else:
        d.line([(x1, y), (x2, y)], fill=(0, 0, 0, 200), width=lw + 6)
        d.line([(x1, y), (x2, y)], fill=col, width=lw)
    d.polygon(head, fill=col)
    font = find_font(int(H * 0.024))
    d.text(cap, "flow", font=font, anchor=anchor, fill=(255, 255, 255, a),
           stroke_width=2, stroke_fill=(0, 0, 0, 200))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


def add_watermark(img, text="machmallow", opacity=0.62):
    """Discreet wordmark bottom-right, white with a soft dark halo."""
    W, H = img.size
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    font = find_font(max(18, int(H * 0.023)))
    a = int(255 * opacity)
    m = int(H * 0.028)
    d.text((W - m, H - m), text, font=font, anchor="rs",
           fill=(255, 255, 255, a), stroke_width=max(2, H // 540),
           stroke_fill=(0, 0, 0, int(a * 0.85)))
    return Image.alpha_composite(img.convert("RGBA"), over).convert("RGB")


# ---- geometry ------------------------------------------------------------
def to_vertical(img, W=1080, H=1920, top=0.5):
    """Rotate a horizontal wake 90° CW (flow -> top-to-bottom), then scale to
    COVER WxH and crop (no black bars): centre horizontally, anchor the
    vertical crop with `top` (0=top .. 1=bottom, 0.5=centre)."""
    r = img.transpose(Image.ROTATE_270)          # 90° clockwise
    rw, rh = r.size
    s = max(W / rw, H / rh)                       # cover
    r = r.resize((max(W, round(rw * s)), max(H, round(rh * s))),
                 Image.LANCZOS)
    rw, rh = r.size
    x0 = (rw - W) // 2                            # centre horizontally
    y0 = max(0, min(int(round((rh - H) * top)), rh - H))
    return r.crop((x0, y0, x0 + W, y0 + H))


def to_wide(img, W=1920, H=1080):
    """Center-crop/scale a frame to 16:9."""
    iw, ih = img.size
    s = max(W / iw, H / ih)
    r = img.resize((int(round(iw * s)), int(round(ih * s))), Image.LANCZOS)
    rw, rh = r.size
    return r.crop(((rw - W) // 2, (rh - H) // 2, (rw - W) // 2 + W,
                  (rh - H) // 2 + H))


# ---- seamless loop -------------------------------------------------------
def loop_length(frames, fps, pmin_s=0.2, slack=1.4):
    """Pick the longest whole-period segment with an invisible seam.

    Builds tiny grayscale signatures, computes seam(L) = ||f0 - f[L]|| for each
    candidate length L, and returns the LARGEST L whose seam is within `slack`x
    the best seam found (a high multiple of the shedding period). Returns the
    number of frames to keep (loop = frames[0:L])."""
    n = len(frames)
    sig = []
    for f in frames:
        g = np.asarray(f.convert("L").resize((96, 160), Image.BILINEAR),
                       dtype=np.float32)
        sig.append(g)
    sig = np.stack(sig)
    pmin = max(2, int(pmin_s * fps))
    seam = np.array([np.sqrt(np.mean((sig[L] - sig[0]) ** 2))
                     for L in range(n)])
    window = seam[pmin:]
    if len(window) == 0:
        return n
    best = window.min()
    thresh = best * slack
    good = [L for L in range(pmin, n) if seam[L] <= thresh]
    return max(good) if good else (pmin + int(np.argmin(window)))


def crossfade_loop(frames, k):
    """Smooth the loop seam by blending the tail into the head over k frames.
    Returns a clip of length len-k whose end->start transition is continuous —
    essential for APERIODIC flows (e.g. the turbulent Mach-2 wake) that have no
    exact period to cut on."""
    n = len(frames)
    if k <= 0 or n < 2 * k + 1:
        return frames
    head = []
    for i in range(k):
        w = 1.0 - (i + 1) / (k + 1)                 # tail weight, high->low
        a = np.asarray(frames[i], np.float32)
        b = np.asarray(frames[n - k + i], np.float32)
        head.append(Image.fromarray(((1 - w) * a + w * b).astype(np.uint8)))
    return head + frames[k:n - k]


# ---- io ------------------------------------------------------------------
def load_frames(frames_dir):
    paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not paths:
        raise SystemExit(f"no PNG frames in {frames_dir}")
    return [Image.open(p).convert("RGB") for p in paths]


def render_frames(prefix, frames_dir, start, end, mask_circle, height):
    """Render vorticity-icefire frames via schlieren_video.py."""
    os.makedirs(frames_dir, exist_ok=True)
    cmd = ["python3", os.path.join(HERE, "schlieren_video.py"),
           "--prefix", prefix, "--style", "vorticity", "--cmap", "icefire",
           "--full", "--height", str(height), "--start", str(start),
           "--frames-dir", frames_dir, "--out",
           os.path.join(tempfile.gettempdir(), "hero_raw.mp4")]
    if end is not None:
        cmd += ["--end", str(end)]
    if mask_circle:
        cmd += ["--mask-circle", mask_circle]
    subprocess.run(cmd, check=True)


def encode(frames, out, fps):
    tmp = tempfile.mkdtemp(prefix="hero_")
    for i, f in enumerate(frames):
        f.save(os.path.join(tmp, f"f_{i:04d}.png"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-i", os.path.join(tmp, "f_%04d.png"), "-c:v", "libx264",
         "-crf", "18", "-pix_fmt", "yuv420p", out], check=True)
    print(f"wrote {out}  ({len(frames)} frames @ {fps} fps "
          f"= {len(frames)/fps:.1f}s)")


def main():
    ap = argparse.ArgumentParser(description="machmallow hero — loop + brand")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames-dir", help="directory of rendered PNG frames")
    src.add_argument("--prefix", help=".vthb prefix to render first")
    ap.add_argument("--out", required=True, help="output .mp4")
    ap.add_argument("--format", choices=["vertical", "wide"],
                    default="vertical", help="9:16 Short (default) or 16:9")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-loop", action="store_true",
                    help="keep the full sequence (no seamless-loop cut)")
    ap.add_argument("--no-watermark", action="store_true")
    ap.add_argument("--top", type=float, default=0.04,
                    help="vertical crop anchor (0=top); margin above the body")
    ap.add_argument("--min-seconds", type=float, default=0.0,
                    help="tile the loop to reach at least this duration")
    ap.add_argument("--label", default=None,
                    help="bold title line at the top (e.g. 'Bow Shock')")
    ap.add_argument("--sublabel", default=None,
                    help="smaller keyword-rich subtitle under the title (SEO)")
    ap.add_argument("--flow-arrow",
                    choices=["down", "up", "left", "right", "none"],
                    default="none", help="draw a flow-direction arrow")
    ap.add_argument("--crossfade", type=int, default=0,
                    help="blend N frames at the loop seam (aperiodic flows)")
    # rendering (with --prefix)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--mask-circle", default=None)
    ap.add_argument("--render-height", type=int, default=1280)
    args = ap.parse_args()

    if args.prefix:
        frames_dir = tempfile.mkdtemp(prefix="hero_frames_")
        render_frames(args.prefix, frames_dir, args.start, args.end,
                      args.mask_circle, args.render_height)
    else:
        frames_dir = args.frames_dir
    frames = load_frames(frames_dir)
    print(f"loaded {len(frames)} frames")

    fit = to_wide if args.format == "wide" else \
        (lambda im: to_vertical(im, top=args.top))
    frames = [fit(f) for f in frames]

    if not args.no_loop:
        L = loop_length(frames, args.fps)
        print(f"seamless loop: keeping {L}/{len(frames)} frames "
              f"({L/args.fps:.2f}s base loop)")
        frames = frames[:L]

    if args.crossfade > 0:
        frames = crossfade_loop(frames, args.crossfade)
        print(f"crossfaded seam over {args.crossfade} frames "
              f"-> {len(frames)} frames")

    if args.min_seconds > 0:
        need = int(np.ceil(args.min_seconds * args.fps))
        reps = max(1, int(np.ceil(need / len(frames))))
        frames = (frames * reps)
        print(f"tiled x{reps} -> {len(frames)} frames "
              f"({len(frames)/args.fps:.1f}s)")

    # overlays on every output frame
    if args.label:
        frames = [add_label(f, args.label, args.sublabel) for f in frames]
    if args.flow_arrow != "none":
        frames = [add_flow_arrow(f, args.flow_arrow) for f in frames]
    if not args.no_watermark:
        frames = [add_watermark(f) for f in frames]

    encode(frames, args.out, args.fps)


if __name__ == "__main__":
    main()
