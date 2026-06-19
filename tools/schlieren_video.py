#!/usr/bin/env python3
"""Post-traitement "cinema" des sorties AMR (.vthb) -> images + video.

Lit chaque frame vtkOverlappingAMR, la reechantillonne sur une grille
uniforme (niveau le plus fin compose automatiquement), calcule un champ
qui CLAQUE (schlieren numerique |grad rho|, le standard des ecoulements
compressibles : ca fait exploser chocs, contacts et enroulements KH),
recadre, colorie, et assemble en MP4 via ffmpeg.

CAMERA QUI SUIT (defaut pour le DMR) : le point triple balaie le
domaine ; un recadrage fixe serait vide au debut. Le script detecte le
pied de Mach a chaque frame, lisse la trajectoire (pan
lineaire) et recentre la fenetre 9:16 dessus -> la cascade KH reste
plein cadre du debut a la fin.

Exemples
--------
python3 tools/schlieren_video.py --prefix out/dmr_hero               # suivi auto
python3 tools/schlieren_video.py --prefix out/dmr_hero --full        # domaine entier
python3 tools/schlieren_video.py --prefix out/dmr_hero --crop 2,2.95,0,0.8

Dependances : vtk, numpy, matplotlib (Pillow), et ffmpeg dans le PATH.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    sys.exit("vtk introuvable : pip install vtk")
import matplotlib
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

# "icefire" : divergente à CENTRE NOIR — la vorticité nulle (free-stream)
# devient noire et les tourbillons brillent en cyan (sens horaire) / ambre
# (anti-horaire). Fond sombre = rendu hypnotique de l'allée tourbillonnaire.
if "icefire" not in colormaps:
    colormaps.register(LinearSegmentedColormap.from_list("icefire", [
        (0.00, "#bdf0ff"), (0.18, "#1fb6ff"), (0.36, "#0a3a6b"),
        (0.50, "#000000"),
        (0.64, "#6b1a0a"), (0.82, "#ff5a1f"), (1.00, "#ffe39a")]))


def read_amr(path):
    r = vtk.vtkXMLUniformGridAMRReader()
    r.SetFileName(path)
    r.Update()
    return r.GetOutput()


def resample(amr, bounds, dims, field="rho"):
    """AMR -> tableau 2D (H, W), y croissant vers le HAUT de l'image.

    Les .vti portent des donnees PAR CELLULE (constantes par morceaux) :
    telles quelles, les mailles grossieres donnent des ESCALIERS dont le
    gradient (schlieren) fait ressortir les bords de maille AMR. On
    convertit d'abord cellule->point (interpolation lineaire) -> rampes
    lisses, plus d'escaliers, SANS flouter chocs/KH.

    On COMPOSITE ensuite a la main (numpy) au lieu de vtkResampleToImage :
    ce dernier sonde chaque pixel dans la hierarchie (locate tres lent,
    ~100 s a 1600x400). Ici on peint chaque patch (bilineaire) dans la
    grille de sortie, du grossier au fin (le fin ecrase) -> ~1 s/frame."""
    x0, x1, y0, y1 = bounds
    W, H = dims
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputDataObject(amr)
    c2p.Update()
    amr_pt = c2p.GetOutputDataObject(0)
    acc = np.full((H, W), np.nan, dtype=np.float32)
    xs = x0 + (np.arange(W) + 0.5) / W * (x1 - x0)   # centres pixels (domaine)
    ys = y0 + (np.arange(H) + 0.5) / H * (y1 - y0)
    for lvl in range(amr_pt.GetNumberOfLevels()):
        for k in range(amr_pt.GetNumberOfBlocks(lvl)):
            g = amr_pt.GetDataSetAsImageData(lvl, k)
            if g is None:
                continue
            av = g.GetPointData().GetArray(field)
            if av is None:
                continue
            ox, oy, _ = g.GetOrigin()
            dx, dy, _ = g.GetSpacing()
            nx, ny, _ = g.GetDimensions()
            arr = vtk_to_numpy(av).reshape(ny, nx)       # (j, i) point data
            fi = (xs - ox) / dx
            fj = (ys - oy) / dy
            ci = np.where((fi >= 0) & (fi <= nx - 1))[0]
            cj = np.where((fj >= 0) & (fj <= ny - 1))[0]
            if len(ci) == 0 or len(cj) == 0:
                continue
            fic, fjc = fi[ci], fj[cj]
            i0 = np.clip(np.floor(fic).astype(int), 0, nx - 2); i1 = i0 + 1
            j0 = np.clip(np.floor(fjc).astype(int), 0, ny - 2); j1 = j0 + 1
            ti = (fic - i0)[None, :]; tj = (fjc - j0)[:, None]
            v = (arr[np.ix_(j0, i0)] * (1 - ti) * (1 - tj) +
                 arr[np.ix_(j0, i1)] * ti * (1 - tj) +
                 arr[np.ix_(j1, i0)] * (1 - ti) * tj +
                 arr[np.ix_(j1, i1)] * ti * tj)
            acc[np.ix_(cj, ci)] = v                       # le fin ecrase
    return np.flipud(np.nan_to_num(acc))                  # y=0 en bas


def blur(a, sigma):
    """Flou gaussien separable, VECTORISE (somme de translations -> O(taille
    du noyau) operations numpy, pas une boucle Python par ligne). Padding
    reflechi (pas de liseré de bord). Tue les marches aux frontieres de
    mailles AMR grossieres sans toucher aux structures physiques."""
    if sigma <= 0:
        return a
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    n = a.shape[1]
    ap = np.pad(a, ((0, 0), (r, r)), mode="reflect")
    a = sum(w * ap[:, t:t + n] for t, w in enumerate(k))
    m = a.shape[0]
    ap = np.pad(a, ((r, r), (0, 0)), mode="reflect")
    a = sum(w * ap[t:t + m, :] for t, w in enumerate(k))
    return a


def schlieren(rho, sigma=0.0):
    """|grad rho| (grille uniforme -> magnitude du gradient en pixels)."""
    gy, gx = np.gradient(blur(rho, sigma))
    return np.hypot(gx, gy)


def vorticity(amr, bounds, dims, sigma=0.0):
    """omega = dv/dx - du/dy (signe = sens de rotation). On rééchantillonne
    u et v sur la même grille puis on dérive. La grille a la rangée 0 en
    HAUT (y décroissant vers le bas), d'où le signe sur du/dy."""
    u = blur(resample(amr, bounds, dims, "u"), sigma)
    v = blur(resample(amr, bounds, dims, "v"), sigma)
    x0, x1, y0, y1 = bounds
    W, H = dims
    dx = (x1 - x0) / W
    dy = (y1 - y0) / H
    dvdx = np.gradient(v, dx, axis=1)
    dudy = -np.gradient(u, dy, axis=0)  # rangée croissante = y décroissant
    return dvdx - dudy


def stem_x(amr, full_bounds, x_corner):
    """x du pied de Mach (choc le plus a droite touchant la
    paroi). On EXCLUT la zone du coin (x = 1/6, ou la BC bascule
    inflow->mur et cree un gradient parasite) en ne cherchant qu'a droite
    de x_corner + 0.2. La cascade KH traine juste a gauche du pied."""
    x0, x1, y0, y1 = full_bounds
    W = 600
    wall = y0 + 0.12 * (y1 - y0)                  # bande collee a la paroi
    g = schlieren(resample(amr, (x0, x1, y0, wall), (W, 24)))
    cols = g.max(axis=0)
    cut = int((x_corner + 0.20 - x0) / (x1 - x0) * (W - 1))
    cols[:max(cut, 0)] = 0.0                      # ignore le coin/l'inflow
    hot = np.where(cols > 0.40 * cols.max())[0]
    xpix = hot[-1] if len(hot) else cut
    return max(x0 + (x1 - x0) * xpix / (W - 1), x_corner + 0.05)


def robust_line(y):
    """Droite de Theil-Sen (mediane des pentes par paires) : insensible
    aux ratures de detection (le DMR auto-semblable est exactement
    lineaire en temps)."""
    n = len(y)
    i = np.arange(n)
    a, b = np.triu_indices(n, 1)
    slope = np.median((y[b] - y[a]) / (b - a))
    inter = np.median(y - slope * i)
    return slope * i + inter


def track_window(files, full_bounds, ymax):
    """Camera zoom+pan auto-semblable : ancree a la paroi, hauteur ~
    distance au coin (le DMR est auto-semblable), pan lineaire en x."""
    x0, x1, y0, y1 = full_bounds
    x_corner = 1.0 / 6.0
    xs = robust_line(
        np.array([stem_x(read_amr(f), full_bounds, x_corner) for f in files]))
    idx = np.arange(len(files))
    h = np.clip(0.62 * (xs - x_corner), 0.16, ymax - y0)  # hauteur fenetre
    w = h * 9 / 16                                        # 9:16 vertical
    cx = np.clip(xs - 0.20 * w, x0 + w / 2, x1 - w / 2)   # tige a ~70%
    return [(cx[i] - w[i] / 2, cx[i] + w[i] / 2, y0, y0 + h[i])
            for i in idx]


def annotate_dmr(rgb, rho, bounds, title, out):
    """Surcouche pedagogique : detecte le point triple et legende les
    structures du DMR (chocs incident/reflechi, pied de Mach, points
    triples primaire ET secondaire -- d'ou le "double" --, ligne de
    glissement / Kelvin-Helmholtz, jet de paroi)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    # style "Math / LaTeX" sans installation LaTeX : polices STIX (serif
    # facon journal, avec accents) + mathtext pour les expressions math.
    mpl.rcParams.update({"font.family": "STIXGeneral",
                         "mathtext.fontset": "stix",
                         "axes.unicode_minus": False})
    H, W = rgb.shape[:2]
    x0, x1, y0, y1 = bounds

    def px(x, y):                          # domaine -> pixel (row0 = haut)
        return ((x - x0) / (x1 - x0) * (W - 1),
                (y1 - y) / (y1 - y0) * (H - 1))

    # Pied du choc de Mach (base, contre la paroi) : choc le plus a droite
    # touchant la paroi, hors zone du coin. Tout le DMR etant AUTO-SEMBLABLE,
    # on en deduit chaque structure par des decalages normalises par
    # l'echelle L = (x_pied - x_coin), mesures une fois sur la grille.
    g = schlieren(rho)
    x_corner = 1.0 / 6.0
    band = g[int(H * 0.88):, :]            # bande collee a la paroi (bas)
    cols = band.max(axis=0)
    cut = int((x_corner + 0.20 - x0) / (x1 - x0) * (W - 1))
    cols[:max(cut, 0)] = 0.0
    hot = np.where(cols > 0.40 * cols.max())[0]
    xf = x0 + (hot[-1] if len(hot) else W - 1) / (W - 1) * (x1 - x0)
    L = max(xf - x_corner, 1e-3)
    f = L / 2.62                           # echelle vs la frame de reference

    # (dx, dy) normalises, mesures sur la grille (pied de Mach = origine).
    # DMR double : T1 (point triple primaire) au sommet du pied de Mach ;
    # T2 (secondaire) sur le choc reflechi PRIMAIRE, d'ou part le choc
    # reflechi SECONDAIRE.
    P = lambda dx, dy: (xf + f * dx, f * dy)
    A = {"inc": P(0.07, 0.66), "t1": P(-0.05, 0.45),
         "ref1": P(-0.29, 0.425), "ref2": P(-0.49, 0.30),
         "t2": P(-0.59, 0.39), "stem1": P(-0.02, 0.22),
         "stem2": P(-0.79, 0.45), "kh": P(-0.24, 0.26),
         "jet": P(-0.24, 0.06)}

    # axe plein cadre -> sortie EXACTEMENT W x H (memes dimensions que les
    # frames video, donc collable a la fin du .mp4)
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb)
    ax.set_axis_off()
    s = H / 640.0                          # echelle des polices vs reference
    halo = [pe.withStroke(linewidth=3 * s, foreground="white")]

    def lab(text, anchor, text_xy, ha="center"):
        ax.annotate(
            text, xy=px(*anchor), xytext=px(*text_xy), ha=ha,
            va="center", fontsize=7 * s, color="#1a1a1a",
            path_effects=halo, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#1a1a1a", lw=0.8 * s,
                            shrinkA=0, shrinkB=2 * s))

    # textes : structures de droite -> labels a droite ; le reste a gauche
    # (dans la zone blanche), ordonnes par hauteur d'ancre -> pas de
    # croisement de fleches.
    # labels dans le blanc ADJACENT a chaque structure -> fleches courtes,
    # texte hors des ondes. Droite : features de droite ; au-dessus de
    # l'arc : chocs reflechis + T2 ; bas-gauche : glissement / jet.
    # droite : structures de droite
    lab("Choc incident", A["inc"], P(0.18, 0.70), ha="left")
    lab("Point triple primaire", A["t1"], P(0.18, 0.50), ha="left")
    lab("Pied de Mach primaire", A["stem1"], P(0.18, 0.26), ha="left")
    # au-dessus de l'arc : T2 et les structures secondaires hautes
    lab("Choc réfléchi primaire", A["ref1"], P(-0.31, 0.58), ha="center")
    lab("Point triple secondaire", A["t2"], P(-0.69, 0.50), ha="center")
    lab("Pied de Mach secondaire", A["stem2"], P(-1.12, 0.58), ha="right")
    # gauche-bas : choc reflechi secondaire, glissement, jet
    lab("Choc réfléchi secondaire", A["ref2"], P(-0.99, 0.34), ha="right")
    lab("Ligne de glissement\n(Kelvin-Helmholtz)", A["kh"],
        P(-0.99, 0.18), ha="right")
    lab("Jet", A["jet"], P(-0.99, 0.02), ha="right")

    if title:
        ax.text(0.5, 0.97, title, transform=ax.transAxes, ha="center",
                va="top", fontsize=10 * s, fontweight="bold", color="#111",
                path_effects=halo)
    fig.savefig(out, dpi=200)         # plein cadre exact (pas de bbox tight)
    plt.close(fig)


def colorize(rho, scale, style, gamma, k, floor, sigma, cmap):
    if style == "vorticity":
        # `rho` porte déjà omega ; échelle symétrique -> divergente centrée.
        n = 0.5 + 0.5 * np.clip(rho / max(scale, 1e-12), -1, 1)
        rgb = colormaps[cmap](n)[..., :3]
    elif style == "density":
        n = np.clip((rho - scale[0]) / max(scale[1] - scale[0], 1e-9), 0, 1)
        rgb = colormaps[cmap](n)[..., :3]
    else:
        g = schlieren(rho, sigma)
        n = np.clip(g / max(scale, 1e-12), 0, 1)
        n = np.clip((n - floor) / (1.0 - floor), 0, 1) ** gamma
        if style == "schlieren":          # fond blanc -> noir (classique)
            rgb = cm.gray(np.exp(-k * g / max(scale, 1e-12)))[..., :3]
        elif style == "light":            # FOND BLANC + teinte coloree
            col = colormaps[cmap](n)[..., :3]
            a = n[..., None]
            rgb = (1 - a) * 1.0 + a * col  # lisse = blanc, structures = teinte
        else:                             # "fire" : fond noir, aretes vives
            rgb = colormaps[cmap](n)[..., :3]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def mask_circle(rgb, bounds, cx, cy, r):
    """Peint un disque (corps immergé) en BLANC sur l'image (le solveur y
    fige le free-stream, donc le champ ne marque pas le corps)."""
    x0, x1, y0, y1 = bounds
    H, W = rgb.shape[:2]
    xs = x0 + (np.arange(W) + 0.5) / W * (x1 - x0)
    ys = y0 + (np.arange(H) + 0.5) / H * (y1 - y0)
    X, Y = np.meshgrid(xs, ys)
    inside = np.flipud((X - cx) ** 2 + (Y - cy) ** 2 <= r * r)  # row0 = haut
    rgb[inside] = 255
    return rgb


def overlay_schlieren(rgb, rho, scale, floor, gamma, sigma, weight):
    """Superpose les CHOCS (|grad rho| fort) en BLANC sur l'image (mélange
    vers blanc). Le seuil `floor` élevé n'y laisse que les discontinuités
    fortes (chocs) — les gradients doux du sillage restent transparents."""
    g = schlieren(rho, sigma)
    n = np.clip(g / max(scale, 1e-12), 0, 1)
    n = np.clip((n - floor) / (1 - floor), 0, 1) ** gamma
    a = (n * weight)[..., None]
    out = rgb.astype(np.float32) / 255.0
    out = out * (1 - a) + a
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def amr_boxes(amr, bounds):
    """Etendues (x0, x1, y0, y1, niveau) de chaque patch AMR (niveaux >= 1),
    pour dessiner le maillage adaptatif et le voir evoluer."""
    bx0, bx1, by0, by1 = bounds
    out = []
    for lvl in range(1, amr.GetNumberOfLevels()):
        for k in range(amr.GetNumberOfBlocks(lvl)):
            g = amr.GetDataSetAsImageData(lvl, k)
            if g is None:
                continue
            ox, oy, _ = g.GetOrigin()
            dx, dy, _ = g.GetSpacing()
            nx, ny, _ = g.GetDimensions()
            ex0, ex1 = ox, ox + (nx - 1) * dx
            ey0, ey1 = oy, oy + (ny - 1) * dy
            if ex1 < bx0 or ex0 > bx1 or ey1 < by0 or ey0 > by1:
                continue
            out.append((ex0, ex1, ey0, ey1, lvl))
    return out


def density_panel(rho, vmin, vmax, cmap, boxes, bounds, light_bg=False):
    """Champ de densite colorise + contours des patchs AMR -> on voit le
    raffinement adaptatif suivre les structures. `light_bg` : fond clair
    (densite blanc->teinte) avec un maillage en UNE teinte chaude qui
    contraste a la fois sur le blanc et sur la couleur dense."""
    n = np.clip((rho - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgb = (colormaps[cmap](n)[..., :3] * 255).astype(np.uint8).copy()
    H, W = rgb.shape[:2]
    x0, x1, y0, y1 = bounds
    pal = ([(214, 64, 18)] if light_bg                 # maillage : 1 teinte chaude
           else [(255, 255, 255), (0, 255, 255), (160, 255, 0),
                 (255, 190, 0), (255, 0, 200)])        # par niveau (fond sombre)
    pX = lambda x: int(round((x - x0) / (x1 - x0) * (W - 1)))
    pY = lambda y: int(round((y1 - y) / (y1 - y0) * (H - 1)))  # y=0 en bas
    for ex0, ex1, ey0, ey1, lvl in boxes:
        i0, i1 = sorted((max(0, min(W - 1, pX(ex0))),
                         max(0, min(W - 1, pX(ex1)))))
        ja, jb = sorted((max(0, min(H - 1, pY(ey1))),
                         max(0, min(H - 1, pY(ey0)))))
        c = pal[(lvl - 1) % len(pal)]
        rgb[ja, i0:i1 + 1] = c; rgb[jb, i0:i1 + 1] = c
        rgb[ja:jb + 1, i0] = c; rgb[ja:jb + 1, i1] = c
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="out/dmr_hero")
    ap.add_argument("--out", default=None)
    ap.add_argument("--style", default="fire",
                    choices=["fire", "light", "schlieren", "density",
                             "vorticity"],
                    help="fire=fond noir | light=fond blanc | schlieren="
                         "N&B classique | density=champ rempli | "
                         "vorticity=rotation (divergente, allée tourbillonnaire)")
    ap.add_argument("--cmap", default=None,
                    help="colormap matplotlib (ex: inferno, plasma, viridis, "
                         "magma, cividis, turbo). Defaut selon le style.")
    ap.add_argument("--crop", default=None,
                    help="x0,x1,y0,y1 fixe (sinon : suivi auto)")
    ap.add_argument("--mask-circle", default=None,
                    help="cx,cy,r : disque blanc (corps immergé)")
    ap.add_argument("--rho-range", default=None,
                    help="lo,hi : borne la densité (style density) pour "
                         "étaler la dynamique au lieu de min/max auto")
    ap.add_argument("--schlieren-overlay", type=float, default=0.0,
                    help="superpose les chocs (|grad rho|) en blanc, poids "
                         "0..1 (ex 0.85) — p.ex. sur --style vorticity")
    ap.add_argument("--overlay-floor", type=float, default=0.45,
                    help="seuil de l'overlay schlieren (haut = seuls les "
                         "chocs ; défaut 0.45)")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--ytop", type=float, default=0.8,
                    help="hauteur de la fenetre de suivi (9:16)")
    ap.add_argument("--height", type=int, default=1920)
    ap.add_argument("--max-width", type=int, default=2560,
                    help="plafond de largeur (le --full paysage exploserait "
                         "sinon a 7680px)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--still", default=None,
                    help="ne rendre QU'UNE frame -> PNG (index, ou 'last') ; "
                         "instantane, pas de video")
    ap.add_argument("--annotate", action="store_true",
                    help="surcouche pedagogique : legende les structures du "
                         "DMR (avec --still)")
    ap.add_argument("--title", default=r"Réflexion de Mach double — "
                    r"$M_s = 10$ (WENO5 + AMR)",
                    help="titre de la figure annotée (mathtext supporte)")
    ap.add_argument("--gamma", type=float, default=0.6,
                    help="<1 = remonte les faibles gradients (style fire)")
    ap.add_argument("--floor", type=float, default=None,
                    help="plancher : separe les residus AMR (faibles) du "
                         "KH/chocs (forts) SANS lisser. Defaut 0.10 (fond "
                         "noir) / 0.22 (fond blanc, qui revele les residus)")
    ap.add_argument("--blur", type=float, default=0.0,
                    help="flou optionnel (px @ hauteur 1280). 0 par defaut : "
                         "l'interpolation cellule->point retire deja les "
                         "escaliers AMR sans toucher au KH")
    ap.add_argument("--k", type=float, default=14.0)
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--start", type=int, default=0,
                    help="index de depart (ignore les frames AVANT ; "
                         "0-base sur la liste triee)")
    ap.add_argument("--end", type=int, default=None,
                    help="index de fin (exclu)")
    ap.add_argument("--hold", type=int, default=1,
                    help="repeter chaque frame K fois (ralenti ; K=2 -> 2x "
                         "plus lent)")
    ap.add_argument("--freeze", type=float, default=0.0,
                    help="figer la frame annotee SEC secondes a la fin "
                         "(DMR uniquement ; ex. 2.5)")
    ap.add_argument("--amr-panel", action="store_true",
                    help="empiler sous le schlieren un panneau densite + "
                         "blocs AMR (montre le raffinement evoluer)")
    ap.add_argument("--amr-bg", default="light", choices=["light", "dark"],
                    help="fond du panneau densite : light (blanc->teinte, "
                         "maillage chaud) ou dark (colormap saturee)")
    ap.add_argument("--amr-cmap", default=None,
                    help="colormap du panneau densite (defaut : Blues si "
                         "light, turbo si dark)")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.prefix}_*.vthb"))
    if not files:
        sys.exit(f"aucune frame {args.prefix}_*.vthb")
    files = files[args.start:args.end]    # sous-sequence demandee
    if not files:
        sys.exit(f"plage vide (--start {args.start} --end {args.end})")
    print(f"plage : frames [{args.start}:{args.end}] -> {len(files)} frames")
    if args.cmap is None:                 # defaut de colormap selon le style
        args.cmap = {"fire": "inferno", "light": "magma_r",
                     "density": "turbo", "vorticity": "icefire"}.get(
                         args.style, "inferno")
    if args.floor is None:                # fond blanc : couper plus haut
        args.floor = 0.1 if args.style == "light" else 0.10
    if args.amr_cmap is None:             # colormap panneau densite
        args.amr_cmap = "Blues" if args.amr_bg == "light" else "turbo"
    light_bg = args.amr_bg == "light"
    maskc = (tuple(float(v) for v in args.mask_circle.split(","))
             if args.mask_circle else None)
    rhorange = (tuple(float(v) for v in args.rho_range.split(","))
                if args.rho_range else None)

    full = [0.0] * 6
    read_amr(files[0]).GetBounds(full)
    full_bounds = (full[0], full[1], full[2], full[3])

    # --- geometrie de la fenetre : fixe, complete, ou suivi ---
    track = None
    if args.full:
        bounds_of = lambda i: full_bounds
        asp = (full[1] - full[0]) / (full[3] - full[2])
    elif args.crop:
        c = tuple(float(v) for v in args.crop.split(","))
        bounds_of = lambda i: c
        asp = (c[1] - c[0]) / (c[3] - c[2])
    elif args.still is not None:          # 1 frame en suivi : fenetre directe
        idx0 = (len(files) - 1) if args.still == "last" else int(args.still)
        xs = stem_x(read_amr(files[idx0]), full_bounds, 1 / 6)
        h = min(max(0.62 * (xs - 1 / 6), 0.16), args.ytop)
        w = h * 9 / 16
        cx = min(max(xs - 0.20 * w, full[0] + w / 2), full[1] - w / 2)
        one = (cx - w / 2, cx + w / 2, full[2], full[2] + h)
        bounds_of = lambda i: one
        asp = 9 / 16
    else:  # suivi auto (camera zoom+pan sur le point triple)
        print("suivi auto du point triple (zoom+pan)...")
        track = track_window(files, full_bounds, full[2] + args.ytop)
        bounds_of = lambda i: track[i]
        asp = 9 / 16

    H = args.height
    W = int(round(H * asp / 2) * 2)
    if W > args.max_width:               # evite le 7680px du --full paysage
        H = int(round(args.max_width / asp / 2) * 2)
        W = int(round(H * asp / 2) * 2)
    dims = (W, H)
    sigma = args.blur * H / 1280.0       # flou a l'echelle de la resolution
    print(f"{len(files)} frames | sortie {W}x{H} | style {args.style}"
          f"{' | suivi' if track is not None else ''} | flou {sigma:.1f}px")

    # --- raccourci : une seule frame -> PNG, echelle sur cette frame ---
    if args.still is not None:
        from PIL import Image
        idx = (len(files) - 1) if args.still == "last" else int(args.still)
        amr0 = read_amr(files[idx])
        if args.style == "vorticity":
            rho = vorticity(amr0, bounds_of(idx), dims, sigma)
            scale = float(np.percentile(np.abs(rho), 99.5))
        else:
            rho = resample(amr0, bounds_of(idx), dims)
            if args.style == "density":
                scale = rhorange or (float(rho.min()), float(rho.max()))
            else:
                scale = float(np.percentile(schlieren(rho, sigma), 99.5))
        rgb = colorize(rho, scale, args.style, args.gamma, args.k,
                       args.floor, sigma, args.cmap)
        if args.schlieren_overlay > 0:
            dens = resample(amr0, bounds_of(idx), dims)
            ssc = float(np.percentile(schlieren(dens, sigma), 99.5))
            rgb = overlay_schlieren(rgb, dens, ssc, args.overlay_floor,
                                    args.gamma, sigma, args.schlieren_overlay)
        if maskc:
            mask_circle(rgb, bounds_of(idx), *maskc)
        out = args.out or f"{args.prefix}_{args.style}_{idx:04d}.png"
        if args.annotate:
            annotate_dmr(rgb, rho, bounds_of(idx), args.title, out)
        else:
            Image.fromarray(rgb).save(out)
        print(f"image : {out}  (frame {idx})")
        return

    # --- passe 1 : echelle robuste (sous-echantillon, basse resolution) ---
    sub = files[::max(1, len(files) // 10)]
    subi = list(range(0, len(files), max(1, len(files) // 10)))
    dlo, dhi = 1e30, -1e30                 # plage densite (panneau AMR)
    if args.style == "density":
        for f, i in zip(sub, subi):
            r = resample(read_amr(f), bounds_of(i), (W // 4, H // 4))
            dlo, dhi = min(dlo, r.min()), max(dhi, r.max())
        scale = rhorange or (dlo, dhi)
        print(f"  densite dans [{dlo:.3f}, {dhi:.3f}]"
              f"{f' -> bornee {rhorange}' if rhorange else ''}")
    elif args.style == "vorticity":
        pcts = []
        for f, i in zip(sub, subi):
            w = vorticity(read_amr(f), bounds_of(i), (W // 4, H // 4),
                          sigma / 4)
            pcts.append(np.percentile(np.abs(w), 99.5))
        scale = float(np.median(pcts))
        print(f"  echelle |omega| = {scale:.4f}")
    else:
        pcts = []
        for f, i in zip(sub, subi):
            r = resample(read_amr(f), bounds_of(i), (W // 4, H // 4))
            pcts.append(np.percentile(schlieren(r, sigma / 4), 99.5))
            if args.amr_panel:
                dlo, dhi = min(dlo, r.min()), max(dhi, r.max())
        scale = float(np.median(pcts))
        print(f"  echelle |grad rho| = {scale:.4f}")
        if args.amr_panel:
            print(f"  panneau densite dans [{dlo:.3f}, {dhi:.3f}]")

    # échelle de l'overlay schlieren (chocs en blanc) si demandé
    schscale = None
    if args.schlieren_overlay > 0:
        ps = []
        for f, i in zip(sub, subi):
            r = resample(read_amr(f), bounds_of(i), (W // 4, H // 4))
            ps.append(np.percentile(schlieren(r, sigma / 4), 99.5))
        schscale = float(np.median(ps))
        print(f"  overlay chocs |grad rho| = {schscale:.4f}")

    # --- passe 2 : rendu plein resolution (ralenti --hold + fige --freeze) ---
    tmp = args.frames_dir or tempfile.mkdtemp(prefix="schlieren_")
    os.makedirs(tmp, exist_ok=True)
    import shutil
    from PIL import Image
    hold = max(1, args.hold)
    oi = 0                                          # index de sortie courant
    for i, f in enumerate(files):
        amr = read_amr(f)
        rho = (vorticity(amr, bounds_of(i), dims, sigma)
               if args.style == "vorticity"
               else resample(amr, bounds_of(i), dims))
        top = colorize(rho, scale, args.style, args.gamma, args.k,
                       args.floor, sigma, args.cmap)
        if args.schlieren_overlay > 0:              # chocs en blanc par-dessus
            dens = resample(amr, bounds_of(i), dims)
            top = overlay_schlieren(top, dens, schscale, args.overlay_floor,
                                    args.gamma, sigma, args.schlieren_overlay)
        if maskc:
            mask_circle(top, bounds_of(i), *maskc)
        if args.amr_panel:                          # densite + blocs AMR dessous
            bot = density_panel(rho, dlo, dhi, args.amr_cmap,
                                amr_boxes(amr, bounds_of(i)), bounds_of(i), light_bg)
            top = np.vstack([top, bot])
        img = Image.fromarray(top)
        for _ in range(hold):                       # ralenti : K copies
            img.save(os.path.join(tmp, f"f_{oi:04d}.png"))
            oi += 1
        print(f"\r  rendu {i + 1}/{len(files)}", end="", flush=True)
    print()

    if args.freeze > 0:                             # fige annote a la fin
        last = len(files) - 1
        amr = read_amr(files[last])
        rho = resample(amr, bounds_of(last), dims)
        top = colorize(rho, scale, args.style, args.gamma, args.k,
                       args.floor, sigma, args.cmap)
        fz = os.path.join(tmp, "_freeze.png")
        annotate_dmr(top, rho, bounds_of(last), args.title, fz)
        if args.amr_panel:                          # empiler densite+AMR dessous
            top_annot = np.array(Image.open(fz).convert("RGB"))
            bot = density_panel(rho, dlo, dhi, args.amr_cmap,
                                amr_boxes(amr, bounds_of(last)), bounds_of(last), light_bg)
            Image.fromarray(np.vstack([top_annot, bot])).save(fz)
        nfz = int(round(args.freeze * args.fps))
        for _ in range(nfz):
            shutil.copyfile(fz, os.path.join(tmp, f"f_{oi:04d}.png"))
            oi += 1
        os.remove(fz)
        print(f"  fige annote : {nfz} frames ({args.freeze:.1f} s)")

    out = args.out or f"{args.prefix}_{args.style}.mp4"
    cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
           "-i", os.path.join(tmp, "f_%04d.png"),
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", out]
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"video : {out}  ({oi} frames @ {args.fps} fps = "
          f"{oi / args.fps:.1f} s)")
    print(f"apercu PNG : {tmp}/f_*.png")


if __name__ == "__main__":
    main()
