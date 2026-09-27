#!/usr/bin/env python3
"""
Where the wearer looks: heatmap of the gaze point over the frame for every embedded frame, with the
centre crop boxes drawn on top (ratio 0.5 solid = what `center_05` crops; 0.35 / 0.25 dashed).

Uses the same 16 frame times per clip and the same gaze_at() lookup as embed_arms.py, so the points
are exactly the ones the `gazef` arm cropped around. Writes a light and a dark SVG to analysis/.
Standard library only (runs on the login node).

    python experiments/gaze_crop/plot_gaze_heatmap.py
"""

import json
import math
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gaze_common import GAZE_PATH, gaze_at  # noqa: E402

OUT = os.path.join(HERE, "analysis")
NFRAMES = 16
BINS = 44
RATIOS = [0.5, 0.35, 0.25]          # first one is the ratio used by every crop arm

# reference sequential ramp (blue 100 -> 700)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
        "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
THEMES = {
    "light": {"surface": "#ffffff", "text": "#1a1a1a", "text2": "#555555", "frame": "#333333",
              "box": "#1a1a1a", "ramp": RAMP},
    "dark": {"surface": "#1a1a19", "text": "#f2f2f2", "text2": "#c3c2b7", "frame": "#bdbdbd",
             "box": "#f2f2f2", "ramp": list(reversed(RAMP))},
}
FONT = "'Helvetica Neue', Helvetica, Arial, 'DejaVu Sans', sans-serif"

W, H = 900, 640
FX, FY, FS = 70, 90, 480            # frame square: left, top, side (px)


def gaze_points() -> List[Tuple[float, float]]:
    clips = json.load(open(GAZE_PATH))["clips"]
    pts = []
    for e in clips.values():
        s = e.get("samples") or []
        if not s:
            continue
        dur = max(float(x[0]) for x in s)
        for i in range(NFRAMES):
            p = gaze_at(e, dur * i / (NFRAMES - 1))
            if p:
                pts.append(p)
    return pts


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def ramp_color(ramp: List[str], t: float) -> str:
    t = min(1.0, max(0.0, t)) * (len(ramp) - 1)
    i = min(int(t), len(ramp) - 2)
    a, b = hex_to_rgb(ramp[i]), hex_to_rgb(ramp[i + 1])
    f = t - i
    return "#%02x%02x%02x" % tuple(round(a[j] + (b[j] - a[j]) * f) for j in range(3))


def render(mode: str, pts: List[Tuple[float, float]]) -> str:
    th = THEMES[mode]
    n = len(pts)
    grid = [[0] * BINS for _ in range(BINS)]
    for x, y in pts:
        grid[min(BINS - 1, int(y * BINS))][min(BINS - 1, int(x * BINS))] += 1
    peak = max(max(r) for r in grid)
    cell = FS / BINS

    s = [f'<text x="{FX}" y="38" font-size="20" font-weight="700" fill="{th["text"]}">Where the wearer looks</text>',
         f'<text x="{FX}" y="62" font-size="14" fill="{th["text2"]}">gaze point in every embedded frame '
         f'({n:,} frames = {n // NFRAMES:,} clips × {NFRAMES}), with the centre crop boxes</text>']
    # heatmap cells; sqrt scale so the thin tail stays visible next to the dense centre
    for r in range(BINS):
        for c in range(BINS):
            v = grid[r][c]
            if not v:
                continue
            col = ramp_color(th["ramp"], math.sqrt(v / peak))
            s.append(f'<rect x="{FX + c * cell:.2f}" y="{FY + r * cell:.2f}" width="{cell + 0.3:.2f}" '
                     f'height="{cell + 0.3:.2f}" fill="{col}"/>')
    s.append(f'<rect x="{FX}" y="{FY}" width="{FS}" height="{FS}" fill="none" stroke="{th["frame"]}" stroke-width="1.5"/>')
    s.append(f'<text x="{FX + FS / 2}" y="{FY + FS + 24}" font-size="13" text-anchor="middle" fill="{th["text2"]}">'
             f'full frame (1408 × 1408 px)</text>')

    # centre crop boxes + share of gaze points inside each
    lx = FX + FS + 40
    s.append(f'<text x="{lx}" y="{FY + 12}" font-size="15" font-weight="700" fill="{th["text"]}">Centre crop box</text>')
    s.append(f'<text x="{lx}" y="{FY + 32}" font-size="13" fill="{th["text2"]}">share of gaze points inside</text>')
    for i, r in enumerate(RATIOS):
        side = FS * r
        x0 = FX + (FS - side) / 2
        inside = sum(1 for x, y in pts if abs(x - 0.5) <= r / 2 and abs(y - 0.5) <= r / 2) / n
        dash = "" if i == 0 else ' stroke-dasharray="7 5"'
        width = 2.5 if i == 0 else 1.6
        s.append(f'<rect x="{x0:.1f}" y="{x0 - FX + FY:.1f}" width="{side:.1f}" height="{side:.1f}" fill="none" '
                 f'stroke="{th["box"]}" stroke-width="{width}"{dash}/>')
        yy = FY + 70 + i * 58
        s.append(f'<line x1="{lx}" x2="{lx + 34}" y1="{yy - 5}" y2="{yy - 5}" stroke="{th["box"]}" '
                 f'stroke-width="{width}"{dash}/>')
        label = f"ratio {r}" + ("  (used by all crop arms)" if i == 0 else "")
        s.append(f'<text x="{lx + 44}" y="{yy}" font-size="14" fill="{th["text"]}">{label}</text>')
        s.append(f'<text x="{lx + 44}" y="{yy + 24}" font-size="{22 if i == 0 else 17}" font-weight="700" '
                 f'fill="{th["text"]}">{100 * inside:.0f}% inside</text>')

    # colour scale
    cy = FY + 280
    s.append(f'<text x="{lx}" y="{cy}" font-size="13" fill="{th["text2"]}">frames per cell (sqrt scale)</text>')
    bw, bh = 220, 14
    for j in range(60):
        s.append(f'<rect x="{lx + j * bw / 60:.2f}" y="{cy + 10}" width="{bw / 60 + 0.4:.2f}" height="{bh}" '
                 f'fill="{ramp_color(th["ramp"], j / 59)}"/>')
    s.append(f'<text x="{lx}" y="{cy + 42}" font-size="12" fill="{th["text2"]}">few</text>')
    s.append(f'<text x="{lx + bw}" y="{cy + 42}" font-size="12" text-anchor="end" fill="{th["text2"]}">'
             f'most ({peak:,})</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}" role="img" aria-label="Heatmap of gaze points over the frame with centre crop boxes">\n'
            f'<rect width="{W}" height="{H}" fill="{th["surface"]}"/>\n' + "\n".join(s) + "\n</svg>\n")


def main() -> None:
    pts = gaze_points()
    os.makedirs(OUT, exist_ok=True)
    for mode, name in (("light", "gaze_heatmap.svg"), ("dark", "gaze_heatmap_dark.svg")):
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(mode, pts))
        print("wrote", path)


if __name__ == "__main__":
    main()
