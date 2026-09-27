#!/usr/bin/env python3
"""
Recall@k curves for the four crop arms, one panel per target tolerance (tol=60 | tol=0).

Reads results/recall_arms_tol60.json and results/recall_arms_tol0.json and writes a light and a
dark SVG to analysis/ (the README picks one with <picture>). Paper-figure style: bold panel titles,
in-panel legend box, left/bottom axes only, light dashed horizontal grid. Standard library only, so
it runs on the login node. Colors: the reference categorical palette's first three slots (validated
all-pairs in both modes) for the crop arms, neutral gray for the `full` baseline and chance.

    python experiments/gaze_crop/plot_recall_curves.py
"""

import json
import math
import os
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "analysis")

# (arm key in the results file, legend label, light color, dark color)
ARMS = [
    ("gazef_05", "gazef (gaze crop)", "#2a78d6", "#3987e5"),
    ("center_05", "center (centre crop)", "#eb6834", "#d95926"),
    ("gazef_05_randf", "randf (random crop)", "#1baf7a", "#199e70"),
    ("full", "full (no crop)", "#898781", "#898781"),
]
THEMES = {
    "light": {"surface": "#ffffff", "text": "#1a1a1a", "tick": "#333333", "spine": "#333333",
              "grid": "#cfcfcf", "chance": "#b0aea6", "legend_bg": "#ffffff", "legend_border": "#cccccc",
              "k3": "#d9d9d9"},
    "dark": {"surface": "#1a1a19", "text": "#f2f2f2", "tick": "#d4d4d4", "spine": "#bdbdbd",
             "grid": "#3a3a38", "chance": "#6b6a64", "legend_bg": "#1a1a19", "legend_border": "#4a4a47",
             "k3": "#3a3a38"},
}
PANELS = [("recall_arms_tol60.json", "Recall@k  (tol = 60 s)", 50, 10),
          ("recall_arms_tol0.json", "Recall@k  (tol = 0 s, strict)", 30, 5)]
FONT = "'Helvetica Neue', Helvetica, Arial, 'DejaVu Sans', sans-serif"

W, H = 1280, 470
PANEL_W = W / 2
ML, MR, MT, MB = 92, 36, 58, 72      # plot margins inside a panel
MAIN_K = 3


def load(name: str) -> Tuple[List[int], Dict[str, List[float]], List[float]]:
    d = json.load(open(os.path.join(RESULTS, name)))
    ks = [int(k) for k in d["ks"]]
    curves = {a: [100 * d["arms"][a]["recall"][str(k)] for k in ks] for a, *_ in ARMS}
    chance = [100 * d["chance"]["recall"][str(k)] for k in ks]
    return ks, curves, chance


def panel(x0: float, fname: str, title: str, ymax: int, ystep: int, th: Dict[str, str], mode: str) -> List[str]:
    ks, curves, chance = load(fname)
    px0, px1 = x0 + ML, x0 + PANEL_W - MR
    py0, py1 = MT, H - MB
    pad = 22                                              # keep end markers off the spines
    lk0, lk1 = math.log(ks[0]), math.log(ks[-1])

    def X(k: float) -> float:
        return px0 + pad + (math.log(k) - lk0) / (lk1 - lk0) * (px1 - px0 - 2 * pad)

    def Y(v: float) -> float:
        return py1 - pad / 2 - v / ymax * (py1 - py0 - pad / 2)

    s = [f'<text x="{(px0 + px1) / 2:.1f}" y="{MT - 22}" font-size="20" font-weight="700" '
         f'text-anchor="middle" fill="{th["text"]}">{title}</text>']
    # grid + y ticks
    for v in range(0, ymax + 1, ystep):
        y = Y(v)
        s.append(f'<line x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" '
                 f'stroke-width="1" stroke-dasharray="4 4"/>')
        s.append(f'<line x1="{px0 - 6}" x2="{px0}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["spine"]}" stroke-width="1.2"/>')
        s.append(f'<text x="{px0 - 11}" y="{y + 5:.1f}" font-size="15" text-anchor="end" fill="{th["tick"]}">{v}</text>')
    # x ticks
    for k in ks:
        x = X(k)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{py1}" y2="{py1 + 6}" stroke="{th["spine"]}" stroke-width="1.2"/>')
        s.append(f'<text x="{x:.1f}" y="{py1 + 25}" font-size="15" text-anchor="middle" fill="{th["tick"]}">{k}</text>')
    # axis titles
    s.append(f'<text x="{(px0 + px1) / 2:.1f}" y="{py1 + 54}" font-size="17" text-anchor="middle" '
             f'fill="{th["text"]}">k (top-k, log scale)</text>')
    cy = (py0 + py1) / 2
    s.append(f'<text x="{px0 - 52}" y="{cy:.1f}" font-size="17" text-anchor="middle" fill="{th["text"]}" '
             f'transform="rotate(-90 {px0 - 52} {cy:.1f})">Recall (%)</text>')
    # k=3: the depth E' actually consumes (explained in the README caption; a label here collides)
    xk = X(MAIN_K)
    s.append(f'<line x1="{xk:.1f}" x2="{xk:.1f}" y1="{py0}" y2="{py1}" stroke="{th["k3"]}" stroke-width="1.5"/>')
    # spines: left and bottom only
    s.append(f'<line x1="{px0}" x2="{px0}" y1="{py0}" y2="{py1}" stroke="{th["spine"]}" stroke-width="1.2"/>')
    s.append(f'<line x1="{px0}" x2="{px1}" y1="{py1}" y2="{py1}" stroke="{th["spine"]}" stroke-width="1.2"/>')

    # chance reference, then baseline, then crops, hero (gazef) drawn last
    pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(ks, chance))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{th["chance"]}" stroke-width="2" stroke-dasharray="7 5"/>')
    for key, _, cl, cd in reversed(ARMS):
        col = cl if mode == "light" else cd
        pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(ks, curves[key]))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="3" '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for k, v in zip(ks, curves[key]):
            s.append(f'<circle cx="{X(k):.1f}" cy="{Y(v):.1f}" r="6.5" fill="{col}" '
                     f'stroke="{th["surface"]}" stroke-width="1.5"/>')

    # legend box, upper left (the curves rise to the right, so this corner is empty)
    lx, ly, row = px0 + 16, py0 + 8, 27
    items = [(label, cl if mode == "light" else cd, False) for _, label, cl, cd in ARMS]
    items.append(("chance (random ranking)", th["chance"], True))
    bw, bh = 230, 14 + row * len(items)
    s.append(f'<rect x="{lx}" y="{ly}" width="{bw}" height="{bh}" rx="5" fill="{th["legend_bg"]}" '
             f'stroke="{th["legend_border"]}" stroke-width="1"/>')
    for i, (label, col, dashed) in enumerate(items):
        yy = ly + 20 + i * row
        if dashed:
            s.append(f'<line x1="{lx + 12}" x2="{lx + 46}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="2" stroke-dasharray="7 5"/>')
        else:
            s.append(f'<line x1="{lx + 12}" x2="{lx + 46}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="3"/>')
            s.append(f'<circle cx="{lx + 29}" cy="{yy}" r="6.5" fill="{col}" stroke="{th["legend_bg"]}" stroke-width="1.5"/>')
        s.append(f'<text x="{lx + 56}" y="{yy + 5}" font-size="15" fill="{th["text"]}">{label}</text>')
    return s


def render(mode: str) -> str:
    th = THEMES[mode]
    body: List[str] = []
    for i, (fname, title, ymax, ystep) in enumerate(PANELS):
        body += panel(i * PANEL_W, fname, title, ymax, ystep, th, mode)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
            f'font-family="{FONT}" role="img" aria-label="Recall at k by crop arm, tolerance 60 s and 0 s">\n'
            f'<rect width="{W}" height="{H}" fill="{th["surface"]}"/>\n' + "\n".join(body) + "\n</svg>\n")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for mode, name in (("light", "recall_curves.svg"), ("dark", "recall_curves_dark.svg")):
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render(mode))
        print("wrote", path)


if __name__ == "__main__":
    main()
