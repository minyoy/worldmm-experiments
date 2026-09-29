#!/usr/bin/env python3
"""
Three figures for the README, each as a light and a dark SVG (the README picks one with <picture>):

  analysis/figures/recall_at_k_curves{,_dark}.svg  Recall@k (k <= 20), 3 rows x 3 tolerances (0 | 30 | 60 s).
                                             A = frame selection   (full / fixsel_ctl / fixsel)
                                             B = selection + crop  (full / fixsel / fixsel_gaze / fixsel_gazef)
                                             C = gaze crop vs centre (full / center / gazef / fixsel_gazef)
  analysis/figures/paired_diff_recall_at_3{,_dark}.svg             k=3 paired differences with 95% CI, one row per contrast,
                                             one mark per tolerance.
  analysis/figures/recall_at_k_fixsel_vs_gazef{,_dark}.svg  fixsel (frame selection only) vs gazef (gaze crop), with full and chance
  analysis/figures/recall_diff_ci_fixsel_gazef{,_dark}.svg             recall@k difference of fixsel_gazef vs gazef and vs center,
                                             with a 95% bootstrap CI over questions.

Reads results/recall_pool_all_subset_r005_tol{0,30,60}.json. The tol=0 run did not score
`center_05`; its column comes from ../gaze_crop/results/recall_arms_tol0.json, which is the same
pool, the same embeddings and the same 487 questions -- checked below question by question on the
two arms both files hold, and the script refuses to merge if they disagree. Standard library only.

    python experiments/fixation_frame_selection/plot_recall_curves.py
"""

import json
import math
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "analysis", "figures")
GC_TOL0 = os.path.join(HERE, "..", "gaze_crop", "results", "recall_arms_tol0.json")

TOLS = [0, 30, 60]
MAIN_K = 3
FONT = "'Helvetica Neue', Helvetica, Arial, 'DejaVu Sans', sans-serif"

# arm -> (legend label, light, dark, dashed, hollow marker)
# Every arm has its own hue so curves that overlap stay distinguishable; solid + filled = the main arm,
# dashed + hollow = its control or variant. `full` is the neutral baseline (gray, hollow markers).
STYLE = {
    "full": ("full (16 frames, no crop)", "#6b6a64", "#a3a29a", False, True),
    "fixsel_ctl": ("fixsel_ctl (same count, uniform)", "#d6479a", "#e879b6", True, True),
    "fixsel": ("fixsel (fixation frames)", "#4a3aa7", "#9085e9", False, False),
    "center_05": ("center (centre crop)", "#eb6834", "#d95926", False, False),
    "gazef_05": ("gazef (gaze crop)", "#2a78d6", "#3987e5", False, False),
    "fixsel_gaze_05": ("fixsel_gaze (centroid crop)", "#c98a00", "#e0a92e", True, True),
    "fixsel_gazef_05": ("fixsel_gazef (per-frame crop)", "#1baf7a", "#2fd19a", False, False),
}
ROWS = [("A. frame selection", ["full", "fixsel_ctl", "fixsel"]),
        ("B. selection + crop", ["full", "fixsel", "fixsel_gaze_05", "fixsel_gazef_05"]),
        ("C. gaze crop vs center", ["full", "center_05", "gazef_05", "fixsel_gazef_05"])]
# frame selection alone vs the gaze crop, one panel per tolerance
FIXSEL_VS_GAZEF = [("fixsel vs gazef", ["full", "fixsel", "gazef_05"])]
YSCALE = {0: (20, 5), 30: (30, 5), 60: (35, 5)}   # per column, shared by all rows (k <= 20)

THEMES = {
    "light": {"surface": "#ffffff", "text": "#1a1a1a", "muted": "#555555", "tick": "#333333",
              "spine": "#333333", "grid": "#cfcfcf", "chance": "#b0aea6", "legend_bg": "#ffffff",
              "legend_border": "#cccccc", "k3": "#d9d9d9", "band": "#f3f3f1",
              "tol": ["#9ec5f4", "#2a78d6", "#104281"]},
    "dark": {"surface": "#1a1a19", "text": "#f2f2f2", "muted": "#b5b4ad", "tick": "#d4d4d4",
             "spine": "#bdbdbd", "grid": "#3a3a38", "chance": "#6b6a64", "legend_bg": "#1a1a19",
             "legend_border": "#4a4a47", "k3": "#3a3a38", "band": "#232322",
             "tol": ["#1c5cab", "#3987e5", "#9ec5f4"]},
}
TOL_SHAPE = ["circle", "square", "diamond"]


# ---------------------------------------------------------------- data

def load_all() -> Dict[int, dict]:
    """{tol: {"ks", "chance", "recall": {arm: [..]}, "ranks": {qid: {arm: rank}}}}"""
    out = {}
    for t in TOLS:
        d = json.load(open(os.path.join(RESULTS, f"recall_pool_all_subset_r005_tol{t}.json")))
        assert int(d["target_tolerance_sec"]) == t, (t, d["target_tolerance_sec"])
        ks = [int(k) for k in d["ks"]]
        out[t] = {
            "ks": ks,
            "n": d["n_scored"],
            "chance": [100 * d["chance"]["recall"][str(k)] for k in ks],
            "recall": {a: [100 * v["recall"][str(k)] for k in ks] for a, v in d["arms"].items()},
            "ranks": {q["ID"]: dict(q["rank"]) for q in d["per_question"]},
        }
    t0 = out[0]
    if "center_05" not in t0["recall"]:
        g = json.load(open(GC_TOL0))
        granks = {q["ID"]: q["rank"] for q in g["per_question"]}
        same_q = set(granks) == set(t0["ranks"])
        same_r = same_q and all(granks[q][a] == t0["ranks"][q][a]
                                for q in t0["ranks"] for a in ("full", "gazef_05"))
        if not (same_q and same_r and int(g["target_tolerance_sec"]) == 0):
            raise SystemExit(f"{GC_TOL0} does not match the tol=0 run (questions or ranks differ); "
                             "rescore tol=0 with center@0.5 instead of merging.")
        t0["recall"]["center_05"] = [100 * g["arms"]["center_05"]["recall"][str(k)] for k in t0["ks"]]
        for q, r in t0["ranks"].items():
            r["center_05"] = granks[q]["center_05"]
    return out


def scored(data: Dict[int, dict], arm: str) -> bool:
    """An arm is drawn only when every tolerance scored it (fixsel_gazef until it has been run)."""
    return all(arm in data[t]["recall"] for t in TOLS)


def paired(ranks: Dict[str, Dict[str, int]], a: str, b: str, k: int = MAIN_K):
    """diff in pp, 95% Wald CI for a paired difference of proportions, a-only, b-only, McNemar p"""
    n = len(ranks)
    ao = sum(r[a] <= k < r[b] for r in ranks.values())
    bo = sum(r[b] <= k < r[a] for r in ranks.values())
    d = (ao - bo) / n
    se = math.sqrt(max(ao + bo - (ao - bo) ** 2 / n, 0)) / n
    m = ao + bo
    p = 1.0 if m == 0 else min(1.0, 2 * sum(math.comb(m, i) for i in range(min(ao, bo) + 1)) / 2 ** m)
    return 100 * d, 100 * (d - 1.96 * se), 100 * (d + 1.96 * se), ao, bo, p


# ---------------------------------------------------------------- shared drawing

def marker(shape: str, x: float, y: float, r: float, fill: str, stroke: str, sw: float = 1.5) -> str:
    if shape == "square":
        return (f'<rect x="{x - r * .85:.1f}" y="{y - r * .85:.1f}" width="{r * 1.7:.1f}" height="{r * 1.7:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    if shape == "diamond":
        rr = r * 1.15
        return (f'<polygon points="{x:.1f},{y - rr:.1f} {x + rr:.1f},{y:.1f} {x:.1f},{y + rr:.1f} {x - rr:.1f},{y:.1f}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def series_mark(arm: str, x: float, y: float, th: Dict[str, str], mode: str, bg: str) -> str:
    _, cl, cd, _, hollow = STYLE[arm]
    col = cl if mode == "light" else cd
    if hollow:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{bg}" stroke="{col}" stroke-width="2.5"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{col}" stroke="{bg}" stroke-width="1.5"/>'


def svg(w: int, h: int, label: str, th: Dict[str, str], body: List[str]) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{FONT}" role="img" aria-label="{label}">\n'
            f'<rect width="{w}" height="{h}" fill="{th["surface"]}"/>\n' + "\n".join(body) + "\n</svg>\n")


# ---------------------------------------------------------------- figure 1: recall curves

CW, CH = 480, 400                      # one panel
ML, MR, MT, MB = 78, 24, 52, 66
ROW_HEAD = 58                          # row title strip above each row (legend may take 2 lines)


def curve_panel(x0: float, y0: float, data: dict, t: int, arms: List[str],
                th: Dict[str, str], mode: str) -> List[str]:
    ks, chance = data["ks"], data["chance"]
    ymax, ystep = YSCALE[t]
    px0, px1, py0, py1 = x0 + ML, x0 + CW - MR, y0 + MT, y0 + CH - MB
    pad = 20
    lk0, lk1 = math.log(ks[0]), math.log(ks[-1])

    def X(k):
        return px0 + pad + (math.log(k) - lk0) / (lk1 - lk0) * (px1 - px0 - 2 * pad)

    def Y(v):
        return py1 - pad / 2 - v / ymax * (py1 - py0 - pad / 2)

    title = f"tol = {t} s" + ("  (strict)" if t == 0 else "")
    s = [f'<text x="{(px0 + px1) / 2:.1f}" y="{py0 - 18}" font-size="18" font-weight="700" '
         f'text-anchor="middle" fill="{th["text"]}">{title}</text>']
    for v in range(0, ymax + 1, ystep):
        y = Y(v)
        s.append(f'<line x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" stroke-dasharray="4 4"/>')
        s.append(f'<text x="{px0 - 10}" y="{y + 5:.1f}" font-size="14" text-anchor="end" fill="{th["tick"]}">{v}</text>')
    for k in ks:
        x = X(k)
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{py1}" y2="{py1 + 6}" stroke="{th["spine"]}" stroke-width="1.2"/>')
        s.append(f'<text x="{x:.1f}" y="{py1 + 23}" font-size="14" text-anchor="middle" fill="{th["tick"]}">{k}</text>')
    s.append(f'<text x="{(px0 + px1) / 2:.1f}" y="{py1 + 50}" font-size="15" text-anchor="middle" '
             f'fill="{th["text"]}">k (log scale)</text>')
    cy = (py0 + py1) / 2
    s.append(f'<text x="{px0 - 46}" y="{cy:.1f}" font-size="15" text-anchor="middle" fill="{th["text"]}" '
             f'transform="rotate(-90 {px0 - 46} {cy:.1f})">Recall (%)</text>')
    xk = X(MAIN_K)
    s.append(f'<line x1="{xk:.1f}" x2="{xk:.1f}" y1="{py0}" y2="{py1}" stroke="{th["k3"]}" stroke-width="1.5"/>')
    s.append(f'<line x1="{px0}" x2="{px0}" y1="{py0}" y2="{py1}" stroke="{th["spine"]}" stroke-width="1.2"/>')
    s.append(f'<line x1="{px0}" x2="{px1}" y1="{py1}" y2="{py1}" stroke="{th["spine"]}" stroke-width="1.2"/>')

    pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(ks, chance))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{th["chance"]}" stroke-width="2" stroke-dasharray="7 5"/>')
    for arm in arms:                                   # baseline first, the row's hero last
        _, cl, cd, dashed, _ = STYLE[arm]
        col = cl if mode == "light" else cd
        ys = data["recall"][arm]
        pts = " ".join(f"{X(k):.1f},{Y(v):.1f}" for k, v in zip(ks, ys))
        dash = ' stroke-dasharray="8 5"' if dashed else ""
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"{dash} '
                 f'stroke-linejoin="round" stroke-linecap="round"/>')
        for k, v in zip(ks, ys):
            s.append(series_mark(arm, X(k), Y(v), th, mode, th["surface"]))

    return s


def row_legend(x: float, y: float, arms: List[str], th: Dict[str, str], mode: str,
               max_x: float) -> List[str]:
    """One horizontal legend per row, in the row's title strip: in-panel boxes cover the curves.
    Wraps to a second line when the row has too many arms for one."""
    s = []
    x0 = x
    for a in list(reversed(arms)) + ["chance"]:
        label = "chance (random ranking)" if a == "chance" else STYLE[a][0]
        if x + 48 + 7.4 * len(label) > max_x:
            x, y = x0, y + 24
        if a == "chance":
            label = "chance (random ranking)"
            s.append(f'<line x1="{x}" x2="{x + 40}" y1="{y}" y2="{y}" stroke="{th["chance"]}" '
                     f'stroke-width="2" stroke-dasharray="7 5"/>')
        else:
            label, cl, cd, dashed, _ = STYLE[a]
            col = cl if mode == "light" else cd
            dash = ' stroke-dasharray="8 5"' if dashed else ""
            s.append(f'<line x1="{x}" x2="{x + 40}" y1="{y}" y2="{y}" stroke="{col}" stroke-width="2.5"{dash}/>')
            s.append(series_mark(a, x + 20, y, th, mode, th["surface"]))
        s.append(f'<text x="{x + 48}" y="{y + 5}" font-size="14" fill="{th["text"]}">{label}</text>')
        x += 48 + 7.4 * len(label) + 28
    return s


def render_curves(data: Dict[int, dict], mode: str, rows=None) -> str:
    rows = rows or ROWS
    th = THEMES[mode]
    w, h = CW * len(TOLS), (CH + ROW_HEAD) * len(rows)
    body: List[str] = []
    for r, (rtitle, arms) in enumerate(rows):
        arms = [a for a in arms if scored(data, a)]
        y0 = r * (CH + ROW_HEAD)
        body.append(f'<text x="16" y="{y0 + 24}" font-size="19" font-weight="700" fill="{th["text"]}">{rtitle}</text>')
        body += row_legend(250, y0 + 18, arms, th, mode, w - 16)
        for c, t in enumerate(TOLS):
            body += curve_panel(c * CW, y0 + ROW_HEAD, data[t], t, arms, th, mode)
    return svg(w, h, "Recall at k by arm, rows frame selection and crop, columns tolerance 0, 30, 60 s", th, body)


# ---------------------------------------------------------------- figure 4: recall difference with bootstrap CI

CI_KS = [1, 3, 5, 10, 20]
CI_CONTRASTS = [("fixsel_gazef_05", "gazef_05", "fixsel_gazef − gazef", "selection on top of the gaze crop"),
                ("fixsel_gazef_05", "center_05", "fixsel_gazef − center", "selection + gaze crop vs the centre crop")]
N_BOOT = 4000


def boot_diff(ranks: Dict[str, Dict[str, int]], a: str, b: str, k: int, seed: int = 0):
    """Paired difference in recall@k (pp) with a 95% percentile bootstrap CI, resampling questions."""
    import random
    d = [(r[a] <= k) - (r[b] <= k) for r in ranks.values()]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(d, k=n)) / n for _ in range(N_BOOT))
    return 100 * sum(d) / n, 100 * means[int(0.025 * N_BOOT)], 100 * means[int(0.975 * N_BOOT) - 1]


def render_ci(data: Dict[int, dict], mode: str) -> str:
    th = THEMES[mode]
    pw, ph = 480, 330
    ml, mr, mt, mb = 78, 24, 52, 58
    rows = len(CI_CONTRASTS)
    w, h = pw * len(TOLS), (ph + 40) * rows
    body: List[str] = []
    ylo, yhi, ystep = -4, 8, 2
    kx = {k: i for i, k in enumerate(CI_KS)}
    for r, (a, b, name, role) in enumerate(CI_CONTRASTS):
        y0 = r * (ph + 40)
        body.append(f'<text x="16" y="{y0 + 24}" font-size="19" font-weight="700" fill="{th["text"]}">{name}</text>')
        body.append(f'<text x="{16 + 12 * len(name) + 12}" y="{y0 + 24}" font-size="14" fill="{th["muted"]}">'
                    f'{role} · recall@k difference (pp), 95% bootstrap CI</text>')
        for c, t in enumerate(TOLS):
            px0, px1 = c * pw + ml, c * pw + pw - mr
            py0, py1 = y0 + 40 + mt - 20, y0 + 40 + ph - mb

            def X(k):
                return px0 + 30 + kx[k] / (len(CI_KS) - 1) * (px1 - px0 - 60)

            def Y(v):
                return py1 - (v - ylo) / (yhi - ylo) * (py1 - py0)

            body.append(f'<text x="{(px0 + px1) / 2:.1f}" y="{py0 - 14}" font-size="18" font-weight="700" '
                        f'text-anchor="middle" fill="{th["text"]}">tol = {t} s{"  (strict)" if t == 0 else ""}</text>')
            v = ylo
            while v <= yhi:
                y = Y(v)
                if v == 0:
                    body.append(f'<line x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["spine"]}" stroke-width="1.6"/>')
                else:
                    body.append(f'<line x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{th["grid"]}" stroke-dasharray="4 4"/>')
                body.append(f'<text x="{px0 - 10}" y="{y + 5:.1f}" font-size="14" text-anchor="end" fill="{th["tick"]}">'
                            f'{f"{v:+d}" if v else "0"}</text>')
                v += ystep
            for k in CI_KS:
                body.append(f'<text x="{X(k):.1f}" y="{py1 + 22}" font-size="14" text-anchor="middle" fill="{th["tick"]}">{k}</text>')
            body.append(f'<text x="{(px0 + px1) / 2:.1f}" y="{py1 + 46}" font-size="15" text-anchor="middle" fill="{th["text"]}">k</text>')
            cy = (py0 + py1) / 2
            body.append(f'<text x="{px0 - 46}" y="{cy:.1f}" font-size="15" text-anchor="middle" fill="{th["text"]}" '
                        f'transform="rotate(-90 {px0 - 46} {cy:.1f})">Recall diff (pp)</text>')
            body.append(f'<line x1="{px0}" x2="{px0}" y1="{py0}" y2="{py1}" stroke="{th["spine"]}" stroke-width="1.2"/>')
            col = STYLE[a][1] if mode == "light" else STYLE[a][2]
            for k in CI_KS:
                d, lo, hi = boot_diff(data[t]["ranks"], a, b, k)
                sig = lo > 0 or hi < 0
                body.append(f'<line x1="{X(k):.1f}" x2="{X(k):.1f}" y1="{Y(max(lo, ylo)):.1f}" y2="{Y(min(hi, yhi)):.1f}" '
                            f'stroke="{col}" stroke-width="3" stroke-linecap="round"/>')
                fill = col if sig else th["surface"]
                body.append(f'<circle cx="{X(k):.1f}" cy="{Y(d):.1f}" r="6.5" fill="{fill}" stroke="{col}" stroke-width="2.5"/>')
    body.append(f'<text x="{w - 16}" y="24" font-size="13" text-anchor="end" fill="{th["muted"]}">'
                f'filled = 95% CI excludes 0 · hollow = includes 0</text>')
    return svg(w, h, "Recall difference with bootstrap 95% CI for fixsel_gazef vs gazef and vs center", th, body)


# ---------------------------------------------------------------- figure 2: k=3 paired differences

CONTRASTS = [
    ("fixsel", "full", "fixsel − full", "selection + fewer frames"),
    ("fixsel", "fixsel_ctl", "fixsel − fixsel_ctl", "primary: selection effect"),
    ("fixsel", "gazef_05", "fixsel − gazef", "selection alone vs gaze crop"),
    ("fixsel_gazef_05", "gazef_05", "fixsel_gazef − gazef", "selection on top of gaze crop"),
]
FW = 1280
F_LAB, F_PLOT0, F_PLOT1 = 20, 330, 900    # label column, plot x-range; stats column after it
F_TOP, F_ROW, F_GAP = 70, 34, 26          # per tolerance row height, gap between contrasts
XMIN, XMAX, XSTEP = -6, 8, 2


def render_paired(data: Dict[int, dict], mode: str) -> str:
    th = THEMES[mode]
    contrasts = [c for c in CONTRASTS if scored(data, c[0]) and scored(data, c[1])]
    h = F_TOP + len(contrasts) * (len(TOLS) * F_ROW + F_GAP) + 50

    def X(v):
        return F_PLOT0 + (v - XMIN) / (XMAX - XMIN) * (F_PLOT1 - F_PLOT0)

    s = [f'<text x="{F_LAB}" y="30" font-size="19" font-weight="700" fill="{th["text"]}">'
         f'Paired difference in Recall@3 (pp), 95% CI</text>',
         f'<text x="{F_PLOT1 + 24}" y="30" font-size="14" fill="{th["muted"]}">a only : b only · McNemar p</text>']
    # tolerance legend (shape + shade, never shade alone)
    lx = F_PLOT0
    for i, t in enumerate(TOLS):
        x = lx + i * 120
        s.append(marker(TOL_SHAPE[i], x + 8, 52, 6, th["tol"][i], th["surface"]))
        s.append(f'<text x="{x + 22}" y="57" font-size="14" fill="{th["text"]}">tol {t} s</text>')

    y_end = h - 50
    # the primary contrast gets a band, drawn first so the grid shows through it
    block = len(TOLS) * F_ROW
    for ci, c in enumerate(contrasts):
        if c[3].startswith("primary"):
            s.append(f'<rect x="{F_LAB - 8}" y="{F_TOP + ci * (block + F_GAP) - 4}" '
                     f'width="{FW - 2 * F_LAB + 16}" height="{block + 8}" rx="6" fill="{th["band"]}"/>')
    for v in range(XMIN, XMAX + 1, XSTEP):
        x = X(v)
        if v == 0:
            s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{F_TOP}" y2="{y_end}" stroke="{th["spine"]}" stroke-width="1.4"/>')
        else:
            s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{F_TOP}" y2="{y_end}" stroke="{th["grid"]}" '
                     f'stroke-dasharray="4 4"/>')
        s.append(f'<text x="{x:.1f}" y="{y_end + 22}" font-size="14" text-anchor="middle" '
                 f'fill="{th["tick"]}">{f"{v:+d}" if v else "0"}</text>')
    s.append(f'<text x="{X(XMIN) + 4:.1f}" y="{y_end + 42}" font-size="13" fill="{th["muted"]}">← b better</text>')
    s.append(f'<text x="{X(XMAX) - 4:.1f}" y="{y_end + 42}" font-size="13" text-anchor="end" '
             f'fill="{th["muted"]}">a better →</text>')

    y = F_TOP
    for ci, (a, b, name, role) in enumerate(contrasts):
        block = len(TOLS) * F_ROW
        s.append(f'<text x="{F_LAB}" y="{y + block / 2 - 3:.1f}" font-size="16" font-weight="700" '
                 f'fill="{th["text"]}">{name}</text>')
        s.append(f'<text x="{F_LAB}" y="{y + block / 2 + 17:.1f}" font-size="13" fill="{th["muted"]}">{role}</text>')
        for i, t in enumerate(TOLS):
            yy = y + F_ROW * i + F_ROW / 2
            d, lo, hi, ao, bo, p = paired(data[t]["ranks"], a, b)
            col = th["tol"][i]
            s.append(f'<line x1="{X(max(lo, XMIN)):.1f}" x2="{X(min(hi, XMAX)):.1f}" y1="{yy:.1f}" y2="{yy:.1f}" '
                     f'stroke="{col}" stroke-width="2.5" stroke-linecap="round"/>')
            s.append(marker(TOL_SHAPE[i], X(d), yy, 7, col, th["surface"]))
            ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
            weight = ' font-weight="700"' if p < 0.05 else ""
            s.append(f'<text x="{F_PLOT1 + 24}" y="{yy + 5:.1f}" font-size="14" fill="{th["text"]}"{weight}>'
                     f'{d:+.1f} pp   {ao}:{bo}   {ptxt}</text>')
        y += block + F_GAP
    return svg(FW, h, "Paired Recall@3 differences with 95% CI across tolerances 0, 30, 60 s", th, s)


KMAX = 20      # curves stop at k=20: k=50 is 0.8% of the pool and far from the top-3 the system uses.
               # The tables keep k=50, and paired_diff_recall_at_3 is a k=3 figure, so it is unaffected.


def cut_k(data: Dict[int, dict], kmax: int = KMAX) -> Dict[int, dict]:
    n = {t: sum(k <= kmax for k in d["ks"]) for t, d in data.items()}
    return {t: dict(d, ks=d["ks"][:n[t]], chance=d["chance"][:n[t]],
                    recall={a: v[:n[t]] for a, v in d["recall"].items()})
            for t, d in data.items()}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    data = load_all()
    cut = cut_k(data)
    for mode, suf in (("light", ""), ("dark", "_dark")):
        for name, fn, d in (("recall_at_k_curves", render_curves, cut), ("paired_diff_recall_at_3", render_paired, data),
                            ("recall_diff_ci_fixsel_gazef", render_ci, data),
                            ("recall_at_k_fixsel_vs_gazef", lambda dd, m: render_curves(dd, m, FIXSEL_VS_GAZEF), cut)):
            path = os.path.join(OUT, f"{name}{suf}.svg")
            with open(path, "w", encoding="utf-8") as f:
                f.write(fn(d, mode))
            print("wrote", path)


if __name__ == "__main__":
    main()
