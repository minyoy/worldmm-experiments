"""Slide figures: does Qwen3-VL's attention follow the gaze it is given?

(1) bars: attention at the gaze point relative to oracle_full at the same point (middle layers), per method,
    for the real gaze (oracle_*) and for a random point (placebo_*). 1 = no change.
(2) frames: for chosen (question, frame) pairs, the input with the red dot, the attention of full / text /
    mark / saliency on that frame, and condition - full difference maps.

    python plot_attention_evidence.py --dir run_data/gaze_attention_prompt68 --frames 465:27 349:7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle  # noqa: E402
from oracle import base, draw_mark, load_questions, oracle_slots  # noqa: E402
from gaze_attention import gaze_concentration  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"                 # categorical slots 1-2 (dataviz default palette)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGE = LinearSegmentedColormap.from_list("blue_grey_red", ["#2a78d6", "#f0efec", "#e34948"])
MID = range(12, 24)
METHODS = ("text", "mark", "saliency")


def bars(d: Path, out: Path, radius: float):
    vals = {(m, k): [] for m in METHODS for k in ("real", "placebo")}
    recs = sorted(d.glob("[0-9]*.json"))
    for jp in recs:
        r = json.load(open(jp))
        z = np.load(jp.with_suffix(".npz"))
        tags = r["conds"]["oracle_full"]["frame_tags"]
        F = z["oracle_full"].astype(np.float32)
        for m in METHODS:
            for k, cond in (("real", f"oracle_{m}"), ("placebo", f"placebo_{m}")):
                C = z[cond].astype(np.float32)
                pts = r["real_xy"] if k == "real" else r["conds"][cond]["shown_xy"]
                a = [gaze_concentration(C[f, li], pts[t], radius) for li in MID for f, t in enumerate(tags)]
                b = [gaze_concentration(F[f, li], pts[t], radius) for li in MID for f, t in enumerate(tags)]
                vals[(m, k)].append((np.nanmean(a), np.nanmean(b)))
    ratio = {key: np.nanmean([x[0] for x in v]) / np.nanmean([x[1] for x in v]) for key, v in vals.items()}

    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURF)
    ax.set_facecolor(SURF)
    x = np.arange(len(METHODS))
    w = 0.34
    for i, (k, col, lab) in enumerate((("real", BLUE, "at the real gaze point"),
                                        ("placebo", ORANGE, "at a random point (placebo)"))):
        ys = [ratio[(m, k)] for m in METHODS]
        xs = x + (i - 0.5) * (w + 0.02)
        ax.bar(xs, ys, width=w, color=col, label=lab, zorder=3)
        for xi, yi in zip(xs, ys):
            ax.text(xi, yi + 0.04, f"×{yi:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
    ax.axhline(1, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.5, 0.97, "no change", ha="center", va="top", fontsize=9, color=INK2)
    ax.set_xticks(x, ["coordinates as text", "red dot on frame", "saliency map"], fontsize=10, color=INK)
    ax.set_ylabel("attention near the point,\nrelative to no gaze input", fontsize=10, color=INK2)
    ax.set_ylim(0, max(ratio.values()) * 1.2)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(f"Qwen3-VL-8B attention at the answer position ({len(recs)} questions, layers 12-23)",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURF)
    print(out, {f"{m}/{k}": round(v, 2) for (m, k), v in ratio.items()})


def frames_fig(d: Path, picks: list[str], out: Path, args):
    _, _, _, clips = load_questions(args)
    conds = ("oracle_full", "oracle_text", "oracle_mark", "oracle_saliency")
    names = {"oracle_full": "no gaze", "oracle_text": "text", "oracle_mark": "red dot", "oracle_saliency": "saliency map"}
    fig, axes = plt.subplots(len(picks), 8, figsize=(20, 2.9 * len(picks) + 0.4), facecolor=SURF)
    axes = np.atleast_2d(axes)
    for ri, pick in enumerate(picks):
        qid, fr = pick.split(":")
        r = json.load(open(d / f"{qid}.json"))
        z = np.load(d / f"{qid}.npz")
        tags = r["conds"]["oracle_full"]["frame_tags"]
        f = int(fr) - 1
        tag = tags[f]
        key, idx = tag.split(":")
        vr = {k: v for k, v, i, _ in oracle_slots(r["target_keys"], clips, args.video_root)}[key]
        from PIL import Image
        img = np.array(draw_mark(Image.fromarray(vr[int(idx)].asnumpy()).convert("RGB"), *r["real_xy"][tag]))
        H, W = img.shape[:2]
        gx, gy = r["real_xy"][tag]
        maps = {c: z[c][f, MID.start:MID.stop].astype(np.float32).mean(0) for c in conds}
        dist = {c: m / m.sum() for c, m in maps.items()}
        dl = max(float(np.abs(dist[c] - dist["oracle_full"]).max()) for c in conds[1:])

        def show(ax, title, m=None, cmap=None, vmin=None, vmax=None, base_img=True):
            ax.imshow(img if base_img else np.full_like(img, 245))
            if m is not None:
                ax.imshow(m, cmap=cmap, alpha=0.65 if cmap == "inferno" else 0.85, vmin=vmin, vmax=vmax, extent=(0, W, H, 0), interpolation="nearest")
            ax.scatter(gx * W, gy * H, s=170, facecolors="none", edgecolors="#1baf7a", linewidths=2.4)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(title, fontsize=10, color=INK)

        show(axes[ri, 0], "input (red dot condition)")
        axes[ri, 0].set_ylabel(f"Q{qid}, frame {fr}", fontsize=11, color=INK)
        for ci, c in enumerate(conds):
            share = gaze_concentration(maps[c], (gx, gy), 0.1)
            show(axes[ri, ci + 1], f"{names[c]}\nnear gaze ×{share:.2f} of uniform", maps[c] / maps[c].max(), "inferno", 0, 1)
        for ci, c in enumerate(conds[1:]):
            show(axes[ri, ci + 5], f"{names[c]} − no gaze", dist[c] - dist["oracle_full"], DIVERGE, -dl, dl)
    fig.suptitle("Attention of the answer position on one frame (layers 12-23, head mean). Green circle = real gaze. "
                 "Difference maps: red = more attention than with no gaze, blue = less.", fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=170, facecolor=SURF)
    print(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=HERE / "run_data" / "gaze_attention_prompt68")
    ap.add_argument("--frames", nargs="+", default=["465:27", "349:7"], help="qid:frame (1-based)")
    ap.add_argument("--radius", type=float, default=0.1)
    ap.add_argument("--out-dir", type=Path, default=HERE / "run_data" / "gaze_attention_cases")
    ap.add_argument("--pool", type=Path, default=base.ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--video-root", type=Path, default=Path(oracle.VIDEO_ROOT))
    ap.add_argument("--out", type=Path, default=HERE / "run_data")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bars(args.dir, args.out_dir / "evidence_bars.png", args.radius)
    frames_fig(args.dir, args.frames, args.out_dir / "evidence_frames.png", args)


if __name__ == "__main__":
    main()
