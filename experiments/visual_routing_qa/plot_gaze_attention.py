"""Draw the saved last-position attention maps of gaze_attention.py over the frames.

Rows = conditions, columns = frames (evenly picked). Each cell: the frame, the head-averaged attention of the
answer position over that frame's visual tokens (mean of a layer band), the real gaze (green) and, for gaze
conditions, the point shown to the model (red; equal to green except in placebos).

    python plot_gaze_attention.py --qid 379 --dir run_data/gaze_attention_smoke --layers 12-23
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle  # noqa: E402
from oracle import base, load_questions, oracle_slots  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", required=True)
    ap.add_argument("--dir", type=Path, default=HERE / "run_data" / "gaze_attention")
    ap.add_argument("--layers", default="12-23", help="layer band to average, e.g. 12-23")
    ap.add_argument("--frames", type=int, default=6, help="number of frames to show")
    ap.add_argument("--per-frame", action="store_true", help="scale each cell to its own max (shows the pattern)")
    ap.add_argument("--style", choices=("reveal", "heatmap"), default="heatmap",
                    help="reveal: grey image, colour shown where attention is high; heatmap: colour-map overlay")
    ap.add_argument("--dim", type=float, default=0.35, help="reveal: brightness of the grey background")
    ap.add_argument("--gamma", type=float, default=0.5, help="reveal: opacity = (scaled attention)^gamma")
    ap.add_argument("--top", type=float, default=0.15, help="reveal: share of a frame's cells that can light up")
    ap.add_argument("--boost", type=float, default=2.0, help="reveal: brightness of the revealed colour")
    ap.add_argument("--pool", type=Path, default=base.ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--video-root", type=Path, default=Path(oracle.VIDEO_ROOT))
    ap.add_argument("--out", type=Path, default=HERE / "run_data")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    r = json.load(open(args.dir / f"{args.qid}.json"))
    z = np.load(args.dir / f"{args.qid}.npz")
    lo, hi = map(int, args.layers.split("-"))
    rows, _, _, clips = load_questions(args)
    choices = oracle.build_choices(rows[str(args.qid)])
    slots = oracle_slots(r["target_keys"], clips, args.video_root)
    frames = {f"{k}:{i}": vr[i].asnumpy() for k, vr, i, _ in slots}

    conds = list(r["conds"])
    tags = r["conds"][conds[0]]["frame_tags"]
    pick = np.linspace(0, len(tags) - 1, min(args.frames, len(tags))).round().astype(int)
    import textwrap
    wrap = lambda t, w=150: "\n".join(textwrap.wrap(t, w)) or t                     # noqa: E731
    header = [wrap(f"Q{r['ID']} ({r.get('type')}): {r['question']}")]
    header += [wrap(f"{'*' if k == r['answer'] else ' '}({k}) {v}", 150) for k, v in sorted(choices.items())]
    header += [f"gold = {r['answer']}  ·  last-position attention, head mean, layers {lo}-{hi} mean  ·  "
               f"green = real gaze, red = point shown to the model (placebo)"
               f"{'  ·  each cell scaled to its own max' if args.per_frame else ''}"
               f"{f'  ·  colour = top {args.top:.0%} attention cells of the frame (grey = the rest)' if args.style == 'reveal' else ''}"]
    n_lines = sum(h.count("\n") + 1 for h in header)
    vmax = max(float(z[c][:, lo:hi + 1].astype(np.float32).mean(1)[pick].max()) for c in conds)
    has_map = any(f"{c}__map" in z.files for c in conds)
    m0 = 1 if has_map else 0                                       # first column = saliency map (saliency rows only)
    ncol = len(pick) + m0
    fig = plt.figure(figsize=(2.6 * ncol, 2.75 * len(conds) + 0.22 * n_lines + 0.3))
    gs = fig.add_gridspec(len(conds) + 1, ncol, height_ratios=[0.08 * n_lines + 0.1] + [1] * len(conds))
    hax = fig.add_subplot(gs[0, :]); hax.axis("off")
    hax.text(0, 1, "\n".join(header), va="top", ha="left", fontsize=10, family="monospace")
    allax = np.array([[fig.add_subplot(gs[ri + 1, ci]) for ci in range(ncol)] for ri in range(len(conds))])
    axes = allax[:, m0:]
    if has_map:
        from PIL import Image
        first_key = tags[0].split(":")[0]
        size = Image.fromarray(frames[tags[0]]).size
        for ri, cond in enumerate(conds):
            ax = allax[ri, 0]
            ax.set_xticks([]); ax.set_yticks([])
            if f"{cond}__map" not in z.files:
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            c = r["conds"][cond]
            pts = [c["shown_xy"][t] for t in tags if t.startswith(first_key + ":")]
            mimg = np.array(oracle.saliency_map(size, [tuple(p) for p in pts]).convert("RGB"))
            MH, MW = mimg.shape[:2]
            M = z[f"{cond}__map"][0, lo:hi + 1].astype(np.float32).mean(0)
            ax.imshow(mimg)
            ax.imshow(M / M.max() if args.per_frame else M / vmax, cmap="inferno", alpha=0.6, vmin=0, vmax=1,
                      extent=(0, MW, MH, 0), interpolation="nearest")
            col = "#d62728" if cond.startswith("placebo") else "#2ca02c"
            ax.scatter([p[0] * MW for p in pts], [p[1] * MH for p in pts], s=14, facecolors="none", edgecolors=col, linewidths=0.8, alpha=0.8)
            share = float(z[f"{cond}__map"][:, lo:hi + 1].astype(np.float32).sum()) / (
                float(z[f"{cond}__map"][:, lo:hi + 1].astype(np.float32).sum()) + float(z[cond][:, lo:hi + 1].astype(np.float32).sum()))
            ax.set_title(f"saliency map input\n{share:.0%} of visual attn", fontsize=9)
    for ri, cond in enumerate(conds):
        c = r["conds"][cond]
        A = z[cond][:, lo:hi + 1].astype(np.float32).mean(1)              # (frames, gh, gw)
        for ci, f in enumerate(pick):
            ax = axes[ri, ci]
            tag = tags[f]
            img = frames[tag]
            H, W = img.shape[:2]
            m = A[f] / A[f].max() if args.per_frame else A[f] / vmax
            if args.style == "reveal":
                # dim grey everywhere; a cell shows its original colour in proportion to its attention
                gh, gw = m.shape
                # only the frame's top cells light up: opacity ramps from the (1 - top) quantile to the max
                lo_q = np.quantile(m, 1 - args.top)
                a = np.clip((m - lo_q) / max(m.max() - lo_q, 1e-12), 0, 1) ** args.gamma
                a = np.kron(a, np.ones((int(np.ceil(H / gh)), int(np.ceil(W / gw)))))[:H, :W, None]
                grey = img.mean(2, keepdims=True).repeat(3, 2) * args.dim
                lit = np.clip(img.astype(np.float32) * args.boost, 0, 255)
                ax.imshow((grey * (1 - a) + lit * a).astype(np.uint8))
            else:
                ax.imshow(img)
                ax.imshow(m, cmap="inferno", alpha=0.6, vmin=0, vmax=1, extent=(0, W, H, 0), interpolation="nearest")
            gx, gy = r["real_xy"][tag]
            ax.scatter(gx * W, gy * H, s=120, facecolors="none", edgecolors="#2ca02c", linewidths=2.2)
            if c["shown_xy"] and c["shown_xy"][tag] != r["real_xy"][tag]:
                sx, sy = c["shown_xy"][tag]
                ax.scatter(sx * W, sy * H, s=120, facecolors="none", edgecolors="#d62728", linewidths=2.2)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"frame {f + 1}", fontsize=9)
            if ci == 0:
                ax = allax[ri, 0]                                  # label the row on its first column
                orig = HERE / "run_data" / "results" / cond / f"{r['ID']}.json"
                o = base.read_json(orig) if orig.exists() else None
                orig_txt = (f"\noriginal run: {str(o.get('response'))[:12]} {'O' if o.get('correct') else 'X'}" if o else "")
                ax.set_ylabel(f"{cond}\nthis run: {c['pred']} {'O' if c['pred'] == r['answer'] else 'X'}{orig_txt}", fontsize=9)

    out = args.dir / f"{args.qid}_attention_L{lo}-{hi}{'_perframe' if args.per_frame else ''}{'_reveal' if args.style == 'reveal' else ''}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
