#!/usr/bin/env python3
"""Eyeball the long fixations: are they real, or is it smooth pursuit glued together?

fixation_sweep.py reports a `>3s` share (6.7% at gap=0.2). Sitting on one spot for three
seconds happens, but so does following a moving object -- I-DT cannot tell the two apart,
because in both cases consecutive samples stay near each other. This draws the long ones so
the difference is visible, and scores it so the drawing is not the only evidence.

The discriminator is the SHAPE of the gaze path inside the fixation:

    straightness = |last - first| / (sum of step lengths)

  ~0.0-0.3  a random walk around one point .......... a real fixation
  ~0.7-1.0  a steady march in one direction ......... smooth pursuit, wrongly merged

A pursuit also shows up in the frames: the thing under the gaze dot stays the same object
while the background slides past it. A real fixation has a still background too.

    python plot_long_fixations.py                       # 6 longest, gap=0.2
    python plot_long_fixations.py --min-dur-shown 5 --sort straightness
    python plot_long_fixations.py --rows 10 --gap 0.1

Needs decord + PIL (the project venv); CPU only, no GPU.
"""

import argparse
import contextlib
import json
import math
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# gaze_points.json, pool.json and the crop helpers live with the crop experiment.
GAZE_CROP = os.path.abspath(os.path.join(HERE, "..", "gaze_crop"))
sys.path.insert(0, HERE)
sys.path.insert(0, GAZE_CROP)
sys.path.insert(0, os.path.join(HERE, "streamgaze"))

from gaze_common import GAZE_PATH, crop_box, load_gaze, load_json  # noqa: E402
from preprocess.gaze_processing import extract_fixation_segments  # noqa: E402

VB = os.path.abspath(os.path.join(HERE, "..", "visual_bottleneck"))
sys.path.insert(0, VB)
from common import VIDEO_ROOT, resolve_video_path  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_DEVNULL = open(os.devnull, "w")

GAZE_COL = (255, 214, 0)        # current gaze -- same yellow as plot_arm_examples.py
TRAIL_COL = (42, 120, 214)      # earlier samples in this fixation
BOX_COL = (42, 120, 214)


def fixations_for(samples, **kw):
    rows = sorted([s for s in samples if float(s[0]) >= 0], key=lambda s: float(s[0]))
    if len(rows) < 2:
        return []
    df = pd.DataFrame({"time_seconds": [float(s[0]) for s in rows],
                       "px": [float(s[1]) for s in rows],
                       "py": [float(s[2]) for s in rows]})
    with contextlib.redirect_stdout(_DEVNULL):
        return extract_fixation_segments(df, **kw)


def samples_in(samples, t0, t1):
    return sorted([(float(t), float(x), float(y)) for t, x, y in samples if t0 <= float(t) <= t1])


def smooth(pts, bin_sec=0.3):
    """Average the samples into `bin_sec` bins. Kills tremor, keeps drift.

    Without this, straightness cannot see pursuit at all: path length is dominated by
    per-sample tracker noise while net displacement is capped by the detector's own radius,
    so every long segment scores near zero whether it drifts or not. Simulated pure pursuit
    through this detector scores 0.00-0.25 on the raw path -- inside the "wandering" band.
    At 10 Hz a 0.3 s bin is 3 samples, short enough to keep a real drift intact.
    """
    if not pts:
        return []
    out, bucket, t0 = [], [], pts[0][0]
    for p in pts:
        if p[0] - t0 > bin_sec and bucket:
            out.append((bucket[0][0], sum(b[1] for b in bucket) / len(bucket),
                        sum(b[2] for b in bucket) / len(bucket)))
            bucket, t0 = [], p[0]
        bucket.append(p)
    if bucket:
        out.append((bucket[0][0], sum(b[1] for b in bucket) / len(bucket),
                    sum(b[2] for b in bucket) / len(bucket)))
    return out


def straightness(pts, bin_sec=0.3):
    """|net displacement| / path length, on the tremor-free path. 1 = a straight march.

    Computed on smooth(pts), not the raw samples -- see smooth() for why the raw version
    is blind to the pursuit it is meant to catch.
    """
    sm = smooth(pts, bin_sec)
    if len(sm) < 3:
        return 0.0
    path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(sm, sm[1:]))
    if path <= 1e-9:
        return 0.0
    net = math.hypot(sm[-1][1] - sm[0][1], sm[-1][2] - sm[0][2])
    return net / path


def collect(gaze, gap, radius, min_dur, dropout, min_shown):
    """Every fixation longer than min_shown, with its samples and straightness."""
    out = []
    for key, entry in gaze.items():
        samples = entry.get("samples") or []
        for f in fixations_for(samples, radius_thresh=radius, duration_thresh=min_dur,
                               gap_thresh=gap, dropout_thresh=dropout):
            if f["duration"] < min_shown:
                continue
            pts = samples_in(samples, f["start_time"], f["end_time"])
            f = dict(f, key=key, pts=pts, straightness=straightness(pts),
                     net=math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2]) if len(pts) > 1 else 0.0)
            out.append(f)
    return out


def main():
    from PIL import Image, ImageDraw, ImageFont
    from decord import VideoReader, cpu

    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze", default=GAZE_PATH)
    ap.add_argument("--pool", default=None, help="pool json with video_path (default: pool_all.json, else pool.json)")
    ap.add_argument("--gap", type=float, default=0.2)
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--min-dur-shown", type=float, default=3.0,
                    help="only fixations at least this long -- the ones the sweep's '>3s' counts")
    ap.add_argument("--rows", type=int, default=6, help="how many fixations to draw")
    ap.add_argument("--cols", type=int, default=6, help="frames per fixation")
    ap.add_argument("--sort", default="duration", choices=["duration", "straightness", "random"],
                    help="duration = the longest; straightness = the most pursuit-like first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=float, default=0.5, help="crop ratio, to draw the box the arm would use")
    ap.add_argument("--cell", type=int, default=300)
    ap.add_argument("--out", default=os.path.join(HERE, "analysis", "figures", "long_fixations.jpg"))
    args = ap.parse_args()

    gaze = load_gaze(args.gaze)
    if not gaze:
        sys.exit(f"no clips in {args.gaze} -- run prepare_gaze.py first")

    pool_path = args.pool or next((p for p in (os.path.join(GAZE_CROP, "pool_all.json"),
                                               os.path.join(GAZE_CROP, "pool.json")) if os.path.exists(p)), None)
    if not pool_path:
        sys.exit("no pool json found; pass --pool")
    pool = load_json(pool_path)
    paths = {c["key"]: c["video_path"] for c in (pool["clips"] if isinstance(pool, dict) else pool)}

    print(f"scanning {len(gaze)} clips (gap={args.gap} radius={args.radius}) ...", flush=True)
    longs = collect(gaze, args.gap, args.radius, args.min_dur, args.dropout, args.min_dur_shown)
    if not longs:
        sys.exit(f"no fixation longer than {args.min_dur_shown}s")

    # The population-level answer, before any picture: if these were mostly pursuit, the
    # straightness distribution would sit high.
    ss = sorted(f["straightness"] for f in longs)
    q = lambda p: ss[min(len(ss) - 1, int(p * len(ss)))]
    print(f"\n{len(longs)} fixations > {args.min_dur_shown}s")
    print(f"  straightness  p10={q(.10):.2f}  median={q(.50):.2f}  p90={q(.90):.2f}")
    print(f"  pursuit-like (>= 0.30): {sum(1 for s in ss if s >= 0.30)} "
          f"({100 * sum(1 for s in ss if s >= 0.30) / len(ss):.1f}%)")
    print(f"  fixation     (<= 0.15): {sum(1 for s in ss if s <= 0.15)} "
          f"({100 * sum(1 for s in ss if s <= 0.15) / len(ss):.1f}%)")
    print(f"  ambiguous            : {sum(1 for s in ss if 0.15 < s < 0.30)} "
          f"({100 * sum(1 for s in ss if 0.15 < s < 0.30) / len(ss):.1f}%)")
    print("  (no-drift control scores 0.05; see straightness() for the calibration)\n", flush=True)

    if args.sort == "duration":
        longs.sort(key=lambda f: -f["duration"])
    elif args.sort == "straightness":
        longs.sort(key=lambda f: -f["straightness"])
    else:
        import random
        random.Random(args.seed).shuffle(longs)

    picked, seen = [], set()
    for f in longs:                       # one per clip, so the figure is not six views of one moment
        if f["key"] in seen or f["key"] not in paths:
            continue
        seen.add(f["key"])
        picked.append(f)
        if len(picked) == args.rows:
            break

    cell, gap_px, left, top = args.cell, 12, 300, 54
    Wd = left + args.cols * (cell + gap_px) + 10
    Ht = top + len(picked) * (cell + gap_px) + 56
    fig = Image.new("RGB", (Wd, Ht), "white")
    d = ImageDraw.Draw(fig)
    f_b = ImageFont.truetype(FONT_B, 19)
    f_n = ImageFont.truetype(FONT, 15)
    f_s = ImageFont.truetype(FONT, 13)

    d.text((left, top - 30), "frames spanning the fixation, left to right", font=f_b, fill=(26, 26, 26))

    for ri, f in enumerate(picked):
        y = top + ri * (cell + gap_px)
        vr = VideoReader(resolve_video_path(paths[f["key"]], VIDEO_ROOT), ctx=cpu(0), num_threads=4)
        fps = float(vr.get_avg_fps()) or 30.0
        times = [f["start_time"] + (f["end_time"] - f["start_time"]) * i / (args.cols - 1)
                 for i in range(args.cols)]
        idx = [max(0, min(len(vr) - 1, int(round(t * fps)))) for t in times]
        arr = vr.get_batch(idx).asnumpy()

        sn = f["straightness"]
        verdict = "PURSUIT?" if sn >= 0.30 else ("fixation" if sn <= 0.15 else "ambiguous")
        d.text((18, y + 22), f["key"], font=f_n, fill=(26, 26, 26))
        d.text((18, y + 46), f"{f['start_time']:.1f}-{f['end_time']:.1f}s  ({f['duration']:.1f}s, "
                             f"{len(f['pts'])} samples)", font=f_s, fill=(85, 85, 85))
        d.text((18, y + 72), f"straightness {f['straightness']:.2f}  ->  {verdict}", font=f_s,
               fill=(200, 60, 40) if sn >= 0.30 else (60, 60, 60))
        d.text((18, y + 94), f"net drift {f['net']:.3f} of frame", font=f_s, fill=(85, 85, 85))

        for ci, (a, t) in enumerate(zip(arr, times)):
            frame = Image.fromarray(a).convert("RGB")
            w, h = frame.size
            # the box a fixation-aware arm would crop: centred on the fixation centroid, fixed
            # for the whole fixation
            box = crop_box(w, h, f["center_x"], f["center_y"], args.ratio)
            dim = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), 0.55)
            dim.paste(frame.crop(box), box[:2])
            thumb = dim.resize((cell, cell), Image.LANCZOS)
            td = ImageDraw.Draw(thumb)
            sc = cell / w
            td.rectangle([round(v * sc) for v in box], outline=BOX_COL, width=3)

            # the gaze path so far: earlier samples small, the current one big. A pursuit draws
            # a line across the panel; a fixation draws a blob.
            trail = [p for p in f["pts"] if p[0] <= t]
            for p in trail[:-1]:
                px, py = p[1] * cell, p[2] * cell
                td.ellipse([px - 3, py - 3, px + 3, py + 3], fill=TRAIL_COL)
            if trail:
                px, py = trail[-1][1] * cell, trail[-1][2] * cell
                td.ellipse([px - 8, py - 8, px + 8, py + 8], fill=GAZE_COL, outline=(0, 0, 0), width=2)
            # the radius the detector allowed, around the centroid
            cx, cy = f["center_x"] * cell, f["center_y"] * cell
            r = args.radius * cell
            td.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 255), width=2)
            td.text((6, 4), f"{t:.1f}s", font=f_s, fill=(255, 255, 255))
            fig.paste(thumb, (left + ci * (cell + gap_px), y))

    ly = Ht - 30
    d.ellipse([left, ly - 8, left + 16, ly + 8], fill=GAZE_COL, outline=(0, 0, 0), width=2)
    d.text((left + 26, ly), "gaze now;", font=f_s, fill=(60, 60, 60), anchor="lm")
    d.ellipse([left + 100, ly - 4, left + 108, ly + 4], fill=TRAIL_COL)
    d.text((left + 116, ly), "earlier samples in this fixation;  white circle = the detector's "
                             f"radius ({args.radius}) around the centroid;  blue box = the {args.ratio} "
                             "crop a fixation-aware arm would hold for the whole fixation.",
           font=f_s, fill=(60, 60, 60), anchor="lm")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.save(args.out, quality=88)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
