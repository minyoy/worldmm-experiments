#!/usr/bin/env python3
"""Sweep the fixation detector's gap tolerance over the real 10 Hz gaze samples.

Answers the question the threshold choice actually turns on: at 10 Hz, gap_thresh is a
sample-count knob (0.1 / 0.2 / 0.3 s = 1 / 2 / 3 samples), and the way to tell splitting
from merging is what happens to fixation count, duration and coverage as it grows.

    (a) fixations per clip          -- a big drop means the smaller value was splitting
    (b) median fixation duration    -- rises as splitting stops, keeps rising if merging starts
    (c) frame coverage              -- share of the 16 sampled frame times that land inside a
                                       fixation; this is the number that decides whether a
                                       fixation-aware crop arm is even worth building
    (d) long-fixation share         -- fixations > 3 s. The over-merge alarm: real fixations in
                                       this footage are rarely that long, so a rising tail here
                                       means two fixations got glued together.

Reads gaze_points.json (prepare_gaze.py's output), not the raw CSVs.

    python fixation_sweep.py --gaze gaze_points.json
    python fixation_sweep.py --gap 0.1,0.2,0.3 --radius 0.05 --min-dur 0.3
"""

import argparse
import contextlib
import os
import statistics
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# gaze_points.json and its loader live with the crop experiment; this one only reads them.
GAZE_CROP = os.path.abspath(os.path.join(HERE, "..", "gaze_crop"))
sys.path.insert(0, HERE)
sys.path.insert(0, GAZE_CROP)
sys.path.insert(0, os.path.join(HERE, "streamgaze"))

from gaze_common import GAZE_PATH, NFRAMES, load_gaze  # noqa: E402
from preprocess.gaze_processing import extract_fixation_segments  # noqa: E402


_DEVNULL = open(os.devnull, "w")


def clip_fixations(samples, **kw):
    """Run the detector on one clip's samples. Returns [] if there is nothing usable."""
    rows = [s for s in samples if float(s[0]) >= 0]
    if len(rows) < 2:
        return []
    df = pd.DataFrame({
        "time_seconds": [float(s[0]) for s in rows],
        "px": [float(s[1]) for s in rows],
        "py": [float(s[2]) for s in rows],
    }).sort_values("time_seconds")
    return extract_fixation_segments(df, **kw)


def frame_times(samples, nframes=NFRAMES):
    """The nframes instants embed_arms.py samples a clip at, from the samples' own span.

    embed_arms uses the decoded clip duration; gaze_points.json does not carry it, so the
    last sample's timestamp stands in. On 30 s clips at 10 Hz the two differ by <0.1 s,
    which cannot move a frame in or out of a fixation by more than one sample.
    """
    ts = [float(s[0]) for s in samples if float(s[0]) >= 0]
    if not ts:
        return []
    dur = max(ts)
    return [dur * i / (nframes - 1) for i in range(nframes)]


def summarise(gaze, gap, radius, min_dur, dropout, long_sec=3.0, progress=None):
    per_clip, durs, covered, total_frames, clips_with_none = [], [], 0, 0, 0

    for i, (key, entry) in enumerate(gaze.items(), 1):
        samples = entry.get("samples") or []
        # The detector prints a line per clip. Discard it here rather than buffering it --
        # this loop runs once per gap value over every clip.
        with contextlib.redirect_stdout(_DEVNULL):
            fixes = clip_fixations(samples, radius_thresh=radius, duration_thresh=min_dur,
                                   gap_thresh=gap, dropout_thresh=dropout)
        if progress and (i % 50 == 0 or i == len(gaze)):
            progress(i, len(gaze))
        per_clip.append(len(fixes))
        durs.extend(f["duration"] for f in fixes)
        if not fixes:
            clips_with_none += 1

        for t in frame_times(samples):
            total_frames += 1
            if any(f["start_time"] <= t <= f["end_time"] for f in fixes):
                covered += 1

    n_fix = len(durs)
    return {
        "gap": gap,
        "clips": len(per_clip),
        "fix_per_clip": statistics.mean(per_clip) if per_clip else 0.0,
        "fix_per_clip_med": statistics.median(per_clip) if per_clip else 0.0,
        "dur_med": statistics.median(durs) if durs else 0.0,
        "dur_p90": float(np.percentile(durs, 90)) if durs else 0.0,
        "coverage": covered / total_frames if total_frames else 0.0,
        "long_share": sum(1 for d in durs if d > long_sec) / n_fix if n_fix else 0.0,
        "empty_clips": clips_with_none / len(per_clip) if per_clip else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze", default=GAZE_PATH)
    ap.add_argument("--gap", default="0.1,0.2,0.3",
                    help="gap_thresh values to sweep, comma separated (seconds)")
    ap.add_argument("--radius", type=float, default=0.05)
    ap.add_argument("--min-dur", type=float, default=0.3)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (a quick look)")
    args = ap.parse_args()

    print(f"loading {args.gaze} ...", flush=True)
    gaze = load_gaze(args.gaze)
    if not gaze:
        sys.exit(f"no clips in {args.gaze} -- run prepare_gaze.py first")
    if args.limit:
        gaze = dict(list(gaze.items())[:args.limit])

    gaps = [float(g) for g in args.gap.split(",")]
    print(f"{len(gaze)} clips, {sum(len(e.get('samples') or []) for e in gaze.values())} samples  "
          f"| radius={args.radius} min_dur={args.min_dur}s dropout={args.dropout}s  "
          f"| sweeping gap={gaps}", flush=True)

    rows = []
    for gap in gaps:
        t0 = time.time()

        def tick(i, n, _gap=gap):
            print(f"\r  gap={_gap}: {i}/{n} clips", end="", flush=True)

        rows.append(summarise(gaze, gap, args.radius, args.min_dur, args.dropout,
                              progress=tick))
        print(f"\r  gap={gap}: {len(gaze)}/{len(gaze)} clips  ({time.time() - t0:.1f}s)",
              flush=True)
    print()
    hdr = f"{'gap':>5} {'fix/clip':>9} {'median':>7} {'dur_med':>8} {'dur_p90':>8} " \
          f"{'coverage':>9} {'>3s':>7} {'no-fix clips':>13}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['gap']:>5.1f} {r['fix_per_clip']:>9.2f} {r['fix_per_clip_med']:>7.0f} "
              f"{r['dur_med']:>7.2f}s {r['dur_p90']:>7.2f}s {100*r['coverage']:>8.1f}% "
              f"{100*r['long_share']:>6.1f}% {100*r['empty_clips']:>12.1f}%")
    print()
    print("fix/clip 가 크게 줄고 dur_med 가 오르면 작은 값이 fixation 을 쪼개고 있었다는 뜻.")
    print("dur_med 와 함께 '>3s' 가 같이 오르기 시작하면 과병합 신호다.")


if __name__ == "__main__":
    main()
