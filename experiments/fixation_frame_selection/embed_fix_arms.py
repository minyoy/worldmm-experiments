#!/usr/bin/env python3
"""
Embed each pool clip using only the frames that fall inside a fixation.

The gaze-crop experiment changes *where* the 16 frames are cropped. This one changes *which
frames are embedded at all*: a frame whose instant lands mid-saccade shows motion blur and a
scene the subject was not looking at, so it is dropped rather than cropped. A clip with no
fixation-covered frame is not embedded at all -- it is absent from the arm, which recall_eval.py
already reports as unscorable instead of silently substituting something else.

Arms written here (files land in ../gaze_crop/emb/ so recall_eval.py compares them against
`full` and `gazef@0.5` in one paired table):

    fixsel              fixation frames only, whole frame.  frame SELECTION alone.
    fixsel_gazef@R      fixation frames only, each cropped on its fixation's CENTROID.
                        selection + gaze crop -- the arm the experiment is for.
    fixsel_ctl          the count-matched control. Same number of frames per clip as `fixsel`,
                        and the same clips, but spaced uniformly over the whole clip: it ignores
                        fixations entirely.

fixsel_ctl is not optional. Dropping ~26% of frames is itself a change to what the encoder sees
(fewer frames, denser in time), and VLM2Vec pools over frames. Without the control, "fixsel beat
full" cannot be separated from "16 frames were worse than 12". It plays the role center@R plays
for gazef@R.

The centroid, not the per-frame gaze sample, is what fixsel_gazef crops on. Inside a fixation the
samples scatter by up to `radius`; their mean is the better estimate of the one point being looked
at, and it holds the box still for the whole fixation. Frames outside any fixation do not arise
here -- they were dropped.

    python embed_fix_arms.py --dry-run                        # no GPU: frame bookkeeping only
    python embed_fix_arms.py --pool ../gaze_crop/pool_all.json
    python embed_fix_arms.py --arms fixsel fixsel_gazef@0.5    # skip the control (don't)
    python embed_fix_arms.py --select refill                   # keep 16 frames, all inside fixations

Writes ../gaze_crop/emb/<arm>.npz plus <arm>.meta.json, incrementally (resumes unless --overwrite).
"""

import argparse
import contextlib
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GAZE_CROP = os.path.abspath(os.path.join(HERE, "..", "gaze_crop"))
sys.path.insert(0, HERE)
sys.path.insert(0, GAZE_CROP)
sys.path.insert(0, os.path.join(HERE, "streamgaze"))

from gaze_common import (  # noqa: E402
    EMB_DIR, GAZE_PATH, MAX_PIXELS, NFRAMES, POOL_PATH, VIDEO_ROOT, crop_box, decord_threads,
    load_gaze, load_json, resolve_video_path,
)
from embed_arms import Store, frame_indices_uniform  # noqa: E402
from fixation_sweep import clip_fixations  # noqa: E402

# gap=0.2 is settled (README: coverage moves 1.3 pp over the whole sweep, so it is not the
# parameter this experiment turns on). radius is, and it stays a flag.
FIX_DEFAULTS = {"radius": 0.05, "min_dur": 0.3, "gap": 0.2, "dropout": 0.4}

# fixsel | fixsel_ctl | fixsel_gazef@<ratio>
ARM_RE = re.compile(r"^(fixsel|fixsel_ctl|fixsel_gazef)(?:@([0-9.]+))?$")


def parse_arm(name: str) -> Tuple[str, Optional[float]]:
    m = ARM_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad arm '{name}'. Use fixsel | fixsel_gazef@0.5 | fixsel_ctl")
    mode, ratio = m.group(1), m.group(2)
    if mode == "fixsel_gazef":
        if ratio is None:
            raise ValueError(f"arm '{name}' needs a crop ratio, e.g. fixsel_gazef@0.5")
        r = float(ratio)
        if not 0 < r <= 1:
            raise ValueError(f"crop ratio must be in (0, 1], got {r}")
        return mode, r
    if ratio is not None:
        raise ValueError(f"arm '{name}' takes no ratio")
    return mode, None


def arm_filename(name: str) -> str:
    """Same spelling rule gaze_common.arm_filename uses, so recall_eval.py --arms matches."""
    return name.replace("@", "_").replace(".", "") + ".npz"


# ---------------------------------------------------------------------------
# which frames
# ---------------------------------------------------------------------------

def fixation_of(fixes: Sequence[Dict[str, Any]], t: float) -> Optional[Dict[str, Any]]:
    """The fixation containing instant `t`, or None. Intervals are closed at both ends."""
    for f in fixes:
        if f["start_time"] <= t <= f["end_time"]:
            return f
    return None


def _time_to_index(t: float, fps: float, total: int) -> int:
    return int(min(max(int(round(t * fps)), 0), total - 1))


def select_frames(fixes: Sequence[Dict[str, Any]], total: int, fps: float, nframes: int,
                  mode: str) -> List[Tuple[int, Dict[str, Any]]]:
    """(frame index, its fixation) for the frames this clip contributes.

    mode "subset" (default) is the experiment as asked: take the SAME 16 uniform instants
    embed_arms.py uses and throw away the ones outside every fixation. What survives is a
    subset of the baseline's own frames, so `full` and `fixsel` differ by a deletion and
    nothing else. Count varies per clip (~12 of 16 at radius=0.05).

    mode "refill" instead spends all `nframes` inside fixation time: it walks the union of the
    fixation intervals and samples nframes instants equally spaced in COVERED time. Frame count
    is then constant, which removes the count confound at the price of no longer being a subset
    of the baseline's frames (it resamples where the baseline never looked). Reported separately;
    do not mix the two in one table.
    """
    if not fixes:
        return []
    if mode == "subset":
        out = []
        for i in frame_indices_uniform(total, nframes):
            f = fixation_of(fixes, i / fps)
            if f is not None:
                out.append((i, f))
        return out
    if mode != "refill":
        raise ValueError(f"unknown --select '{mode}'")

    # cumulative covered time -> instant. Zero-length fixations cannot occur (duration >= min_dur).
    spans = sorted(fixes, key=lambda f: f["start_time"])
    lengths = [f["end_time"] - f["start_time"] for f in spans]
    covered = sum(lengths)
    if covered <= 0:
        return []
    out: List[Tuple[int, Dict[str, Any]]] = []
    seen = set()
    for j in range(nframes):
        # midpoints of nframes equal slices, so the first and last frame sit inside a fixation
        # rather than exactly on its boundary
        u = covered * (j + 0.5) / nframes
        acc = 0.0
        for f, ln in zip(spans, lengths):
            # `f is spans[-1]` also catches a u that floating point left just past the final
            # boundary; without it the loop would fall through and silently drop that frame
            if u <= acc + ln or f is spans[-1]:
                idx = _time_to_index(f["start_time"] + min(u - acc, ln), fps, total)
                if idx not in seen:
                    seen.add(idx)
                    out.append((idx, f))
                break
            acc += ln
    return sorted(out, key=lambda p: p[0])


def control_frames(total: int, n: int, nframes: int) -> List[int]:
    """n frames spread uniformly over the WHOLE clip: the count-matched control.

    Uniform, not random: `full` is uniform too, so the only difference from `full` is how many
    frames there are. A random draw would add a second difference (irregular spacing) and this
    arm exists to isolate exactly one.
    """
    if n <= 0:
        return []
    if n >= nframes:
        return frame_indices_uniform(total, nframes)
    return frame_indices_uniform(total, n)


# ---------------------------------------------------------------------------
# decode + crop
# ---------------------------------------------------------------------------

def open_reader(path: str):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), num_threads=decord_threads())
    fps = float(vr.get_avg_fps()) or 30.0
    return vr, len(vr), fps


def build_frames(vr, wanted: Sequence[int]):
    """{frame index: RGB array} for the union of every arm's indices, decoded in one batch."""
    idx = sorted(set(int(i) for i in wanted))
    if not idx:
        return {}
    arr = vr.get_batch(idx).asnumpy()
    return {i: a for i, a in zip(idx, arr)}


def to_pil(arr, box: Optional[Tuple[int, int, int, int]]):
    from PIL import Image
    im = Image.fromarray(arr)
    return im.crop(box) if box else im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL_PATH)
    ap.add_argument("--gaze", default=GAZE_PATH)
    ap.add_argument("--arms", nargs="+", default=["fixsel", "fixsel_gazef@0.5", "fixsel_ctl"])
    ap.add_argument("--arm-suffix", default="")
    ap.add_argument("--out-dir", default=EMB_DIR, help="default: ../gaze_crop/emb, next to full/gazef")
    ap.add_argument("--video-root", default=VIDEO_ROOT)
    ap.add_argument("--nframes", type=int, default=NFRAMES)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--select", choices=("subset", "refill"), default="subset",
                    help="subset: keep the fixation-covered ones of the baseline's 16 uniform "
                         "frames (default). refill: resample 16 frames inside fixation time, so "
                         "frame count stays 16. See select_frames().")
    ap.add_argument("--radius", type=float, default=FIX_DEFAULTS["radius"])
    ap.add_argument("--min-dur", type=float, default=FIX_DEFAULTS["min_dur"])
    ap.add_argument("--gap", type=float, default=FIX_DEFAULTS["gap"])
    ap.add_argument("--dropout", type=float, default=FIX_DEFAULTS["dropout"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="decode and select, but load no model: gives the frame-count and "
                         "dropped-clip table without a GPU")
    args = ap.parse_args()

    pool = load_json(args.pool)
    gaze = load_gaze(args.gaze)
    clips = pool["clips"][:args.limit] if args.limit else pool["clips"]

    specs = [(n, *parse_arm(n)) for n in args.arms]
    stores = {n: Store(os.path.join(args.out_dir, arm_filename(n + args.arm_suffix)), args.overwrite)
              for n, _, _ in specs}

    gaze_transform = load_json(args.gaze).get("transform") if os.path.exists(args.gaze) else None
    fix_params = {"radius": args.radius, "min_dur": args.min_dur, "gap": args.gap,
                  "dropout": args.dropout, "select": args.select}
    base_meta = {"pool": args.pool, "nframes": args.nframes, "max_pixels": args.max_pixels,
                 "video_root": args.video_root, "gaze_file": args.gaze,
                 "gaze_transform": gaze_transform, "fixation": fix_params,
                 "n_clips_requested": len(clips)}

    # Rows built under a different gaze transform or different detector thresholds are not
    # comparable, and mixing them inside one npz is invisible afterwards. Same refusal
    # embed_arms.py makes for the transform, extended to the fixation parameters -- a rerun with
    # radius=0.03 must not resume on top of radius=0.05 rows.
    for n, _, _ in specs:
        prev = stores[n].prev_meta
        if not prev:
            continue
        was_t, was_f = prev.get("gaze_transform"), prev.get("fixation")
        if (was_t not in (None, gaze_transform)) or (was_f is not None and was_f != fix_params):
            print(f"WARNING: {n} holds {len(stores[n].rows)} embeddings built with "
                  f"transform '{was_t}' / {was_f}, but this run says '{gaze_transform}' / "
                  f"{fix_params}. Discarding them and rebuilding this arm from scratch.")
            stores[n].reset()

    have = sum(1 for r in clips if r["key"] in gaze)
    print(f"arms: {[n for n, _, _ in specs]}  select={args.select}")
    print(f"gaze coverage: {have}/{len(clips)} clips from {args.gaze}")
    print(f"fixation detector: radius={args.radius} min_dur={args.min_dur}s gap={args.gap}s "
          f"dropout={args.dropout}s")
    if have == 0:
        raise SystemExit("no gaze for any pool clip; point --gaze at a prepared file")

    todo = [r for r in clips if any(not stores[n].has(r["key"]) for n, _, _ in specs)]
    print(f"clips to do: {len(todo)}/{len(clips)} (the rest are already in {args.out_dir})")

    model = None
    if not args.dry_run:
        from worldmm.embedding import EmbeddingModel  # late import: --dry-run needs no GPU
        model = EmbeddingModel()

    devnull = open(os.devnull, "w")
    missing_video, no_gaze, dropped, kept_counts = [], [], [], []
    t_start = time.time()
    for i, r in enumerate(todo, 1):
        path = resolve_video_path(r["video_path"], args.video_root)
        if not os.path.exists(path):
            missing_video.append(path)
            continue
        entry = gaze.get(r["key"]) or {}
        samples = entry.get("samples") or []
        if len(samples) < 2:
            no_gaze.append(r["key"])
            continue
        try:
            vr, total, fps = open_reader(path)
        except Exception as e:
            print(f"  [{r['key']}] decode failed: {e}")
            missing_video.append(path)
            continue

        # The detector prints one line per clip; this loop runs over the whole pool.
        with contextlib.redirect_stdout(devnull):
            fixes = clip_fixations(samples, radius_thresh=args.radius,
                                  duration_thresh=args.min_dur, gap_thresh=args.gap,
                                  dropout_thresh=args.dropout)
        picked = select_frames(fixes, total, fps, args.nframes, args.select)
        if not picked:
            # The clip is not embedded in ANY arm here, control included: a clip present in the
            # control but absent from fixsel would make the two arms score different question
            # sets, and recall_eval's paired comparison would silently lose the difference.
            dropped.append(r["key"])
            continue
        kept_counts.append(len(picked))

        ctl_idx = control_frames(total, len(picked), args.nframes)
        wanted = {n: (ctl_idx if m == "fixsel_ctl" else [j for j, _ in picked])
                  for n, m, _ in specs if not stores[n].has(r["key"])}
        if not wanted:
            continue
        frames = build_frames(vr, [j for v in wanted.values() for j in v])
        if not frames:
            continue
        h, w = next(iter(frames.values())).shape[:2]

        for name, mode, ratio in specs:
            if name not in wanted:
                continue
            if mode == "fixsel_gazef":
                # crop on the frame's own fixation CENTROID, not its nearest gaze sample
                items = [(j, crop_box(w, h, f["center_x"], f["center_y"], ratio)) for j, f in picked]
            else:
                items = [(j, None) for j in wanted[name]]
            pil = [to_pil(frames[j], box) for j, box in items if j in frames]
            if not pil:
                continue
            if args.dry_run:
                continue
            vec = model.encode_video([{"video": pil, "nframes": len(pil),
                                       "max_pixels": args.max_pixels}])
            stores[name].put(r["key"], np.asarray(vec)[0])

        if args.save_every and i % args.save_every == 0:
            rate = i / max(time.time() - t_start, 1e-9)
            if not args.dry_run:
                for name, _, _ in specs:
                    stores[name].save(dict(base_meta, arm=name))
            print(f"  ... {i}/{len(todo)} clips, {rate:.2f} clip/s, "
                  f"ETA {(len(todo) - i) / rate / 60:.0f} min"
                  f"{'' if args.dry_run else ', checkpointed'}")

    if not args.dry_run:
        for name, _, _ in specs:
            stores[name].save(dict(base_meta, arm=name))
            print(f"[{name}] {len(stores[name].rows)} embeddings -> {stores[name].path}")

    # ---- the bookkeeping this experiment is judged on -------------------------
    n_seen = len(kept_counts) + len(dropped)
    print()
    if kept_counts:
        k = np.array(kept_counts)
        print(f"frames kept per clip: mean {k.mean():.2f}/{args.nframes} "
              f"({100 * k.mean() / args.nframes:.1f}%), median {int(np.median(k))}, "
              f"min {k.min()}, max {k.max()}")
    if n_seen:
        print(f"clips with no fixation frame (NOT embedded): {len(dropped)}/{n_seen} = "
              f"{100 * len(dropped) / n_seen:.1f}%"
              + (f", e.g. {dropped[:3]}" if dropped else ""))
        print("  recall_eval.py drops a question whose target clip is one of these, in EVERY arm "
              "it compares, so the paired table stays paired -- watch its 'unscorable' line.")
    dt = time.time() - t_start
    if todo:
        print(f"{len(todo)} clips x {len(specs)} arm(s) in {dt / 60:.1f} min "
              f"= {dt / len(todo):.2f} s/clip -> {6223 * dt / len(todo) / 3600:.1f} h for 6,223.")
    if missing_video:
        print(f"WARNING: {len(missing_video)} clip videos missing/unreadable, e.g. {missing_video[:2]}")
    if no_gaze:
        print(f"WARNING: {len(no_gaze)} clips had fewer than 2 gaze samples and were skipped, "
              f"e.g. {no_gaze[:3]}")


if __name__ == "__main__":
    main()
