#!/usr/bin/env python3
"""
Lay out the gaze->image transform candidates so a person can choose one: a score table as evidence,
and rendered frames with the crop box drawn under every candidate. It does NOT choose.

The problem it helps with: the gaze CSVs hold yaw/pitch in the Central Pupil Frame, not pixels, and
EgoLife's mp4s are re-exported video, so the sign/rotation mapping between the two cannot be settled
from a spec sheet. Get it wrong and the box lands on the ceiling while the wearer was looking at a
pan -- the experiment then reports "gaze does not help" about a bug.

The score is evidence, not a verdict. Each candidate is scored by how well a clip's gaze crop
matches THAT CLIP'S OWN 30-sec caption:

    score(transform) = mean over clips of  cos( emb(gaze crop), emb(clip's own caption text) )

The caption describes what the wearer was doing, so a box on the right thing should sit closer to
its own caption than a box the wrong way up. It uses captions, never the QA questions, never the
target labels, and never the retrieval ranking -- so it cannot hand the experiment its own answer.

Every candidate has the same crop size and differs only in where the box sits, which is what makes
the scores comparable. `full` is printed as a reference line, not as a candidate.

Two things make the test actually discriminate. It crops **frame by frame** at each frame's own gaze
point, not once per clip at the median: on this data the per-clip median sits within a few percent of
the frame centre (30 s of saccades average out), so a median-based test moves the box hardly at all
and every candidate scores the same. And it **skips clips whose gaze barely leaves the centre**
(--min-offset), since those cannot separate a flip from its opposite. All candidate transforms are
isometries about the frame centre, so the offset filter is identical for every candidate and cannot
favour one. That distance is radial by default, which is not always enough -- see --axis below.

Then look at variants/*_variants.jpg: the same frame under every candidate, labelled, at the moment
the gaze sat furthest from the frame centre (where the candidates differ most). Pick the tile whose
box is on what the wearer was plainly looking at, and pass it on:

    python experiments/gaze_crop/compare_transforms.py
    # look at the images, then
    GAZE_TRANSFORM=rot90cw bash experiments/gaze_crop/run_stage1.sh

The table's top row is printed as a suggestion only. When the top two are within noise the score
cannot tell them apart at all, and the pictures are the only thing that can.

**--axis, for when two candidates tie because the test never exercised them.** A left/right mirror
only moves the box when the gaze is off-centre HORIZONTALLY. On this data the biggest excursions are
almost all pitch -- the wearer looking down at their own hands -- so the default radial --min-offset
and the default choice of frame to draw both keep clips whose gaze sat at x ~ 0.5. There an
x-mirrored pair crops the SAME pixels: identical embeddings, identical scores, and a rendered frame
whose two tiles are indistinguishable. The tie is the test's, not the data's.

--axis x measures the offset horizontally instead, both for the --min-offset filter and for
choosing the frame to draw, so only clips that actually swung left or right survive:

    python experiments/gaze_crop/compare_transforms.py \
        --axis x --min-offset 0.08 --transforms flipy flipx+flipy

A rotation swaps the two axes, so an --axis filter is fair within one mirrored pair but not across
all six candidates at once; the script warns if a rotation is in the list. Settle one mirror per
run, with the pair named in --transforms.

Cost: one decode per sampled clip plus one forward per (clip, candidate). 150 clips x 6 candidates
is ~10 minutes.
"""

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    ANALYSIS_DIR, CAPTION_30SEC, GAZE_ROOT, MAX_PIXELS, NFRAMES, POOL_PATH, VIDEO_ROOT, clip_key,
    crop_box, gaze_at, load_json, resolve_video_path, save_json, unit,
)
from embed_arms import decode  # noqa: E402
from prepare_gaze import (  # noqa: E402
    _draw, apply_transform, make_projector, median_xy, parse_csv, rebase,
)

DEFAULT_CANDIDATES = ("none", "flipx", "flipy", "flipx+flipy", "rot90cw", "rot90ccw")
ROTATIONS = ("rot90cw", "rot90ccw")


def point_offset(p: Tuple[float, float], axis: str) -> float:
    """How far one gaze point sits from the frame centre, along `axis`."""
    x, y = p
    if axis == "x":
        return abs(x - 0.5)
    if axis == "y":
        return abs(y - 0.5)
    return math.hypot(x - 0.5, y - 0.5)


def clip_offset(pts: List[Tuple[float, float]], axis: str) -> float:
    return float(np.mean([point_offset(p, axis) for p in pts]))


def find_csv(gaze_root: str, key: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(gaze_root):
        for ext in (".csv", ".json"):
            if key + ext in filenames:
                return os.path.join(dirpath, key + ext)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL_PATH)
    ap.add_argument("--gaze-root", default=GAZE_ROOT, help=f"default: {GAZE_ROOT}")
    ap.add_argument("--captions", default=CAPTION_30SEC)
    ap.add_argument("--transforms", nargs="+", default=list(DEFAULT_CANDIDATES))
    ap.add_argument("--n-clips", type=int, default=150, help="clips to try; targets come first")
    ap.add_argument("--min-offset", type=float, default=0.05,
                    help="skip clips whose gaze stays within this normalised distance of the frame "
                         "centre, averaged over the sampled frames. They cannot tell a flip from its "
                         "opposite. 0 keeps everything.")
    ap.add_argument("--axis", choices=("radial", "x", "y"), default="radial",
                    help="which component of the gaze's distance from centre drives --min-offset "
                         "and picks the frame to draw. radial (default) is the overall distance. "
                         "Use x to settle flipx and y to settle flipy: a clip whose gaze only ever "
                         "moved vertically puts every x-mirrored pair on the SAME crop, so it "
                         "cannot separate them however far off-centre it is. Measured on the "
                         "untransformed points, so the threshold is the same for every candidate; "
                         "meaningful only within a mirrored pair, so pair it with --transforms.")
    ap.add_argument("--pick-mode", choices=("per-frame", "median"), default="per-frame",
                    help="per-frame: crop each frame at its own gaze point (default, discriminates). "
                         "median: one box per clip, which on this data is nearly the centre crop.")
    ap.add_argument("--ratio", type=float, default=0.5, help="crop ratio used for the test")
    ap.add_argument("--video-root", default=VIDEO_ROOT)
    ap.add_argument("--frame-width", type=int, default=None)
    ap.add_argument("--frame-height", type=int, default=None)
    ap.add_argument("--projection", choices=("equidistant", "pinhole"), default="equidistant")
    ap.add_argument("--focal-px", type=float, default=None)
    ap.add_argument("--nframes", type=int, default=NFRAMES)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--dump-variants", type=int, default=4, metavar="N",
                    help="render N clips' most off-centre frame under every candidate, as a labelled "
                         "grid. 0 disables it, and then there is nothing to look at.")
    ap.add_argument("--out", default=os.path.join(ANALYSIS_DIR, "transform_choice.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.gaze_root):
        ap.error(f"gaze root not found: {args.gaze_root} (pass --gaze-root or set WORLDMM_GAZE_ROOT)")
    rot = [n for n in args.transforms if n in ROTATIONS]
    if args.axis != "radial" and rot:
        print(f"WARNING: --axis {args.axis} with {rot} in the candidates. A rotation swaps the two "
              f"axes, so an --axis filter that keeps the clips one candidate needs is not the same "
              f"filter for a rotated one, and the table stops being a fair comparison across all "
              f"six. Use --axis to settle ONE mirror, e.g.\n"
              f"    --axis x --transforms flipy flipx+flipy")
    print(f"gaze root: {args.gaze_root}")
    pool = load_json(args.pool)
    caption_text = {clip_key(c["video_path"]): c.get("text", "") for c in load_json(args.captions)}

    rng = np.random.default_rng(args.seed)
    targets = [r for r in pool["clips"] if r["is_target"]]
    others = [r for r in pool["clips"] if not r["is_target"]]
    rng.shuffle(others)
    picked = (targets + others)[:args.n_clips]

    from worldmm.embedding import EmbeddingModel
    model = EmbeddingModel()

    rows: List[Dict[str, Any]] = []        # per clip: {"key", "caption", "emb": {transform: vec}}
    offsets: List[float] = []
    skipped_flat = 0
    w = h = None
    project = None
    for i, r in enumerate(picked, 1):
        csv_path = find_csv(args.gaze_root, r["key"])
        cap = caption_text.get(r["key"], "").strip()
        vpath = resolve_video_path(r["video_path"], args.video_root)
        if not csv_path or not cap or not os.path.exists(vpath):
            continue
        try:
            arr, idx, fps = decode(vpath, args.nframes)
        except Exception as e:
            print(f"  [{r['key']}] decode failed: {e}")
            continue
        if w is None:
            h, w = int(arr.shape[1]), int(arr.shape[2])
            w = args.frame_width or w
            h = args.frame_height or h
            project, _ = make_projector(w, h, args.projection, args.focal_px, None)
            print(f"frame {w}x{h}, {len(args.transforms)} candidates, crop {args.ratio}, "
                  f"{args.pick_mode}, min {args.axis} offset {args.min_offset}")

        # untransformed samples once; the candidates only rotate/flip the normalised points
        base, _, _, _, _ = parse_csv(csv_path, "auto", project, "none", w, h)
        if not base:
            continue
        base, _ = rebase(base)
        entry_pts = {"samples": base, "median": median_xy(base)}

        # the point used for each frame, before any transform
        if args.pick_mode == "median":
            pts = [tuple(entry_pts["median"])] * len(arr)
        else:
            pts = []
            for fi in idx:
                p = gaze_at(entry_pts, fi / fps, tol=1.0)
                pts.append(tuple(p) if p else tuple(entry_pts["median"]))

        # how far the gaze sits from the centre, along --axis. Measured on the untransformed
        # points, so the same clips survive whichever candidate is being scored.
        off = clip_offset(pts, args.axis)
        offsets.append(off)
        if off < args.min_offset:
            skipped_flat += 1
            continue

        from PIL import Image
        # the most off-centre sampled frame along --axis: where the candidates disagree most, so
        # that is the frame worth drawing
        j = max(range(len(pts)), key=lambda t: point_offset(pts[t], args.axis))
        entry: Dict[str, Any] = {"key": r["key"], "is_target": r["is_target"], "offset": off,
                                 "emb": {}, "draw_frame": arr[j].copy(), "draw_point": pts[j]}
        for name in args.transforms:
            frames = []
            for a, (px, py) in zip(arr, pts):
                tx, ty = apply_transform(px, py, name)
                frames.append(Image.fromarray(a).crop(crop_box(w, h, tx, ty, args.ratio)))
            vec = model.encode_video([{"video": frames, "nframes": len(frames),
                                       "max_pixels": args.max_pixels}])
            entry["emb"][name] = np.asarray(vec, dtype=np.float32)[0]
        full = [Image.fromarray(a) for a in arr]
        entry["emb"]["__full__"] = np.asarray(
            model.encode_video([{"video": full, "nframes": len(full),
                                 "max_pixels": args.max_pixels}]), dtype=np.float32)[0]
        entry["caption"] = cap
        rows.append(entry)
        if i % 25 == 0:
            print(f"  ... {i}/{len(picked)} clips tried, {len(rows)} scored, "
                  f"{skipped_flat} skipped as too central")

    if not rows:
        raise SystemExit(
            f"no clip was scorable ({skipped_flat} skipped as too central on {args.axis} out of "
            f"{len(offsets)} with gaze). Lower --min-offset, raise --n-clips, or widen --axis.")
    if offsets:
        o = sorted(offsets)
        print(f"\ngaze {args.axis} offset from centre ({args.pick_mode}): median {o[len(o)//2]:.3f}, "
              f"p90 {o[int(0.9 * (len(o) - 1))]:.3f} (normalised; 0.5 = frame edge). "
              f"{skipped_flat}/{len(offsets)} clips skipped as too central.")

    caps = unit(np.asarray(model.encode_vis_query([r["caption"] for r in rows]), dtype=np.float32))
    names = list(args.transforms) + ["__full__"]
    scores: Dict[str, np.ndarray] = {}
    for name in names:
        emb = unit(np.stack([r["emb"][name] for r in rows]))
        scores[name] = np.einsum("ij,ij->i", emb, caps)      # per-clip cosine with its own caption

    ranked = sorted(args.transforms, key=lambda n: -float(scores[n].mean()))
    top, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)

    # per-clip votes: which candidate scored highest on each clip. A mean can be carried by a few
    # clips; the vote count says whether the ordering is consistent.
    stacked = np.stack([scores[n] for n in args.transforms])          # [cand, clip]
    votes = {n: int((stacked.argmax(axis=0) == i).sum()) for i, n in enumerate(args.transforms)}

    ci = None
    if second:
        d = scores[top] - scores[second]
        idx = np.random.default_rng(0).integers(0, len(d), size=(5000, len(d)))
        boot = d[idx].mean(axis=1)
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))

    print(f"\ncosine with each clip's own 30-sec caption, {len(rows)} clips, crop {args.ratio}, "
          f"{args.pick_mode}")
    print(f"{'transform':>14} | {'mean cos':>9} | {'vs full':>8} | {'클립 1위':>9}")
    print("-" * 52)
    full_mean = float(scores["__full__"].mean())
    for n in ranked:
        m = float(scores[n].mean())
        print(f"{n:>14} | {m:9.4f} | {m - full_mean:+8.4f} | {votes[n]:>9}")
    print(f"{'full (ref)':>14} | {full_mean:9.4f} |")

    separated = bool(ci and ci[0] > 0)
    print(f"\nThe score is evidence, not a decision. Top row '{top}'"
          + (f" is separated from '{second}' (95% CI of the gap "
             f"{ci[0]:+.4f}..{ci[1]:+.4f})." if separated else
             f" is NOT separated from '{second}' (95% CI {ci[0]:+.4f}..{ci[1]:+.4f}) -- the score "
             f"cannot tell them apart."))

    save_json({
        "n_clips": len(rows), "ratio": args.ratio, "frame_size": [w, h],
        "pick_mode": args.pick_mode, "min_offset": args.min_offset, "axis": args.axis,
        "transforms": list(args.transforms),
        "n_skipped_too_central": skipped_flat,
        "offset_median": (sorted(offsets)[len(offsets) // 2] if offsets else None),
        "criterion": "mean cosine(gaze-crop embedding, own 30-sec caption embedding)",
        "scores": {n: float(scores[n].mean()) for n in names},
        "ranked": ranked, "suggested": top, "runner_up": second, "votes": votes,
        "gap_ci95": list(ci) if ci else None,
        "separated": separated, "chosen_by": "human",
    }, args.out)
    print(f"wrote {args.out}")

    if args.dump_variants:
        grids = variant_grids(rows, args, w, h, scores)
    else:
        grids = []
        print("\n--dump-variants 0: no images rendered, so there is nothing to look at.")

    print("\n" + "-" * 74)
    print("Now look at the images and decide:")
    for g in grids[:4]:
        print(f"  {g}")
    axis_note = {"radial": "furthest from the centre",
                 "x": "furthest from the centre HORIZONTALLY (so a left/right mirror moves the box)",
                 "y": "furthest from the centre VERTICALLY (so an up/down mirror moves the box)"}
    print("\nEach tile is the same frame with the gaze point and crop box drawn under one candidate,")
    print(f"labelled top-left, at the moment the gaze sat {axis_note[args.axis]}.")
    print("Pick the tile whose box is on what the wearer was plainly looking at -- then run stage 1:")
    print(f"\n    GAZE_TRANSFORM=<그 라벨> bash experiments/gaze_crop/run_stage1.sh")
    print(f"\n(the table above suggests '{top}', "
          f"{'and the gap over the runner-up is real' if separated else 'but it is a tie -- the pictures decide'})")
    print("-" * 74)


def variant_grids(rows: List[Dict[str, Any]], args, w: int, h: int,
                  scores: Dict[str, np.ndarray]) -> List[str]:
    """One image per clip: the same frame under every candidate, labelled, biggest gaze offset first."""
    from PIL import Image
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "..", "variants")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    order = sorted(rows, key=lambda r: -r["offset"])[:args.dump_variants]
    means = {n: float(scores[n].mean()) for n in args.transforms}
    paths = []
    for r in order:
        img = Image.fromarray(r["draw_frame"])
        px, py = r["draw_point"]
        tiles = []
        for name in args.transforms:
            tx, ty = apply_transform(px, py, name)
            tiles.append(_draw(img, tx, ty, args.ratio, f"{name}   cos {means[name]:.4f}"))
        cols = 3
        rowsn = (len(tiles) + cols - 1) // cols
        tw, th = tiles[0].width // 2, tiles[0].height // 2
        grid = Image.new("RGB", (tw * cols, th * rowsn), (20, 20, 20))
        for i, t in enumerate(tiles):
            grid.paste(t.resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
        path = os.path.join(out_dir, f"{r['key']}_variants.jpg")
        grid.save(path, quality=85)
        paths.append(path)
    return paths


if __name__ == "__main__":
    main()
