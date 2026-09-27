#!/usr/bin/env python3
"""
Render the gaze->image transform candidates so a person can choose one. It does NOT choose, and it
does not score: it draws, and you look.

The problem: the gaze CSVs hold yaw/pitch in the Central Pupil Frame, not pixels, and EgoLife's
mp4s are re-exported video, so the sign/rotation mapping between the two cannot be settled from a
spec sheet. Get it wrong and the box lands on the ceiling while the wearer was looking at a pan --
the experiment then reports "gaze does not help" about a bug.

For each of a handful of clips it writes variants/<clip>_variants.jpg: the same frame under every
candidate, labelled, with the gaze point and crop box drawn. Pick the tile whose box is on what the
wearer was plainly looking at, and pass it on:

    python experiments/gaze_crop/compare_transforms.py
    # look at the images, then
    GAZE_TRANSFORM=rot90cw bash experiments/gaze_crop/run_stage1.sh

Two things decide whether the pictures can discriminate at all.

**Which frame gets drawn.** Per-frame gaze, not the clip median: 30 s of saccades average out, so a
median-based box sits within a few percent of the frame centre and every candidate looks the same.
Within a clip it draws the most off-centre sampled frame, where the candidates disagree most.

**--axis.** A left/right mirror only moves the box when the gaze is off-centre HORIZONTALLY. On
this data the biggest excursions are almost all pitch -- the wearer looking down at their own
hands -- so the default radial --min-offset keeps clips whose gaze sat at x ~ 0.5, where an
x-mirrored pair crops the SAME pixels and its two tiles are indistinguishable. --axis x measures
the offset horizontally instead, for both the filter and the frame choice:

    python experiments/gaze_crop/compare_transforms.py \
        --axis x --min-offset 0.08 --dump-variants 12 --transforms flipy flipx+flipy

Settle one mirror per run, with the pair named in --transforms: a rotation swaps the two axes, so
an --axis filter is not the same filter for it. Look at 12 images, not 4 -- on this data 4 gave the
wrong answer, and two clips 2.5 minutes apart in the same corridor disagreed with each other.

This script used to also print a table scoring each candidate by the cosine between its crop and
the clip's own 30-sec caption. It was never used to decide anything -- it could not separate the
candidates on this data, and the pictures settled it -- so it is gone, along with the embedding
pass it needed. Decoding and drawing only: no GPU, a couple of minutes.
"""

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    ANALYSIS_DIR, GAZE_ROOT, NFRAMES, POOL_PATH, VIDEO_ROOT, crop_box, gaze_at, load_json,
    resolve_video_path, save_json,
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
    rng = np.random.default_rng(args.seed)
    targets = [r for r in pool["clips"] if r["is_target"]]
    others = [r for r in pool["clips"] if not r["is_target"]]
    rng.shuffle(others)
    picked = (targets + others)[:args.n_clips]

    rows: List[Dict[str, Any]] = []        # 그릴 클립: {"key", "offset", "draw_frame", "draw_point"}
    offsets: List[float] = []
    skipped_flat = 0
    w = h = None
    project = None
    for i, r in enumerate(picked, 1):
        csv_path = find_csv(args.gaze_root, r["key"])
        vpath = resolve_video_path(r["video_path"], args.video_root)
        if not csv_path or not os.path.exists(vpath):
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

        # how far the gaze sits from the centre, along --axis, on the untransformed points
        off = clip_offset(pts, args.axis)
        offsets.append(off)
        if off < args.min_offset:
            skipped_flat += 1          # this clip cannot separate a mirror from its opposite
            continue

        # the most off-centre sampled frame along --axis: where the candidates disagree most, so
        # that is the frame worth drawing
        j = max(range(len(pts)), key=lambda t: point_offset(pts[t], args.axis))
        rows.append({"key": r["key"], "is_target": r["is_target"], "offset": off,
                     "draw_frame": arr[j].copy(), "draw_point": pts[j]})
        if i % 25 == 0:
            print(f"  ... {i}/{len(picked)} clips tried, {len(rows)} usable, "
                  f"{skipped_flat} skipped as too central")

    if not rows:
        raise SystemExit(
            f"no clip was usable ({skipped_flat} skipped as too central on {args.axis} out of "
            f"{len(offsets)} with gaze). Lower --min-offset, raise --n-clips, or widen --axis.")
    if offsets:
        o = sorted(offsets)
        print(f"\ngaze {args.axis} offset from centre ({args.pick_mode}): median {o[len(o)//2]:.3f}, "
              f"p90 {o[int(0.9 * (len(o) - 1))]:.3f} (normalised; 0.5 = frame edge). "
              f"{skipped_flat}/{len(offsets)} clips skipped as too central.")

    save_json({
        "n_clips": len(rows), "ratio": args.ratio, "frame_size": [w, h],
        "pick_mode": args.pick_mode, "min_offset": args.min_offset, "axis": args.axis,
        "transforms": list(args.transforms),
        "n_skipped_too_central": skipped_flat,
        "offset_median": (sorted(offsets)[len(offsets) // 2] if offsets else None),
        "chosen_by": "human",      # 이 스크립트는 고르지 않는다. 그림만 만든다.
    }, args.out)
    print(f"wrote {args.out}")

    if args.dump_variants:
        grids = variant_grids(rows, args, w, h)
    else:
        grids = []
        print("\n--dump-variants 0: no images rendered, so there is nothing to look at.")

    print("\n" + "-" * 74)
    print("Now look at the images and decide:")
    for g in grids:
        print(f"  {g}")
    axis_note = {"radial": "furthest from the centre",
                 "x": "furthest from the centre HORIZONTALLY (so a left/right mirror moves the box)",
                 "y": "furthest from the centre VERTICALLY (so an up/down mirror moves the box)"}
    print("\nEach tile is the same frame with the gaze point and crop box drawn under one candidate,")
    print(f"labelled top-left, at the moment the gaze sat {axis_note[args.axis]}.")
    print("Pick the tile whose box is on what the wearer was plainly looking at -- then run stage 1:")
    print(f"\n    GAZE_TRANSFORM=<그 라벨> bash experiments/gaze_crop/run_stage1.sh")
    print(f"\nNothing here ranks the candidates. Twelve images beat four: on this data four gave "
          f"the wrong answer.")
    print("-" * 74)


def variant_grids(rows: List[Dict[str, Any]], args, w: int, h: int) -> List[str]:
    """클립 하나당 이미지 한 장: 같은 프레임을 후보별로 그려 격자로 붙인다.

    입력:  rows  채점 루프가 모은 {"key", "offset", "draw_frame", "draw_point"} 목록
                 draw_point 는 변환 전 정규화 좌표 (x, y)
    출력:  쓴 파일 경로 목록. variants/<clip>_variants.jpg

    --axis 기준 이탈이 큰 클립부터 --dump-variants 장을 고른다. 후보 간 박스 차이가
    가장 큰 클립이 판단하기 쉽기 때문.
    """
    from PIL import Image
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "..", "variants")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    order = sorted(rows, key=lambda r: -r["offset"])[:args.dump_variants]
    paths = []
    for r in order:
        img = Image.fromarray(r["draw_frame"])
        px, py = r["draw_point"]
        # 같은 프레임 · 같은 crop 크기. 후보마다 박스 위치만 다르다.
        tiles = [_draw(img, *apply_transform(px, py, name), args.ratio, name)
                 for name in args.transforms]
        cols = min(3, len(tiles))
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
