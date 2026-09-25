#!/usr/bin/env python3
"""
Extract and cache oracle frames for every question in subset.json.

  frames/{qid}/{clip_idx:02d}_{frame_idx:03d}.jpg   1 fps over the target clip(s), 64 frames max
                                                     (proportional per-clip sampling, same rule as
                                                     VisualMemory._frames_to_context_dict)
  frames16/{qid}/{clip_idx:02d}_{k:02d}.jpg          masking set only: 16 uniformly sampled frames per clip

Each directory also gets a meta.json listing the source clip and frame index per file, so every
condition (C, D, E', Exp.3) reads identical pixels.

Usage (from repo root):
    python experiments/visual_bottleneck/extract_oracle_frames.py [--max-frames 64] [--n16 16] [--overwrite]
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from decord import VideoReader, cpu
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    FRAMES16_DIR, FRAMES_DIR, SUBSET_PATH, VIDEO_ROOT, load_subset, resolve_video_path, save_json,
)


def frame_indices_1fps(vr: VideoReader, fps: float = 1.0) -> List[int]:
    video_fps = vr.get_avg_fps()
    total = len(vr)
    interval = max(1, int(video_fps / fps)) if fps > 0 else max(1, int(video_fps))
    return list(range(0, max(total - 1, 0) + 1, interval))


def frame_indices_uniform(vr: VideoReader, n: int) -> List[int]:
    total = len(vr)
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n, dtype=int).tolist()


def read_frames(path: str, indices: List[int]) -> List[Image.Image]:
    vr = VideoReader(path, ctx=cpu(0))
    arr = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(a) for a in arr]


def proportional_cap(per_clip: List[List[int]], max_frames: int) -> List[List[int]]:
    """Mirror VisualMemory._frames_to_context_dict: scale each clip's list by max/total, keep >= 1."""
    total = sum(len(x) for x in per_clip)
    if total <= max_frames:
        return per_clip
    ratio = max_frames / total
    out = []
    for idx in per_clip:
        keep = max(1, int(len(idx) * ratio))
        pos = np.linspace(0, len(idx) - 1, keep, dtype=int).tolist()
        out.append([idx[p] for p in pos])
    return out


def write_frames(out_dir: str, clips: List[str], per_clip_indices: List[List[int]], overwrite: bool,
                 video_root: str) -> Tuple[int, bool]:
    if os.path.isdir(out_dir) and os.path.exists(os.path.join(out_dir, "meta.json")) and not overwrite:
        return 0, True
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    n = 0
    for ci, (vp, idxs) in enumerate(zip(clips, per_clip_indices)):
        imgs = read_frames(resolve_video_path(vp, video_root), idxs)
        for fi, (frame_idx, img) in enumerate(zip(idxs, imgs)):
            name = f"{ci:02d}_{fi:03d}.jpg"
            img.save(os.path.join(out_dir, name), quality=90)
            meta.append({"file": name, "clip": vp, "clip_idx": ci, "frame_idx": int(frame_idx)})
            n += 1
    save_json({"frames": meta}, os.path.join(out_dir, "meta.json"))
    return n, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=SUBSET_PATH)
    ap.add_argument("--frames-dir", default=FRAMES_DIR)
    ap.add_argument("--frames16-dir", default=FRAMES16_DIR)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=64)
    ap.add_argument("--n16", type=int, default=16)
    ap.add_argument("--video-root", default=VIDEO_ROOT,
                    help="Directory holding A1_JAKE/DAY*/ mp4 files; replaces the captions' "
                         "'data/EgoLife/' prefix")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    subset = load_subset(args.subset)
    print(f"Video root: {args.video_root}")
    missing_videos = set()
    done = skipped = 0
    for e in subset["entries"]:
        qid, clips = e["ID"], e["target_clips"]
        resolved = [resolve_video_path(vp, args.video_root) for vp in clips]
        if any(not os.path.exists(p) for p in resolved):
            missing_videos.update(p for p in resolved if not os.path.exists(p))
            continue

        readers = [VideoReader(p, ctx=cpu(0)) for p in resolved]
        per_clip = proportional_cap([frame_indices_1fps(vr, args.fps) for vr in readers], args.max_frames)
        n, was_skipped = write_frames(os.path.join(args.frames_dir, qid), clips, per_clip, args.overwrite,
                                      args.video_root)
        skipped += was_skipped
        done += not was_skipped
        if not was_skipped:
            print(f"[{qid}] frames: {n} from {len(clips)} clip(s)")

        if e.get("masking"):
            per16 = [frame_indices_uniform(vr, args.n16) for vr in readers]
            n, was_skipped = write_frames(os.path.join(args.frames16_dir, qid), clips, per16, args.overwrite,
                                          args.video_root)
            if not was_skipped:
                print(f"[{qid}] frames16: {n}")

    print(f"done: {done} | skipped (already cached): {skipped}")
    if missing_videos:
        print(f"WARNING: {len(missing_videos)} target clip files missing, e.g. {sorted(missing_videos)[:3]}")


if __name__ == "__main__":
    main()
