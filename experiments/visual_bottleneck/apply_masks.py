#!/usr/bin/env python3
"""
Exp.3 step 2: turn reviewed masks into the two masked frame sets.

    masked_frames/relevant/{qid}/{frame}.jpg     relevant region filled with gray
    masked_frames/irrelevant/{qid}/{frame}.jpg   a rectangle of the SAME area, not overlapping any
                                                  relevant mask of that question, at a position fixed
                                                  across all frames of the question
    masked_frames/meta.json                       per-qid areas, rectangle, source decisions

Inputs: frames16/{qid}/*.jpg, masks/{qid}/*.png, optional review_decisions.json exported from
review_masks.html ({qid: {"decision": "approve|reject|fix", "note": ...}}). Rejected questions are
skipped; "fix" entries whose note parses as JSON {frame: [[x1,y1,x2,y2], ...]} use those boxes
(a key "all" applies to every frame) instead of the detected masks.

Usage (from repo root):
    python experiments/visual_bottleneck/apply_masks.py [--decisions experiments/visual_bottleneck/review_decisions.json]
"""

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EXP_DIR, FRAMES16_DIR, MASKED_FRAMES_DIR, MASKS_DIR, load_json, save_json  # noqa: E402

GRAY = (128, 128, 128)


def boxes_to_mask(boxes: List[List[float]], h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(round(x2))), min(h, int(round(y2)))
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 255
    return m


def parse_fix_note(note: str) -> Optional[Dict[str, List[List[float]]]]:
    try:
        obj = json.loads(note)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def load_masks(qid: str, frames: List[str], masks_dir: str, fix: Optional[Dict], h: int, w: int) -> Dict[str, np.ndarray]:
    out = {}
    for f in frames:
        if fix is not None and (f in fix or "all" in fix):
            out[f] = boxes_to_mask(fix.get(f, fix.get("all", [])), h, w)
            continue
        p = os.path.join(masks_dir, qid, f.replace(".jpg", ".png"))
        out[f] = (np.array(Image.open(p).convert("L")) > 127).astype(np.uint8) * 255 if os.path.exists(p) \
            else np.zeros((h, w), dtype=np.uint8)
    return out


def place_irrelevant_rect(union: np.ndarray, area: int, rng: random.Random, tries: int = 2000) -> Tuple[int, int, int, int]:
    """Random rectangle of ~`area` pixels that avoids the union of relevant masks. Aspect ratio follows the
    union's bounding box when it exists. Falls back to the least-overlapping candidate."""
    h, w = union.shape
    ys, xs = np.where(union > 0)
    if len(xs):
        bw, bh = max(1, xs.max() - xs.min() + 1), max(1, ys.max() - ys.min() + 1)
        aspect = bw / bh
    else:
        aspect = 1.0
    rh = int(round((area / aspect) ** 0.5))
    rw = int(round(rh * aspect))
    rh, rw = max(1, min(rh, h)), max(1, min(rw, w))
    best, best_overlap = None, None
    for _ in range(tries):
        y = rng.randint(0, h - rh)
        x = rng.randint(0, w - rw)
        ov = int((union[y:y + rh, x:x + rw] > 0).sum())
        if ov == 0:
            return x, y, x + rw, y + rh
        if best_overlap is None or ov < best_overlap:
            best, best_overlap = (x, y, x + rw, y + rh), ov
    print(f"  warning: could not place a non-overlapping rectangle; using overlap={best_overlap} px")
    return best


def fill(img: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.array(img).copy()
    arr[mask > 0] = GRAY
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames16-dir", default=FRAMES16_DIR)
    ap.add_argument("--masks-dir", default=MASKS_DIR)
    ap.add_argument("--out-dir", default=MASKED_FRAMES_DIR)
    ap.add_argument("--decisions", default=os.path.join(EXP_DIR, "review_decisions.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-area-frac", type=float, default=0.002,
                    help="frames whose relevant mask is smaller than this fraction get no relevant fill (nothing to hide)")
    args = ap.parse_args()

    decisions = load_json(args.decisions) if os.path.exists(args.decisions) else {}
    if not decisions:
        print("no review_decisions.json found; applying every detected mask as-is")
    rng = random.Random(args.seed)
    meta: Dict[str, Dict] = {}

    qids = sorted(d for d in os.listdir(args.masks_dir) if os.path.isdir(os.path.join(args.masks_dir, d)))
    for qid in qids:
        dec = decisions.get(qid, {})
        if dec.get("decision") == "reject":
            print(f"[{qid}] rejected in review; skipped")
            continue
        fix = parse_fix_note(dec.get("note", "")) if dec.get("decision") == "fix" else None

        src = os.path.join(args.frames16_dir, qid)
        frames = sorted(f for f in os.listdir(src) if f.endswith(".jpg"))
        if not frames:
            continue
        w, h = Image.open(os.path.join(src, frames[0])).size
        masks = load_masks(qid, frames, args.masks_dir, fix, h, w)

        areas = [int((m > 0).sum()) for m in masks.values()]
        nonzero = [a for a in areas if a >= args.min_area_frac * h * w]
        if not nonzero:
            print(f"[{qid}] no usable relevant mask on any frame; skipped")
            continue
        mean_area = int(round(sum(nonzero) / len(nonzero)))
        union = np.zeros((h, w), dtype=np.uint8)
        for m in masks.values():
            union |= (m > 0).astype(np.uint8)
        rect = place_irrelevant_rect(union, mean_area, rng)
        irr_mask = boxes_to_mask([list(rect)], h, w)

        rel_dir = os.path.join(args.out_dir, "relevant", qid)
        irr_dir = os.path.join(args.out_dir, "irrelevant", qid)
        os.makedirs(rel_dir, exist_ok=True)
        os.makedirs(irr_dir, exist_ok=True)
        for f in frames:
            img = Image.open(os.path.join(src, f)).convert("RGB")
            m = masks[f] if (masks[f] > 0).sum() >= args.min_area_frac * h * w else np.zeros_like(masks[f])
            fill(img, m).save(os.path.join(rel_dir, f), quality=90)
            fill(img, irr_mask).save(os.path.join(irr_dir, f), quality=90)

        meta[qid] = {
            "frames": len(frames), "frames_with_mask": len(nonzero),
            "mean_relevant_area_px": mean_area, "mean_relevant_area_frac": round(mean_area / (h * w), 4),
            "irrelevant_rect": list(rect), "irrelevant_area_px": int((irr_mask > 0).sum()),
            "source": "fix" if fix else "detected", "decision": dec.get("decision"),
        }
        print(f"[{qid}] relevant {mean_area/(h*w)*100:.1f}% of frame on {len(nonzero)}/{len(frames)} frames; "
              f"irrelevant rect {rect}")

    save_json(meta, os.path.join(args.out_dir, "meta.json"))
    print(f"done: {len(meta)} questions -> {args.out_dir}")


if __name__ == "__main__":
    main()
