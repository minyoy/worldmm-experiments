#!/usr/bin/env python3
"""
Example figure for the four crop arms: the same clip at the same moments, with each arm's crop box
drawn on the real frame (outside the box dimmed). Rows = arms, columns = moments. The wearer's gaze
point is marked in every panel, so it shows where `center` misses the gaze and how `gazef` / `randf`
move per frame.

Frame times, gaze lookup and crop boxes are the ones embed_arms.py uses (16 uniform frames per clip,
gaze_at(), crop_box()), so the boxes are exactly what the embeddings saw. Needs decord + PIL (the
project venv); CPU only.

    python experiments/gaze_crop/plot_arm_examples.py                 # auto-pick a clip
    python experiments/gaze_crop/plot_arm_examples.py --clip DAY3_A1_JAKE_12103000
"""

import argparse
import json
import math
import os
import sys
from typing import List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gaze_common import GAZE_PATH, crop_box, gaze_at  # noqa: E402
from embed_arms import decode  # noqa: E402

VB = os.path.abspath(os.path.join(HERE, "..", "visual_bottleneck"))
sys.path.insert(0, VB)
from common import VIDEO_ROOT, resolve_video_path  # noqa: E402

RATIO = 0.5
NFRAMES = 16
RAND_PATH = os.path.join(HERE, "gaze_random_frames.json")
POOL_PATH = os.path.join(HERE, "pool_all.json")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# (row label, sub-label, box color) -- same colors as the recall curves
ROWS = [
    ("full", "no crop (baseline)", (137, 135, 129)),
    ("center", "fixed centre crop", (235, 104, 52)),
    ("gazef", "crop follows the gaze", (42, 120, 214)),
    ("randf", "random spot, per frame", (27, 175, 122)),
]


def frame_gaze(entry, idx: List[int], fps: float) -> List[Tuple[float, float]]:
    return [gaze_at(entry, i / fps) for i in idx]


def pick_clip(gaze, n_candidates: int = 400) -> List[str]:
    """Clips whose gaze strays furthest from the centre at the 16 embedded frame times."""
    scored = []
    for key, e in gaze.items():
        s = e.get("samples") or []
        if not s:
            continue
        dur = max(float(x[0]) for x in s)
        pts = [gaze_at(e, dur * i / (NFRAMES - 1)) for i in range(NFRAMES)]
        offs = sorted(math.hypot(x - 0.5, y - 0.5) for x, y in pts)
        scored.append((sum(offs[-4:]) / 4, key))    # mean of the 4 largest offsets
    scored.sort(reverse=True)
    return [k for _, k in scored[:n_candidates]]


def choose_moments(pts: List[Tuple[float, float]], k: int = 3) -> List[int]:
    """k frame positions (of 16) with large gaze offset, spread out in time."""
    order = sorted(range(len(pts)), key=lambda i: -math.hypot(pts[i][0] - 0.5, pts[i][1] - 0.5))
    chosen: List[int] = []
    for i in order:
        if all(abs(i - j) >= 4 for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return sorted(chosen)


def main() -> None:
    from PIL import Image, ImageDraw, ImageFont

    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None, help="clip key; default picks a bright clip with large gaze offsets")
    ap.add_argument("--out", default=os.path.join(HERE, "analysis", "arms_example.jpg"))
    ap.add_argument("--cell", type=int, default=330, help="thumbnail side in px")
    ap.add_argument("--candidates", type=int, default=40,
                    help="how many large-gaze-offset clips to decode when auto-picking (brightest wins)")
    args = ap.parse_args()

    gaze = json.load(open(GAZE_PATH))["clips"]
    rand = json.load(open(RAND_PATH))["clips"]
    paths = {c["key"]: c["video_path"] for c in json.load(open(POOL_PATH))["clips"]}

    # among the clips whose gaze strays furthest from the centre, keep the brightest one
    candidates = [args.clip] if args.clip else pick_clip(gaze, n_candidates=args.candidates)
    best = None
    for key in candidates:
        if key not in rand or key not in paths:
            continue
        arr, idx, fps = decode(resolve_video_path(paths[key], VIDEO_ROOT), NFRAMES)
        gpts = frame_gaze(gaze[key], idx, fps)
        cols = choose_moments(gpts)
        brightness = sum(arr[c].mean() for c in cols) / len(cols)
        if best is None or brightness > best[0]:
            best = (brightness, key, arr, idx, fps, gpts, cols)
    if best is None:
        raise SystemExit("no usable clip; pass --clip")
    brightness, key, arr, idx, fps, gpts, cols = best
    rpts = frame_gaze(rand[key], idx, fps)
    print(f"clip {key}  frames {[idx[c] for c in cols]}  brightness {brightness:.0f}")

    cell, gap, left, top = args.cell, 14, 250, 70
    Wd = left + len(cols) * (cell + gap) + 10
    Ht = top + len(ROWS) * (cell + gap) + 60
    fig = Image.new("RGB", (Wd, Ht), "white")
    d = ImageDraw.Draw(fig)
    f_b, f, f_s = ImageFont.truetype(FONT_B, 20), ImageFont.truetype(FONT, 17), ImageFont.truetype(FONT, 15)

    for ci, c in enumerate(cols):
        x = left + ci * (cell + gap)
        d.text((x + cell / 2, top - 30), f"t = {idx[c] / fps:.1f} s", font=f_b, fill=(26, 26, 26), anchor="mm")

    for ri, (name, sub, col) in enumerate(ROWS):
        y = top + ri * (cell + gap)
        d.text((20, y + cell / 2 - 14), name, font=f_b, fill=(26, 26, 26), anchor="lm")
        d.text((20, y + cell / 2 + 14), sub, font=f_s, fill=(85, 85, 85), anchor="lm")
        for ci, c in enumerate(cols):
            frame = Image.fromarray(arr[c]).convert("RGB")
            w, h = frame.size
            if name == "full":
                box = (0, 0, w, h)
            elif name == "center":
                box = crop_box(w, h, 0.5, 0.5, RATIO)
            elif name == "gazef":
                box = crop_box(w, h, gpts[c][0], gpts[c][1], RATIO)
            else:
                box = crop_box(w, h, rpts[c][0], rpts[c][1], RATIO)
            # dim outside the crop box
            dim = Image.blend(frame, Image.new("RGB", frame.size, (0, 0, 0)), 0.62)
            dim.paste(frame.crop(box), box[:2])
            thumb = dim.resize((cell, cell), Image.LANCZOS)
            td = ImageDraw.Draw(thumb)
            sc = cell / w
            if name != "full":
                bx = [round(v * sc) for v in box]
                td.rectangle(bx, outline=col, width=4)
            gx, gy = gpts[c][0] * cell, gpts[c][1] * cell      # the wearer's gaze, in every row
            td.ellipse([gx - 9, gy - 9, gx + 9, gy + 9], fill=(255, 214, 0), outline=(0, 0, 0), width=2)
            fig.paste(thumb, (left + ci * (cell + gap), y))

    ly = Ht - 34
    d.ellipse([left, ly - 9, left + 18, ly + 9], fill=(255, 214, 0), outline=(0, 0, 0), width=2)
    d.text((left + 28, ly), f"= the wearer's gaze at that moment.  Box = the {RATIO} crop the arm embeds; "
           f"outside is dimmed.  Clip {key}", font=f_s, fill=(60, 60, 60), anchor="lm")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.save(args.out, quality=88)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
