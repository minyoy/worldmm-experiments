#!/usr/bin/env python3
"""EgoLife eye-gaze CSVs -> gaze_points.json, without the WorldMM repo (standard library only).

Same numbers as prepare_gaze.py with its defaults (equidistant fisheye, focal 611 px at 1408 px,
transform flipx+flipy), for every CSV under --gaze-root instead of only the clips in a pool file.
Use it when another codebase only needs the gaze points.

Input: one CSV per 30 s clip, named like the clip's mp4, 10 Hz Aria MPS eye gaze
    huggingface-cli download Wangtwohappy/EgoLife_EyeTracking_EyeGaze --repo-type dataset \
        --include "EyeGaze/A1_JAKE/*" --local-dir data/EgoLife
    -> data/EgoLife/EyeGaze/A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.csv

    python gaze_points_standalone.py --gaze-root data/EgoLife/EyeGaze/A1_JAKE --out gaze_points.json

Output:
    {"transform": "flipx+flipy", ...,
     "clips": {"DAY1_A1_JAKE_11094208": {"median": [x, y], "samples": [[t_sec, x, y], ...], ...}}}
x, y are in [0, 1] with the origin at the frame's top-left; t_sec is seconds from the clip start.
gaze_at() and crop_box() at the bottom are how the experiments read the file (copied from
gaze_common.py).

The projection is approximate: the gaze release ships no device calibration, so it uses one focal
length for Aria's ~110 degree fisheye and ignores the eye-to-camera offset (tens of px).
"""

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple

ARIA_FOCAL_PX = 611.0          # Aria RGB focal at its native 1408 px width
ARIA_NATIVE_W = 1408
CLIP_SECONDS = 30.0


def combined_gaze(left_yaw: float, right_yaw: float) -> float:
    """Cyclopean yaw: atan of the MEAN OF THE TANGENTS of the two eyes' yaw (not of the angles)."""
    return math.atan((math.tan(left_yaw) + math.tan(right_yaw)) / 2)


def make_projector(w: int, h: int, focal_px: float):
    """(yaw, pitch) in radians -> normalised image point, equidistant fisheye r = f * theta."""
    def project(yaw: float, pitch: float) -> Tuple[float, float]:
        vx, vy, vz = math.tan(yaw), math.tan(pitch), 1.0   # +x right, +y down, +z forward
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        theta = math.acos(max(-1.0, min(1.0, vz / n)))
        r = focal_px * theta
        rho = math.hypot(vx, vy) or 1e-9
        return 0.5 + r * (vx / rho) / w, 0.5 + r * (vy / rho) / h
    return project


def apply_transform(xn: float, yn: float, how: str) -> Tuple[float, float]:
    """Ops joined by '+', applied left to right. EgoLife needs flipx+flipy (a 180 degree turn)."""
    for op in how.split("+"):
        op = op.strip()
        if op == "rot90cw":
            xn, yn = 1.0 - yn, xn
        elif op == "rot90ccw":
            xn, yn = yn, 1.0 - xn
        elif op == "flipx":
            xn = 1.0 - xn
        elif op == "flipy":
            yn = 1.0 - yn
        elif op not in ("", "none"):
            raise ValueError(f"unknown transform op '{op}'")
    return xn, yn


def parse_csv(path: str, project, transform: str) -> Tuple[List[List[float]], int]:
    """One clip's CSV -> [[t_sec, x, y], ...] (device clock, not yet rebased), clamped count."""
    samples, clipped = [], 0
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                t = float(r["tracking_timestamp_us"]) / 1e6
                yaw = combined_gaze(float(r["left_yaw_rads_cpf"]), float(r["right_yaw_rads_cpf"]))
                pitch = float(r["pitch_rads_cpf"])
            except (KeyError, TypeError, ValueError):
                continue
            xn, yn = apply_transform(*project(yaw, pitch), transform)
            # gaze past the frame edge is real; clamp so a crop box still follows it to the edge
            cx, cy = min(max(xn, 0.0), 1.0), min(max(yn, 0.0), 1.0)
            clipped += (cx != xn) or (cy != yn)
            samples.append([t, cx, cy])
    return samples, clipped


def rebase(samples: List[List[float]]) -> Tuple[List[List[float]], bool]:
    """Timestamps are a device clock; shift so the clip's first sample is t=0."""
    ts = [s[0] for s in samples if s[0] >= 0]
    if not ts:
        return samples, False
    t0 = min(ts)
    if 0.0 <= t0 <= CLIP_SECONDS + 5 and max(ts) <= CLIP_SECONDS + 10:
        return samples, False                                  # already clip-relative
    return [[s[0] - t0 if s[0] >= 0 else -1.0, s[1], s[2]] for s in samples], True


def median_xy(samples: List[List[float]]) -> List[float]:
    xs = sorted(s[1] for s in samples)
    ys = sorted(s[2] for s in samples)
    m = len(xs) // 2
    return [xs[m], ys[m]]


# ---------------------------------------------------------------------------
# Reading the file (same as gaze_common.py)
# ---------------------------------------------------------------------------

def gaze_at(entry: Dict, t_sec: float, tol: float = 0.5) -> Optional[Tuple[float, float]]:
    """Gaze at a moment inside the clip: nearest sample within `tol` s, else the clip median."""
    if entry.get("samples"):
        best = min(entry["samples"], key=lambda s: abs(float(s[0]) - t_sec))
        if abs(float(best[0]) - t_sec) <= tol:
            return float(best[1]), float(best[2])
    med = entry.get("median")
    return (float(med[0]), float(med[1])) if med else None


def crop_box(w: int, h: int, cx_norm: float, cy_norm: float, ratio: float = 0.5) -> Tuple[int, int, int, int]:
    """Square box of side ratio * short side centred on the gaze, pushed back inside the frame.
    PIL: image.crop(crop_box(*image.size, x, y, 0.5))."""
    side = max(1, int(round(min(w, h) * ratio)))
    cx, cy = cx_norm * w, cy_norm * h
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return x0, y0, x0 + side, y0 + side


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gaze-root", required=True, help="folder with <clip>.csv files (walked recursively)")
    ap.add_argument("--out", default="gaze_points.json")
    ap.add_argument("--width", type=int, default=1408, help="video frame width (EgoLife: 1408)")
    ap.add_argument("--height", type=int, default=1408)
    ap.add_argument("--transform", default="flipx+flipy",
                    help="coordinate-frame fix; flipx+flipy is the one picked for EgoLife")
    args = ap.parse_args()

    focal = ARIA_FOCAL_PX * args.width / ARIA_NATIVE_W
    project = make_projector(args.width, args.height, focal)
    clips: Dict[str, Dict] = {}
    for dirpath, _, filenames in os.walk(args.gaze_root):
        for fn in sorted(filenames):
            if not fn.lower().endswith(".csv"):
                continue
            path = os.path.join(dirpath, fn)
            samples, clipped = parse_csv(path, project, args.transform)
            if not samples:
                print(f"skip {fn}: no usable rows")
                continue
            samples, rebased = rebase(samples)
            clips[os.path.splitext(fn)[0]] = {"median": median_xy(samples), "samples": samples,
                                              "n": len(samples), "clipped": clipped, "rebased": rebased}

    with open(args.out, "w") as f:
        json.dump({"transform": args.transform, "projection": "equidistant", "focal_px": focal,
                   "frame_size": [args.width, args.height], "clips": clips}, f)
    print(f"{len(clips)} clips -> {args.out}")


if __name__ == "__main__":
    main()
