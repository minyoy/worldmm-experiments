#!/usr/bin/env python3
"""
Step 0: is the gaze release actually on this machine, and is it the shape we expect?

EgoLife's gaze ships separately from the videos:

    huggingface-cli download Wangtwohappy/EgoLife_EyeTracking_EyeGaze --repo-type dataset \
        --include "EyeGaze/A1_JAKE/*" --local-dir data/EgoLife

which gives EyeGaze/A1_JAKE/DAY{1..7}/<clip stem>.csv -- one 10 Hz Aria MPS eye-gaze CSV per 30-sec
clip, named exactly like the clip's mp4. This walks a directory, reports what is there, prints a
header, and says how many files line up with the clips in pool.json.

    python experiments/gaze_crop/check_gaze.py            # 기본 경로: data/EgoLife/EyeGaze/<subject>

Exit status is 0 when candidate gaze files were found, 1 when none were, so run_stage1.sh can branch
on it.
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import GAZE_ROOT, POOL_PATH, load_json  # noqa: E402

DOWNLOAD_HINT = ('    huggingface-cli download Wangtwohappy/EgoLife_EyeTracking_EyeGaze '
                 '--repo-type dataset \\\n        --include "EyeGaze/A1_JAKE/*" '
                 '--local-dir data/EgoLife')

GAZE_HINTS = ("gaze", "eye", "eyetrack", "et_", "pupil", "fixation")
GAZE_EXT = (".csv", ".json", ".jsonl", ".npy", ".npz", ".txt", ".vrs", ".parquet")
MAX_PRINT = 12


def head_bytes(path: str, n: int = 400) -> str:
    try:
        with open(path, "rb") as f:
            return f.read(n).decode("utf-8", "replace").replace("\n", " | ")[:n]
    except OSError as e:
        return f"<unreadable: {e}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaze-root", "--video-root", dest="gaze_root", default=GAZE_ROOT,
                    help=f"directory to walk (default: {GAZE_ROOT})")
    ap.add_argument("--pool", default=POOL_PATH, help="to count how many pool clips have a CSV")
    ap.add_argument("--depth", type=int, default=5, help="how many directory levels to walk")
    args = ap.parse_args()

    root = args.gaze_root
    print(f"walking: {root}")
    if not os.path.isdir(root):
        print(f"  MISSING. Download it:\n{DOWNLOAD_HINT}")
        return 1

    ext_counts = collections.Counter()
    candidates, matched_files = [], []
    dirs_seen = 0
    base_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.count(os.sep) - base_depth >= args.depth:
            dirnames[:] = []
        dirs_seen += 1
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            ext_counts[ext] += 1
            low = fn.lower()
            if any(h in low for h in GAZE_HINTS) or ext == ".vrs":
                if ext in GAZE_EXT or ext == "":
                    candidates.append(os.path.join(dirpath, fn))

    print(f"walked {dirs_seen} directories (depth <= {args.depth})")
    print("file extensions:")
    for ext, n in ext_counts.most_common(15):
        print(f"  {ext or '<none>':>10}  {n}")

    # the useful number: how many pool clips have a same-stem CSV next to them
    matched = 0
    if os.path.exists(args.pool):
        keys = {r["key"] for r in load_json(args.pool)["clips"]}
        stems, all_files = set(), []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.lower().endswith((".csv", ".json")):
                    stems.add(os.path.splitext(f)[0])
                    all_files.append(os.path.join(dirpath, f))
        matched = len(keys & stems)
        matched_files = sorted(p for p in all_files
                               if os.path.splitext(os.path.basename(p))[0] in keys)
        print(f"\npool clips with a same-stem gaze file: {matched}/{len(keys)}")
        if matched == 0 and stems:
            print("  none matched. The CSVs must be named like the clips "
                  "(DAY1_A1_JAKE_11100000.csv); day-level files work too but need absolute times.")

    if not candidates and not matched:
        print("\nNo gaze-looking files here. Where it lives:")
        print(DOWNLOAD_HINT)
        print("\nUntil then, run without arm C: `--arms full center@0.5 center@0.3`")
        print("A vs B still answers 'is the clip embedding diluted by background?'")
        return 1

    if matched and not candidates:
        # the per-clip CSVs are named after the clip, so the "gaze"/"eye" hints never fire on them
        candidates = matched_files[:MAX_PRINT]

    print(f"\n{len(candidates)} file(s) to look at:")
    for p in candidates[:MAX_PRINT]:
        size = os.path.getsize(p) if os.path.exists(p) else -1
        print(f"\n  {p}  ({size / 1e6:.2f} MB)")
        if os.path.splitext(p)[1].lower() in (".csv", ".json", ".jsonl", ".txt"):
            print(f"    head: {head_bytes(p)}")
        elif p.lower().endswith(".vrs"):
            print("    VRS stream: needs projectaria_tools to read; "
                  "prepare_gaze.py --format aria_yawpitch expects the exported "
                  "general_eye_gaze.csv, not the raw VRS.")
    if len(candidates) > MAX_PRINT:
        print(f"\n  ... and {len(candidates) - MAX_PRINT} more")
    print(f"\nNext: python experiments/gaze_crop/prepare_gaze.py --gaze-root {root} --dump-variants 3")
    print("(the Aria CSVs hold yaw/pitch, not pixels -- pick --transform from the variant grid "
          "before embedding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
