"""Convert StreamGaze step1 fixation output to the GazeQwen fixation CSV format.

GazeQwen's `{video}_fixation_filtered.csv` holds every step1 fixation (not the
step1.5-filtered subset) with the columns of demo_data/*_fixation_filtered.csv.
ScanpathLoader reads `duration`, not `duration_seconds`, so that column must exist.

Usage:
  python scripts/make_fixations.py --metadata_dirs DIR [DIR ...] --out_dir dataset/fixations
"""
import argparse
import glob
import os

import pandas as pd

COLUMNS = ["start_time", "end_time", "center_x", "center_y", "duration",
           "start_time_seconds", "end_time_seconds", "fixation_id", "duration_seconds"]


def convert(src: str) -> pd.DataFrame:
    df = pd.read_csv(src)
    # EGTEA step1 writes *_seconds columns; EgoExoLearn writes start_time/end_time.
    start = df["start_time_seconds"] if "start_time_seconds" in df else df["start_time"]
    end = df["end_time_seconds"] if "end_time_seconds" in df else df["end_time"]
    out = pd.DataFrame({
        "start_time": start,
        "end_time": end,
        "center_x": df["center_x"],
        "center_y": df["center_y"],
        "duration": end - start,
        "start_time_seconds": start,
        "end_time_seconds": end,
        "fixation_id": range(len(df)),
        "duration_seconds": end - start,
    })
    return out[COLUMNS]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata_dirs", nargs="+", required=True,
                   help="StreamGaze final_data/<dataset>/metadata directories")
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = 0
    for d in args.metadata_dirs:
        for src in sorted(glob.glob(os.path.join(d, "*", "*_fixation_dataset.csv"))):
            stem = os.path.basename(src)[: -len("_fixation_dataset.csv")]
            convert(src).to_csv(os.path.join(args.out_dir, f"{stem}_fixation_filtered.csv"), index=False)
            n += 1
    print(f"wrote {n} fixation CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
