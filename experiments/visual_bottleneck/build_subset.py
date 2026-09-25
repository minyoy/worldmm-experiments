#!/usr/bin/env python3
"""
Build the 120-question Main set (+40 masking subset) for the visual-bottleneck experiment.

Stratified by type x need_audio x gap bin, with the allocation fixed in PLAN.md section 3.
Questions are eligible when their target_time maps to 1..10 thirty-second clips.

Usage (from repo root):
    python experiments/visual_bottleneck/build_subset.py [--seed 0] [--out experiments/visual_bottleneck/subset.json]
"""

import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CAPTION_30SEC, QA_PATH, QA_TYPES, SUBSET_PATH, build_choices, clip_display_key,
    gap_bin, gap_days, load_json, query_time_int, save_json, target_clips,
)

# type -> need_audio -> (gap0, gap1, gap2+)
ALLOCATION = {
    "EntityLog":    {False: (7, 6, 6), True: (3, 1, 1)},
    "RelationMap":  {False: (7, 4, 6), True: (5, 1, 1)},
    "EventRecall":  {False: (5, 4, 4), True: (6, 3, 2)},
    "TaskMaster":   {False: (2, 1, 1), True: (10, 5, 5)},
    "HabitInsight": {False: (8, 6, 3), True: (4, 2, 1)},
}
GAP_BINS = ["0", "1", "2+"]

# Masking set: need_audio=False and exactly one target clip, drawn from the main set.
MASKING_ALLOCATION = {"EntityLog": 14, "RelationMap": 12, "EventRecall": 9, "HabitInsight": 5}

MIN_CLIPS, MAX_CLIPS = 1, 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=QA_PATH)
    ap.add_argument("--captions", default=CAPTION_30SEC)
    ap.add_argument("--out", default=SUBSET_PATH)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    qa = load_json(args.qa)
    caps = load_json(args.captions)

    # ---- annotate every row -------------------------------------------------
    pool = []
    excluded = collections.Counter()
    for row in qa:
        clips = target_clips(row, caps)
        if len(clips) < MIN_CLIPS:
            excluded["no_clip"] += 1
            continue
        if len(clips) > MAX_CLIPS:
            excluded["too_many_clips"] += 1
            continue
        gap = gap_days(row)
        pool.append({
            "ID": str(row["ID"]),
            "type": row["type"],
            "need_audio": bool(row["need_audio"]),
            "gap": gap,
            "gap_bin": gap_bin(gap),
            "query_time": query_time_int(row),
            "question": row["question"],
            "choices": build_choices(row),
            "answer": row["answer"],
            "keywords": row.get("keywords", ""),
            "reason": row.get("reason", ""),
            "target_time": row["target_time"],
            "target_clips": [c["video_path"] for c in clips],
            "target_clip_keys": [clip_display_key(c) for c in clips],
            "n_target_clips": len(clips),
        })
    print(f"QA rows: {len(qa)} | eligible: {len(pool)} | excluded: {dict(excluded)}")

    # ---- stratified sampling ------------------------------------------------
    strata = collections.defaultdict(list)
    for e in pool:
        strata[(e["type"], e["need_audio"], e["gap_bin"])].append(e)

    selected = []
    shortfall = []
    for qtype in QA_TYPES:
        for need_audio in (False, True):
            for gbin, quota in zip(GAP_BINS, ALLOCATION[qtype][need_audio]):
                cands = list(strata[(qtype, need_audio, gbin)])
                rng.shuffle(cands)
                if qtype == "HabitInsight" and not need_audio:
                    # few single-clip HabitInsight questions exist; prefer them so the
                    # masking set can meet its quota (stable sort keeps the shuffle order)
                    cands.sort(key=lambda e: e["n_target_clips"] != 1)
                take = cands[:quota]
                if len(take) < quota:
                    shortfall.append((qtype, need_audio, gbin, quota, len(take)))
                selected.extend(take)

    if shortfall:
        print("WARNING: strata with fewer candidates than quota:")
        for s in shortfall:
            print("   ", s)

    # ---- masking subset -----------------------------------------------------
    masking_ids = set()
    for qtype, quota in MASKING_ALLOCATION.items():
        cands = [e for e in selected if e["type"] == qtype and not e["need_audio"] and e["n_target_clips"] == 1]
        rng.shuffle(cands)
        if len(cands) < quota:
            print(f"WARNING: masking quota for {qtype} is {quota} but only {len(cands)} single-clip candidates")
        masking_ids.update(e["ID"] for e in cands[:quota])
    for e in selected:
        e["masking"] = e["ID"] in masking_ids

    selected.sort(key=lambda e: int(e["ID"]))

    # ---- report -------------------------------------------------------------
    by_type = collections.Counter(e["type"] for e in selected)
    by_audio = collections.Counter(e["need_audio"] for e in selected)
    by_gap = collections.Counter(e["gap_bin"] for e in selected)
    print(f"Main set: {len(selected)} | by type {dict(by_type)} | need_audio {dict(by_audio)} | gap {dict(by_gap)}")
    print(f"Masking set: {sum(e['masking'] for e in selected)} | by type "
          f"{dict(collections.Counter(e['type'] for e in selected if e['masking']))}")
    print(f"Unique target clips: {len({vp for e in selected for vp in e['target_clips']})}")

    save_json({
        "seed": args.seed,
        "allocation": {t: {str(k): v for k, v in a.items()} for t, a in ALLOCATION.items()},
        "masking_allocation": MASKING_ALLOCATION,
        "n_main": len(selected),
        "n_masking": sum(e["masking"] for e in selected),
        "entries": selected,
    }, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
