#!/usr/bin/env python3
"""
Build the retrieval pool: the clips to search over, plus the questions to search with.

Questions come from questions_500.json (all 500 EgoLifeQA A1_JAKE questions, built by
build_questions.py). Not visual_bottleneck/subset.json (120): recall scoring costs nothing per
question, and 120 gave only 4 discordant pairs -- too few for any test to conclude with.

Why not the full 6,223 clips: an arm costs one VLM2Vec forward pass per clip, and the question
here is only whether cropping moves the ranking at all. If gaze-cropping cannot win at
1-in-~600 it will not win at 1-in-3000, so this is the cheap kill switch. Distractors are drawn
from the days the targets live on, which keeps them visually close (same flat, same people)
instead of trivially separable.

    python experiments/gaze_crop/build_pool.py [--n-distractors 500] [--seed 0]
    python experiments/gaze_crop/build_pool.py --distractor-scope all   # any day, not just target days

Writes pool.json:
    clips      [{key, video_path, date, start_time, end_time, ts_start, ts_end, is_target}]
    questions  [{ID, type, need_audio, gap_bin, question, keywords, query_time, target_keys}]
"""

import argparse
import collections
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    CAPTION_30SEC, POOL_PATH, QUESTIONS_PATH, clip_key, load_json, load_subset, save_json, ts_int,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", dest="subset", default=QUESTIONS_PATH,
                    help="문항 파일. 기본값은 500문항 전체(build_questions.py 가 만든다).")
    ap.add_argument("--captions", default=CAPTION_30SEC)
    ap.add_argument("--out", default=POOL_PATH)
    ap.add_argument("--n-distractors", type=int, default=500)
    ap.add_argument("--distractor-scope", choices=("same-day", "all"), default="same-day",
                    help="'same-day': only clips from days that hold a target (default, harder). "
                         "'all': any 30-sec clip.")
    ap.add_argument("--all-clips", action="store_true",
                    help="put EVERY 30-sec clip in the pool, not a sample. This is the setting the "
                         "shipped visual_embeddings.pkl indexes, so it is what an accuracy run needs "
                         "-- and it costs ~6,200 forward passes per arm instead of ~620.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if not os.path.exists(args.subset):
        raise SystemExit(f"문항 파일이 없다: {args.subset}\n"
                         f"    python experiments/gaze_crop/build_questions.py")
    subset = load_subset(args.subset)
    caps = load_json(args.captions)
    by_path = {c["video_path"]: c for c in caps}

    target_paths, questions = [], []
    seen_targets = set()
    missing_caption = []
    for e in subset["entries"]:
        for vp in e["target_clips"]:
            if vp not in by_path:
                missing_caption.append(vp)
                continue
            if vp not in seen_targets:
                seen_targets.add(vp)
                target_paths.append(vp)
        questions.append({
            "ID": e["ID"],
            "type": e["type"],
            "need_audio": e["need_audio"],
            "gap_bin": e["gap_bin"],
            "question": e["question"],
            "keywords": e.get("keywords", ""),
            "query_time": e["query_time"],
            # the raw annotation, so recall_eval can score against the instant rather than against
            # the clip this repo derived from it
            "target_time": e.get("target_time"),
            "target_keys": [clip_key(vp) for vp in e["target_clips"] if vp in by_path],
        })

    target_days = {by_path[vp]["date"] for vp in target_paths}
    pool_candidates = [
        c for c in caps
        if c["video_path"] not in seen_targets
        and (args.all_clips or args.distractor_scope == "all" or c["date"] in target_days)
    ]
    if args.all_clips:
        distractors = pool_candidates
    else:
        rng.shuffle(pool_candidates)
        distractors = pool_candidates[:args.n_distractors]

    def row(c, is_target):
        return {
            "key": clip_key(c["video_path"]),
            "video_path": c["video_path"],
            "date": c["date"],
            "start_time": str(c["start_time"]),
            "end_time": str(c["end_time"]),
            "ts_start": ts_int(c["date"], c["start_time"]),
            "ts_end": ts_int(c["date"], c["end_time"]),
            "is_target": is_target,
        }

    clips = [row(by_path[vp], True) for vp in target_paths] + [row(c, False) for c in distractors]
    clips.sort(key=lambda r: r["ts_start"])
    keys = [r["key"] for r in clips]
    assert len(set(keys)) == len(keys), "clip keys are not unique -- clip_key() needs fixing"

    n_no_target = sum(1 for q in questions if not q["target_keys"])
    visible = [sum(1 for r in clips if r["ts_end"] <= q["query_time"]) for q in questions]
    per_day = collections.Counter(r["date"] for r in clips)

    save_json({
        "seed": args.seed,
        "n_distractors": len(distractors),
        "distractor_scope": "all-clips" if args.all_clips else args.distractor_scope,
        "n_clips": len(clips),
        "n_targets": len(target_paths),
        "clips": clips,
        "questions": questions,
    }, args.out)

    print(f"questions: {len(questions)} ({n_no_target} with no usable target clip)")
    print(f"pool: {len(clips)} clips = {len(target_paths)} targets + {len(distractors)} distractors")
    print(f"days: {dict(sorted(per_day.items()))}")
    print(f"visible pool per question: min {min(visible)}, median {sorted(visible)[len(visible)//2]}, "
          f"max {max(visible)}")
    if missing_caption:
        print(f"WARNING: {len(missing_caption)} target clip paths are not in the 30-sec captions, "
              f"e.g. {missing_caption[:2]}")
    if args.n_distractors > len(pool_candidates):
        print(f"WARNING: asked for {args.n_distractors} distractors, only {len(pool_candidates)} available")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
