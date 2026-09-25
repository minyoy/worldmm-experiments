#!/usr/bin/env python3
"""
논문 수치(EgoLifeQA A1_JAKE 500문항)와 정면 비교하기 위한 보조 스크립트.

두 가지 모드가 있다.

1) 나머지 서브셋 만들기 (기본)
   main 서브셋(subset.json, 120문항)에 없는 380문항을 subset_rest.json 으로 쓴다.
   oracle 클립 조건(1~10개)은 조건 C/D 용이라 여기서는 적용하지 않는다. 조건 E는 target 클립이
   필요 없으므로 클립이 0개이거나 수천 개인 문항도 전부 포함한다 (그래야 500이 된다).

2) 합산 리포트 (--report)
   results/E.json (120) + results_rest/E.json (380) 을 합쳐 500문항 정확도를 낸다.
   유형별·gap별 분해와 95% 신뢰구간도 같이 출력한다.

실행 (저장소 루트에서):
    python experiments/visual_bottleneck/build_rest_subset.py
    CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
      python experiments/visual_bottleneck/eval_egolife.py --condition E --resume \\
        --subset experiments/visual_bottleneck/subset_rest.json \\
        --results-dir experiments/visual_bottleneck/results_rest
    python experiments/visual_bottleneck/build_rest_subset.py --report

results-dir 를 따로 주는 이유: eval_egolife.py 는 results/{condition}.json 에 쓰므로, 같은 폴더를
쓰면 이미 끝난 120문항짜리 E.json 을 덮어쓴다.
"""

import argparse
import collections
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CAPTION_30SEC, EXP_DIR, QA_PATH, QA_TYPES, RESULTS_DIR, SUBSET_PATH, build_choices, clip_display_key,
    gap_bin, gap_days, load_json, query_time_int, save_json, target_clips,
)

REST_PATH = os.path.join(EXP_DIR, "subset_rest.json")
REST_RESULTS_DIR = os.path.join(EXP_DIR, "results_rest")


def build_rest(args):
    qa = load_json(args.qa)
    caps = load_json(args.captions)
    main_ids = {e["ID"] for e in load_json(args.main)["entries"]}
    entries = []
    for row in qa:
        if str(row["ID"]) in main_ids:
            continue
        clips = target_clips(row, caps)
        gap = gap_days(row)
        entries.append({
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
            "masking": False,
        })
    entries.sort(key=lambda e: int(e["ID"]))
    save_json({"note": "all EgoLifeQA questions not in subset.json; for condition E reproduction only",
               "n": len(entries), "entries": entries}, args.out)
    print(f"QA {len(qa)}문항 − main {len(main_ids)} = rest {len(entries)}문항 → {args.out}")
    print("유형:", dict(collections.Counter(e["type"] for e in entries)))
    print("target 클립 0개(조건 E에는 무관):", sum(1 for e in entries if e["n_target_clips"] == 0))


def report(args):
    paths = [os.path.join(RESULTS_DIR, f"{args.condition}.json"),
             os.path.join(REST_RESULTS_DIR, f"{args.condition}.json")]
    rows = {}
    for p in paths:
        if os.path.exists(p):
            for x in load_json(p):
                rows[str(x["ID"])] = x
            print(f"{p}: {len(load_json(p))}문항")
        else:
            print(f"{p}: 없음")
    r = list(rows.values())
    n = len(r)
    if not n:
        sys.exit("결과 없음")
    acc = sum(x["evaluate"] for x in r) / n
    se = math.sqrt(acc * (1 - acc) / n)
    print(f"\n조건 {args.condition}: {n}문항 정확도 {acc*100:.1f}%  (95% CI ±{1.96*se*100:.1f})")
    print(f"논문 WorldMM-8B: 56.4%  → 격차 {56.4 - acc*100:+.1f}%p")
    for name, key in [("유형", lambda x: x["type"]), ("need_audio", lambda x: bool(x["need_audio"])),
                      ("gap", lambda x: x.get("gap_bin"))]:
        by = collections.defaultdict(lambda: [0, 0])
        for x in r:
            by[key(x)][0] += x["evaluate"]; by[key(x)][1] += 1
        print(f"\n[{name}]")
        for k in sorted(by, key=str):
            c, t = by[k]
            print(f"  {str(k):13s} {c:3d}/{t:3d} = {c/t*100:5.1f}%")
    if n < 500:
        print(f"\n아직 {500-n}문항 남음 (results_rest 에 --resume 으로 이어서 실행)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=QA_PATH)
    ap.add_argument("--captions", default=CAPTION_30SEC)
    ap.add_argument("--main", default=SUBSET_PATH, help="이미 돌린 main 서브셋")
    ap.add_argument("--out", default=REST_PATH)
    ap.add_argument("--report", action="store_true", help="main + rest 결과를 합쳐 500문항 정확도 출력")
    ap.add_argument("--condition", default="E")
    args = ap.parse_args()
    report(args) if args.report else build_rest(args)


if __name__ == "__main__":
    main()
