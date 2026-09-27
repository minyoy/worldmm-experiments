#!/usr/bin/env python3
"""
EgoLifeQA A1_JAKE 500문항 전체를 이 실험이 쓰는 형식으로 뽑는다.

왜 500 전부인가. 이 실험의 지표는 recall 이고, recall 채점에는 LLM 도 캐시도 필요 없다 --
클립 임베딩은 질문과 무관하게 이미 만들어져 있으므로, 문항을 늘리는 비용이 사실상 0 이다.
반면 검정력은 문항 수가 전부다: 120문항에서는 짝지은 불일치가 4건뿐이라 어떤 통계로도
결론이 안 났고, 500문항에서는 55건이 되어 결론이 난다.

(참고로 visual_bottleneck 의 subset.json 은 120문항이다. 그쪽은 문항마다 LLM 을 돌려야 해서
크기를 줄일 이유가 있었지만, 여기는 그 제약이 없다.)

    python experiments/gaze_crop/build_questions.py

쓰는 파일 questions_500.json:
    entries [{ID, type, need_audio, gap, gap_bin, query_time, question, choices, answer,
              keywords, reason, target_time, target_clips, target_clip_keys, n_target_clips}]

target_clips 는 visual_bottleneck/common.py 의 파생값이라 연쇄 타임스탬프를 구간으로 잘못
읽는 버그가 있다(16문항, 최대 2,430칸). recall_eval 은 그 필드를 쓰지 않고 target_time 에서
직접 계산하므로(gaze_common.target_spans) 문제가 되지 않는다. 필드를 남기는 것은 다른
스크립트와의 형식 호환 때문이다.
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import CAPTION_30SEC, EXP_DIR, QA_PATH, load_json, save_json  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "visual_bottleneck"))
from common import (  # noqa: E402
    build_choices, clip_display_key, gap_bin, gap_days, query_time_int, target_clips,
)

QUESTIONS_PATH = os.path.join(EXP_DIR, "questions_500.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=QA_PATH)
    ap.add_argument("--captions", default=CAPTION_30SEC)
    ap.add_argument("--out", default=QUESTIONS_PATH)
    args = ap.parse_args()

    qa = load_json(args.qa)
    caps = load_json(args.captions)
    entries = []
    for row in qa:
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
            "target_time": row["target_time"],        # ← 채점이 실제로 쓰는 값
            "target_clips": [c["video_path"] for c in clips],
            "target_clip_keys": [clip_display_key(c) for c in clips],
            "n_target_clips": len(clips),
            "masking": False,
        })
    entries.sort(key=lambda e: int(e["ID"]))
    save_json({"note": "all EgoLifeQA A1_JAKE questions; scoring uses target_time, not target_clips",
               "n": len(entries), "entries": entries}, args.out)

    print(f"{len(entries)}문항 -> {args.out}")
    print("유형:", dict(sorted(collections.Counter(e["type"] for e in entries).items())))
    print("need_audio:", dict(collections.Counter(e["need_audio"] for e in entries)))
    n0 = sum(1 for e in entries if e["n_target_clips"] == 0)
    print(f"파생 target_clips 가 0칸인 문항: {n0}개 "
          f"(주석 시각이 클립 사이 틈에 떨어진 경우. tolerance>=7s 면 정상 채점된다)")


if __name__ == "__main__":
    main()
