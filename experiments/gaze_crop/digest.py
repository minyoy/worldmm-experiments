#!/usr/bin/env python3
"""
One screen of numbers for the morning after an overnight run.

Collects whatever exists -- stage-1 recall, the transform choice, all-clips recall, stage-2 accuracy
-- and prints the few comparisons the experiment turns on, with the baselines it has to beat. Missing
pieces are reported as missing rather than skipped, so a run that died at 3am is obvious.

    python experiments/gaze_crop/digest.py
    python experiments/gaze_crop/digest.py --markdown experiments/gaze_crop/analysis/DIGEST.md
"""

import argparse
import glob
import re
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import ANALYSIS_DIR, EXP_DIR, RESULTS_DIR, load_json  # noqa: E402

# from ../visual_bottleneck: the lines a gaze arm has to clear
BASELINES = {"B (텍스트만)": 40.8, "E (에이전트 자율)": 41.7, "E' (저자 임베딩, visual 강제)": 45.8}


def acc(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    rows = load_json(path)
    return 100 * sum(int(r["evaluate"]) for r in rows) / len(rows) if rows else None


def recall_block(path: str, label: str, out: List[str]) -> None:
    if not os.path.exists(path):
        out.append(f"### {label}\n\nMISSING ({path})\n")
        return
    d = load_json(path)
    ks = d["ks"]
    out.append(f"### {label}\n")
    out.append(f"pool {os.path.basename(d['pool'])}, {d['n_scored']}/{d['n_questions']} 문항 채점"
               + (" **[self-test: 숫자 무의미]**" if d.get("self_test") else "") + "\n")
    out.append("| arm | " + " | ".join(f"R@{k}" for k in ks) + " | median rank |")
    out.append("|---|" + "---|" * (len(ks) + 1))
    for arm, a in d["arms"].items():
        out.append(f"| `{arm}` | " + " | ".join(f"{100 * a['recall'][str(k)]:.1f}%" for k in ks)
                   + f" | {a['median_rank']} |")
    out.append("")
    if d.get("pairs"):
        out.append(f"paired diff at k={d['main_k']}:")
        for p in d["pairs"]:
            # 엇갈린 문항 수와 정확검정 p값만 싣는다. 부트스트랩 CI 는 불일치가 적을 때
            # 거의 자동으로 0 을 벗어나서 뺐다 (gaze_common.exact_paired_p 주석 참고).
            n_d = p.get("n_discordant", p["a_only"] + p["b_only"])
            if "mcnemar_p" in p:
                sig = "" if p["mcnemar_p"] < 0.05 else " — **유의하지 않음**"
                stat = f"불일치 {n_d}개 중 {p['a']} {p['a_only']} : {p['b']} {p['b_only']}, p={p['mcnemar_p']:.3f}{sig}"
            else:                       # 부트스트랩 제거 이전에 만든 결과 파일
                stat = (f"{p['a']} only {p['a_only']}, {p['b']} only {p['b_only']} "
                        f"(옛 형식: 정확검정 p값 없음)")
            out.append(f"- `{p['a']}` − `{p['b']}` = **{p['diff_pp']:+.1f}pp** ({stat})")
        out.append("")
    audio = (d.get("breakdowns") or {}).get("need_audio")
    if audio:
        out.append("need_audio 분해 (False 행이 실제 여유분):")
        for v, row in audio.items():
            cells = ", ".join(f"{k} {row[k]:.1f}%" for k in row if k != "n")
            out.append(f"- {v} (n={row['n']}): {cells}")
        out.append("")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--qa-dir", default=os.path.join(EXP_DIR, "results_qa"))
    ap.add_argument("--markdown", default=os.path.join(ANALYSIS_DIR, "DIGEST.md"))
    args = ap.parse_args()

    out: List[str] = ["# gaze-crop 실험 다이제스트", ""]

    out.append("### gaze→이미지 transform\n")
    used = None
    gp = os.path.join(EXP_DIR, "gaze_points.json")
    if os.path.exists(gp):
        used = load_json(gp).get("transform")
        out.append(f"- 실제로 쓴 값: **{used}** (사람이 고름; 근거 이미지는 `overlay/`, `variants/`)")
    else:
        out.append("- `gaze_points.json` 없음 — gaze 없이 돌았을 수 있음")
    tc_path = os.path.join(ANALYSIS_DIR, "transform_choice.json")
    if os.path.exists(tc_path):
        tc = load_json(tc_path)
        out.append(f"- 후보를 그린 조건: {tc.get('n_clips')}클립, crop {tc.get('ratio')}, "
                   f"axis {tc.get('axis', 'radial')}, min-offset {tc.get('min_offset')} "
                   f"(중앙 이탈 부족으로 제외 {tc.get('n_skipped_too_central')}개)")
    out.append("")

    recall_block(os.path.join(args.results_dir, "recall_question.json"), "1단계 recall (623클립 풀)", out)
    recall_block(os.path.join(args.results_dir, "recall_allclips.json"),
                 "2단계 recall (6,223클립 = 실제 인덱스 크기)", out)

    out.append("### 2단계 정확도 (조건 E′, 인덱스만 교체)\n")
    qa = sorted(glob.glob(os.path.join(args.qa_dir, "*", "E_prime.json")))
    if not qa:
        out.append(f"MISSING — {args.qa_dir}/*/E_prime.json 없음 (1단계에서 중단됐거나 QA 미실행)\n")
    else:
        out.append("| arm | 정확도 |")
        out.append("|---|---|")
        for name, v in BASELINES.items():
            out.append(f"| _{name}_ | _{v:.1f}%_ |")
        # run_stage2.sh writes results_qa/<arm>_k<K>/, one dir per (arm, retrieval depth). Each k is
        # its own condition -- a k=20 arm must never be differenced against a k=3 baseline -- so
        # group by k and compare inside the group. Dirs with no _k suffix are older runs, which
        # were all k=3 (eval_egolife.py's --visual-top-k default).
        vals: Dict[str, float] = {}
        by_k: Dict[str, Dict[str, float]] = {}
        for p in qa:
            arm = os.path.basename(os.path.dirname(p))
            a = acc(p)
            if a is None:
                continue
            vals[arm] = a
            out.append(f"| `{arm}` | **{a:.1f}%** |")
            m = re.search(r"_k(\d+)$", arm)
            k = m.group(1) if m else "3"
            by_k.setdefault(k, {})[arm[: m.start()] if m else arm] = a
        out.append("")
        for k in sorted(by_k, key=int):
            g = by_k[k]
            if "full" not in g:
                out.append(f"- k={k}: `full` 기준선이 없어 비교 불가 (있는 arm: "
                           f"{', '.join('`%s`' % n for n in sorted(g))})")
                continue
            for arm, a in sorted(g.items()):
                if arm != "full":
                    out.append(f"- k={k}: `{arm}` − `full` = **{a - g['full']:+.1f}pp** "
                               f"(같은 풀·같은 플래그·같은 k의 자체 대조)")
        out.append("")
        out.append("파서 확인: `python experiments/gaze_crop/rescore.py "
                   f"{args.qa_dir}/*/E_prime.json`\n")

    md = "\n".join(out)
    print(md)
    if args.markdown:
        os.makedirs(os.path.dirname(args.markdown), exist_ok=True)
        with open(args.markdown, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\nwrote {args.markdown}")


if __name__ == "__main__":
    main()
