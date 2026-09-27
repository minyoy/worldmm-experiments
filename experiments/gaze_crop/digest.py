#!/usr/bin/env python3
"""
One screen of numbers for the morning after an overnight run.

Collects whatever exists -- the transform choice, every recall table in results/, stage-2 accuracy
-- and prints the few comparisons the experiment turns on, with the baselines it has to beat. Missing
pieces are reported as missing rather than skipped, so a run that died at 3am is obvious.

    python experiments/gaze_crop/digest.py
    python experiments/gaze_crop/digest.py --markdown experiments/gaze_crop/analysis/DIGEST.md

The recall blocks are listed in BLOCKS, weakest evidence first, because the ones that decide the
experiment are the 500-question runs (run_stage{1,2}.sh only produce the 120-question ones: the 120
subset turned out to be an unrepresentatively hard sample -- see README section 3.6 -- and 4
discordant questions cannot settle anything). The headline at the top is picked by scored-question
count, so whatever the most powered run on disk says is the first thing printed.
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

# (파일, 제목, 필수, 없을 때 알려줄 명령, tolerance 폴백) -- 검정력이 약한 것부터.
# 필수=True 는 run_stage{1,2}.sh 가 만드는 파일이라 없으면 실행이 죽은 것이고,
# 필수=False 는 손으로 돌리는 확장 채점이라 "없음 + 만드는 방법"으로 적는다.
# tolerance 폴백은 recall_eval 이 그 값을 결과 파일에 적기 시작하기 전(2026-09-27)에 만들어진
# 파일에만 쓰인다. 새로 돌린 파일은 자기 안에 값을 갖고 있으므로 이 폴백을 무시한다.
# 파일 이름 규칙: recall_{문항집합}_{tolerance}.json
#   문항집합  stage1 = 623클립 풀 / main = 120문항 subset / holdout = 나머지 380 / all = 500문항 전부
#   tolerance tol0 = 엄격(태그된 30초 칸만) / tol60 = ±60초 안이면 적중
RECALL_ARGS = ("--pool experiments/gaze_crop/pool_all.json --arms full gazef@0.5")
STAGE1_ARGS = ("--pool experiments/gaze_crop/pool.json")

# gaze_points.json 과 results_qa/ 는 .gitignore 대상이다. 그 둘이 없는 곳(= 실행 머신이 아닌 곳)에서
# digest 를 돌리면 transform 과 2단계 정확도가 통째로 사라지므로, 값을 적어 둔 기록본을 쓴다.
# 원본이 있으면 항상 원본이 이긴다. 기록본에서 온 숫자는 출력에 '기록본'으로 표시된다.
RECORDED_PATH = os.path.join(ANALYSIS_DIR, "stage2_recorded.json")
BLOCKS = [
    ("recall_stage1_question_tol0.json", "1단계 recall (623클립 풀, 120문항, 질문 질의)", True, None, 0.0),
    ("recall_stage1_keywords_tol0.json", "1단계 recall (623클립 풀, 120문항, 키워드 질의)", False,
     f"recall_eval.py --query-source keywords {STAGE1_ARGS} "
     "--out results/recall_stage1_keywords_tol0.json "
     "--markdown analysis/recall_stage1_keywords_tol0.md", 0.0),
    ("recall_main_tol0.json", "2단계 recall (6,223클립 = 실제 인덱스, 120문항, tol 0s)", True, None, 0.0),
    ("recall_main_tol60.json", "같은 120문항, tolerance 60s", False,
     f"recall_eval.py {RECALL_ARGS} --target-tolerance-sec 60 "
     "--out results/recall_main_tol60.json --markdown analysis/recall_main_tol60.md", 60.0),
    ("recall_holdout_tol0.json", "held-out 380문항 (120문항 제외), tolerance 0s", False,
     "build_pool.py --all-clips --questions <holdout 380> --out pool_holdout.json 후 "
     "recall_eval.py --pool pool_holdout.json --arms full gazef@0.5", 0.0),
    ("recall_holdout_tol60.json", "held-out 380문항, tolerance 60s", False,
     "위와 같고 --target-tolerance-sec 60", 60.0),
    ("recall_all_tol0.json", "★ A1_JAKE 500문항 전부 (6,223클립), tolerance 0s (엄격)", False,
     "merge_recall.py results/recall_main_tol0.json results/recall_holdout_tol0.json "
     "--out results/recall_all_tol0.json --markdown analysis/recall_all_tol0.md "
     "--pool pool_holdout.json --questions questions_500.json --tolerance 0", 0.0),
    ("recall_all_tol60.json", "★ A1_JAKE 500문항 전부 (6,223클립), tolerance 60s", False,
     "build_questions.py 후 build_pool.py --all-clips, 그리고 "
     f"recall_eval.py {RECALL_ARGS} --target-tolerance-sec 60 "
     "--out results/recall_all_tol60.json --markdown analysis/recall_all_tol60.md", 60.0),
    ("recall_arms_tol0.json", "★★ 대조군 포함 4 arm, 500문항, tolerance 0s (엄격)", False,
     "bash experiments/gaze_crop/run_controls.sh  (center@0.5 + gazef_05_randf 임베딩 후 채점)", 0.0),
    ("recall_arms_tol60.json", "★★ 대조군 포함 4 arm, 500문항, tolerance 60s", False,
     "bash experiments/gaze_crop/run_controls.sh", 60.0),
]


def tol_note(d: Dict[str, Any], fallback: Optional[float] = None) -> str:
    """결과 파일의 tolerance 를 사람이 읽는 한 줄로.

    recall_eval 이 이 값을 결과에 적기 전에 만들어진 파일에는 기록이 없다. 그때는 BLOCKS 의
    폴백(그 파일을 만든 명령에 있던 값)을 쓰고, 추정이라는 것을 표시한다. tolerance 를 모른 채
    두 표를 나란히 읽으면 tolerance 가 올려준 몫을 arm 의 이득으로 착각하게 된다.
    """
    tol = d.get("target_tolerance_sec")
    inferred = tol is None
    if inferred:
        if fallback is None:
            return "tolerance 미기록 (파일에도 없고 폴백도 없음 — 다시 채점할 것)"
        tol = fallback
    tol = float(tol)
    note = ("tolerance 0s (엄격: EgoLife 가 태그한 30초 칸만 정답)" if tol == 0 else
            f"tolerance {tol:.0f}s (정답 칸이 늘어나므로 엄격 표와 직접 비교 금지)")
    return note + (" **[파일에 기록 없음 — 실행 명령에서 추정]**" if inferred else "")


def headline(blocks: List[Dict[str, Any]], out: List[str]) -> None:
    """가장 많은 문항으로 채점된 결과의 주요 비교를 맨 위에 한 줄로 올린다."""
    if not blocks:
        out += ["### 헤드라인", "", "recall 결과 파일이 하나도 없음 — 1단계부터 확인", ""]
        return
    def one(b: Dict[str, Any], prefix: str) -> None:
        """한 결과의 주요 비교를 이긴 arm 이 앞에 오도록 적는다.

        recall_eval 은 arm 이름 순서로 뺀 값을 저장하므로 부호만으로는 어느 쪽이 좋은지 알 수
        없다 -- 아침에 한 줄만 읽고 오해하기 쉬운 지점이라 여기서 뒤집는다.
        """
        d = b["d"]
        out.append(f"{prefix}: **{b['label']}** — {d['n_scored']}문항, "
                   f"{tol_note(d, b.get('tol_fallback'))} (`results/{b['file']}`)")
        for p in d.get("pairs", []):
            a, bb, diff = p["a"], p["b"], p["diff_pp"]
            if diff < 0:
                a, bb, diff = bb, a, -diff
            n_d = p.get("n_discordant", p["a_only"] + p["b_only"])
            pv = p.get("mcnemar_p")
            sig = "" if pv is None else (f", p={pv:.3f}"
                                        + ("" if pv < 0.05 else " — **유의하지 않음**"))
            verdict = "동률" if diff == 0 else f"`{a}` 가 `{bb}` 보다 **{diff:+.1f}pp**"
            out.append(f"  - k={p['k']}: {verdict} (불일치 {n_d}건{sig})")

    def tol_of(b: Dict[str, Any]) -> Optional[float]:
        t = b["d"].get("target_tolerance_sec", b.get("tol_fallback"))
        return None if t is None else float(t)

    # 문항 수가 같으면 arm 이 더 많은 표를 고른다: 대조군이 들어간 표가 판정에 필요한 표이므로.
    b = max(blocks, key=lambda b: (b["d"]["n_scored"], len(b["d"]["arms"])))
    d = b["d"]
    out += ["### 헤드라인", ""]
    one(b, "가장 검정력 높은 측정")
    # tolerance 를 올리면 두 arm 의 recall 이 같이 오른다. 헤드라인이 tol>0 이면 엄격 표도 같이
    # 보여 주지 않으면 이득이 tolerance 때문인지 arm 때문인지 한 화면에서 알 수 없다.
    strict = [x for x in blocks if tol_of(x) == 0]
    if tol_of(b) != 0 and strict:
        one(max(strict, key=lambda x: (x["d"]["n_scored"], len(x["d"]["arms"]))),
            "같은 비교, 엄격 채점(tol 0s)")
    if d["n_scored"] <= 120:
        out.append("- ⚠ 120문항 이하로만 채점됨. 이 표본에서는 불일치가 몇 건뿐이라 어느 arm 도 "
                   "판정되지 않는다. `build_questions.py` → 500문항 채점을 돌릴 것.")
    out.append("")


def acc(path: str) -> Optional[float]:
    if not os.path.exists(path):
        return None
    rows = load_json(path)
    return 100 * sum(int(r["evaluate"]) for r in rows) / len(rows) if rows else None


def recall_block(path: str, label: str, out: List[str], required: bool = True,
                 hint: Optional[str] = None,
                 tol_fallback: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """결과 파일 하나를 표로. 파일이 없으면 그 사실(과 만드는 방법)을 적고 None."""
    if not os.path.exists(path):
        out.append(f"### {label}\n")
        out.append(("MISSING (%s)" % path) if required else
                   f"아직 없음 — 만들려면: `{hint}`" if hint else "아직 없음")
        out.append("")
        return None
    d = load_json(path)
    ks = d["ks"]
    out.append(f"### {label}\n")
    out.append(f"pool {os.path.basename(d['pool'])}, {d['n_scored']}/{d['n_questions']} 문항 채점, "
               f"{tol_note(d, tol_fallback)}"
               + (" **[self-test: 숫자 무의미]**" if d.get("self_test") else "") + "\n")
    out.append("| arm | " + " | ".join(f"R@{k}" for k in ks) + " | median rank |")
    out.append("|---|" + "---|" * (len(ks) + 1))
    for arm, a in d["arms"].items():
        out.append(f"| `{arm}` | " + " | ".join(f"{100 * a['recall'][str(k)]:.1f}%" for k in ks)
                   + f" | {a['median_rank']} |")
    # chance 행이 있으면 같이 싣는다. tolerance 를 올리면 recall 과 chance 가 같이 오르므로,
    # arm 이 실제로 검색을 하고 있는지는 이 행과의 거리로만 알 수 있다.
    ch = d.get("chance")
    if ch:
        out.append("| _chance_ | " + " | ".join(f"_{100 * ch['recall'][str(k)]:.2f}%_" for k in ks)
                   + f" | _{ch['median_rank']}_ |")
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
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--qa-dir", default=os.path.join(EXP_DIR, "results_qa"))
    ap.add_argument("--markdown", default=os.path.join(ANALYSIS_DIR, "DIGEST.md"))
    args = ap.parse_args()

    out: List[str] = ["# gaze-crop 실험 다이제스트", ""]
    recorded = load_json(RECORDED_PATH) if os.path.exists(RECORDED_PATH) else None

    # recall 블록을 먼저 만들어 둔다: 헤드라인이 그중 가장 검정력 높은 것을 골라야 하므로.
    blocks: List[Dict[str, Any]] = []
    body: List[str] = []
    for fname, label, required, hint, tol_fb in BLOCKS:
        d = recall_block(os.path.join(args.results_dir, fname), label, body, required, hint, tol_fb)
        if d and not d.get("self_test"):
            blocks.append({"label": label, "file": fname, "d": d, "tol_fallback": tol_fb})
    headline(blocks, out)

    out.append("### gaze→이미지 transform\n")
    used = None
    gp = os.path.join(EXP_DIR, "gaze_points.json")
    rec_tf = (recorded.get("transform") or {}) if recorded else {}
    if os.path.exists(gp):
        used = load_json(gp).get("transform")
        out.append(f"- 실제로 쓴 값: **{used}** (사람이 고름; 근거 이미지는 `overlay/`, `variants/`)")
    elif rec_tf.get("used"):
        used = rec_tf["used"]
        out.append(f"- 실제로 쓴 값: **{used}** — {rec_tf.get('chosen_by', '')} "
                   f"**[기록본: {os.path.basename(RECORDED_PATH)}, `gaze_points.json` 없음]**")
        if rec_tf.get("caption_score_note"):
            out.append(f"- 캡션 코사인 점수(참고): {rec_tf['caption_score_note']}")
    else:
        out.append("- `gaze_points.json` 없음 — gaze 없이 돌았을 수 있음")
    tc_path = os.path.join(ANALYSIS_DIR, "transform_choice.json")
    if os.path.exists(tc_path):
        tc = load_json(tc_path)
        out.append(f"- 후보를 그린 조건: {tc.get('n_clips')}클립, crop {tc.get('ratio')}, "
                   f"axis {tc.get('axis', 'radial')}, min-offset {tc.get('min_offset')} "
                   f"(중앙 이탈 부족으로 제외 {tc.get('n_skipped_too_central')}개)")
    out.append("")

    out += body

    out.append("### 2단계 정확도 (조건 E′, 인덱스만 교체)\n")
    qa = sorted(glob.glob(os.path.join(args.qa_dir, "*", "E_prime.json")))
    # results_qa/ 가 있으면 문항 단위 파일에서 직접 채점하고, 없으면 기록본의 숫자를 쓴다.
    measured: Dict[str, float] = {}
    for p in qa:
        a = acc(p)
        if a is not None:
            measured[os.path.basename(os.path.dirname(p))] = a
    rec_acc = (recorded.get("accuracy") or {}) if recorded else {}
    from_record = not measured and bool(rec_acc.get("arms"))
    if from_record:
        measured = {k: float(v) for k, v in rec_acc["arms"].items()}
    if not measured:
        out.append(f"MISSING — {args.qa_dir}/*/E_prime.json 없음 (1단계에서 중단됐거나 QA 미실행)\n")
    else:
        if from_record:
            out.append(f"**[기록본: {os.path.basename(RECORDED_PATH)} — `results_qa/` 가 이 머신에 "
                       f"없음. 출처: {rec_acc.get('question_set', '?')}, pool "
                       f"{rec_acc.get('pool', '?')}]**\n")
        out.append("| arm | 정확도 |")
        out.append("|---|---|")
        for name, v in BASELINES.items():
            out.append(f"| _{name}_ | _{v:.1f}%_ |")
        # run_stage2.sh writes results_qa/<arm>_k<K>/, one dir per (arm, retrieval depth). Each k is
        # its own condition -- a k=20 arm must never be differenced against a k=3 baseline -- so
        # group by k and compare inside the group. Dirs with no _k suffix are older runs, which
        # were all k=3 (eval_egolife.py's --visual-top-k default).
        by_k: Dict[str, Dict[str, float]] = {}
        for arm, a in sorted(measured.items()):
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
        out.append("- ⚠ 이 정확도는 visual_bottleneck 의 120문항 subset 에서만 잰 값이다. 그 표본에서는 "
                   "두 arm 의 recall@3 이 거의 0 이라(위 120문항 블록) 정확도가 달라질 재료 자체가 "
                   "없다. recall 이득이 있는 문항에서 재려면 나머지 380문항에 조건 B 를 먼저 돌려 "
                   "`visual_bottleneck/text_context/` 를 채워야 한다 (README 3.5, 6-4).")
        out.append("")
        if not from_record:
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
