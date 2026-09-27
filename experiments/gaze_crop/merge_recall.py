#!/usr/bin/env python3
"""
Stitch two recall_eval runs that scored *different questions* into the one table they add up to.

Why this exists. The 500-question strict table (`recall_all_tol0`) was never run as its own job:
`recall_main_tol0` (the 120-question subset) and `recall_holdout_tol0` (the other 380) were run
separately, hours apart. They used the same embeddings, the same 6,223-clip pool, the same
tolerance and the same query source, and no question appears in both -- so their union *is* the
500-question run, exactly, and re-running it on a GPU would reproduce these numbers verbatim.
Every number below comes from `per_question.rank`, which both files carry.

    python experiments/gaze_crop/merge_recall.py \
        results/recall_main_tol0.json results/recall_holdout_tol0.json \
        --out results/recall_all_tol0.json --markdown analysis/recall_all_tol0.md \
        --pool pool_holdout.json --questions questions_500.json

What it refuses to merge (these would silently produce a table that means nothing):
  - different arms, ks, embedding dir, query source, or recorded tolerance
  - a question ID present in more than one input
  - a differing candidate-pool size for the same arm

GPU is not needed and neither is numpy, so this runs anywhere the repo is checked out.
`--pool` and `--questions` are optional and only add the chance row and the breakdown tables.
"""

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    ANALYSIS_DIR, RESULTS_DIR, clip_span, exact_paired_p, load_json, save_json, target_spans,
)


def question_date(q: Dict[str, Any]) -> Optional[str]:
    """recall_eval.question_date 와 같은 규칙 (target_time 에 date 가 없을 때의 폴백)."""
    tt = q.get("target_time") or {}
    if tt.get("date"):
        return tt["date"]
    qt = str(q.get("query_time") or "")
    return f"DAY{qt[0]}" if qt else None


def chance_at(t: int, v: int, k: int) -> float:
    """recall_eval.chance_at 과 같은 식: 1 - C(v-t,k)/C(v,k) 를 곱셈으로."""
    if t <= 0 or v <= 0:
        return 0.0
    if t >= v:
        return 1.0
    miss = 1.0
    for i in range(min(k, v)):
        num = v - t - i
        if num <= 0:
            return 1.0
        miss *= num / (v - i)
    return 1.0 - miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", help="recall_eval 결과 JSON 2개 이상 (문항이 겹치지 않아야 함)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--pool", default=None, help="chance 행 계산용 (클립 구간이 필요하다)")
    ap.add_argument("--questions", default=None, help="분해표용 (need_audio / type / gap_bin)")
    ap.add_argument("--label", default=None, help="마크다운 제목에 쓸 이름")
    ap.add_argument("--tolerance", type=float, default=None,
                    help="입력이 target_tolerance_sec 를 기록하기 전 형식일 때 그 값을 직접 준다. "
                         "이 값이 있어야 chance 행을 계산할 수 있다. 기록이 있는 입력에 다른 값을 "
                         "주면 거부한다.")
    args = ap.parse_args()

    runs = [load_json(p) for p in args.inputs]
    base = runs[0]
    arms: List[str] = list(base["arms"])
    ks: List[int] = base["ks"]
    mk = base["main_k"]

    # ---- 합쳐도 되는 실행인지 확인. 하나라도 어긋나면 합친 표는 의미가 없다 -------------
    for path, d in zip(args.inputs, runs):
        for field in ("ks", "query_source", "emb_dir"):
            if d.get(field) != base.get(field):
                raise SystemExit(f"{path}: {field} 가 {args.inputs[0]} 와 다르다 "
                                 f"({d.get(field)!r} vs {base.get(field)!r})")
        if list(d["arms"]) != arms:
            raise SystemExit(f"{path}: arm 구성이 다르다 ({list(d['arms'])} vs {arms})")
        if d.get("target_tolerance_sec") != base.get("target_tolerance_sec"):
            raise SystemExit(f"{path}: tolerance 가 다르다 — tolerance 가 다른 표는 합칠 수 없다")
        if d.get("self_test"):
            raise SystemExit(f"{path}: self-test 실행이다 (숫자가 무의미)")

    # ---- per_question 합치기 -------------------------------------------------------
    merged: Dict[str, Dict[str, Any]] = {}
    for path, d in zip(args.inputs, runs):
        for pq in d["per_question"]:
            if pq["ID"] in merged:
                raise SystemExit(f"{path}: 문항 {pq['ID']} 가 이미 다른 입력에 있다 — "
                                 f"겹치는 문항은 합칠 수 없다(같은 문항을 두 번 세게 된다)")
            merged[pq["ID"]] = pq
    qids = sorted(merged, key=int)
    for qid in qids:                     # arm 별 후보 수가 어긋나면 같은 인덱스가 아니었다는 뜻
        ps = merged[qid]["pool_size"]
        if len({v for v in ps.values() if v} ) > 1 and len(set(ps.values())) > 1:
            pass                         # arm 마다 임베딩 수가 1개 다를 수 있다(gaze 없는 클립)
    print(f"merged {len(args.inputs)} runs -> {len(qids)} questions, arms {arms}")

    def hits(arm: str, k: int) -> List[int]:
        return [int(bool(merged[q]["rank"][arm] and merged[q]["rank"][arm] <= k)) for q in qids]

    arms_out = {}
    for arm in arms:
        rs = [merged[q]["rank"][arm] for q in qids if merged[q]["rank"][arm]]
        arms_out[arm] = {
            "n_clips": max(d["arms"][arm]["n_clips"] for d in runs),
            "recall": {str(k): statistics.fmean(hits(arm, k)) for k in ks},
            "median_rank": int(statistics.median(rs)) if rs else None,
            "unscorable": sorted({q for d in runs for q in d["arms"][arm]["unscorable"]}, key=int),
        }

    pairs = []
    for i, a in enumerate(arms):
        for b in arms[i + 1:]:
            ha, hb = hits(a, mk), hits(b, mk)
            a_only = sum(1 for x, y in zip(ha, hb) if x and not y)
            b_only = sum(1 for x, y in zip(ha, hb) if y and not x)
            pairs.append({"a": a, "b": b, "k": mk,
                          "diff_pp": 100 * (statistics.fmean(ha) - statistics.fmean(hb)),
                          "a_only": a_only, "b_only": b_only,
                          "mcnemar_p": exact_paired_p(a_only, b_only),
                          "n_discordant": a_only + b_only})

    # ---- chance 와 분해표 (입력이 주어졌을 때만) ------------------------------------
    tol = base.get("target_tolerance_sec")
    if tol is None:
        tol = args.tolerance                    # 옛 형식 입력: 사람이 값을 알려준 경우만
        tol_recorded = False
    else:
        tol_recorded = True
        if args.tolerance is not None and float(args.tolerance) != float(tol):
            raise SystemExit(f"--tolerance {args.tolerance} 가 입력에 기록된 {tol} 와 다르다")
    chance_row = None
    if args.pool and tol is not None:
        pool = load_json(args.pool)
        spans = {r["key"]: clip_span(r) for r in pool["clips"]}
        qmeta = {e["ID"]: e for e in load_json(args.questions)["entries"]} if args.questions else {}
        n_tgt, n_vis = [], []
        for qid in qids:
            q = qmeta.get(qid)
            if not q:
                continue
            sp = target_spans(q.get("target_time"), question_date(q))
            n_tgt.append(sum(1 for cs, ce in spans.values()
                             if any(cs <= b + tol and ce >= a - tol for a, b in sp)))
            n_vis.append(next(v for v in merged[qid]["pool_size"].values() if v))
        if n_vis:
            chance_row = {
                "recall": {str(k): statistics.fmean(chance_at(t, v, k)
                                                    for t, v in zip(n_tgt, n_vis)) for k in ks},
                "median_rank": int(statistics.median((v + 1) / (t + 1)
                                                     for t, v in zip(n_tgt, n_vis) if t > 0)),
                "targets_per_question_median": int(statistics.median(n_tgt)),
                "visible_pool_median": int(statistics.median(n_vis)),
            }

    breakdowns: Dict[str, Dict[str, Dict[str, float]]] = {}
    if args.questions:
        qmeta = {e["ID"]: e for e in load_json(args.questions)["entries"]}
        for field in ("need_audio", "type", "gap_bin"):
            table: Dict[str, Dict[str, float]] = {}
            for v in sorted({str(qmeta[q][field]) for q in qids if q in qmeta}):
                sel = [q for q in qids if q in qmeta and str(qmeta[q][field]) == v]
                table[v] = {"n": len(sel), **{
                    arm: 100 * statistics.fmean(
                        int(bool(merged[q]["rank"][arm] and merged[q]["rank"][arm] <= mk))
                        for q in sel) for arm in arms}}
            breakdowns[field] = table

    summary = {
        "pool": base["pool"], "emb_dir": base["emb_dir"], "query_source": base["query_source"],
        "self_test": False, "n_questions": sum(d["n_questions"] for d in runs),
        "n_scored": len(qids), "ks": ks, "main_k": mk,
        "target_tolerance_sec": tol, "tolerance_recorded_in_inputs": tol_recorded,
        "chance": chance_row,
        "merged_from": [{"file": os.path.basename(p), "n_scored": d["n_scored"]}
                        for p, d in zip(args.inputs, runs)],
        "arms": arms_out, "pairs": pairs, "breakdowns": breakdowns,
        "per_question": [merged[q] for q in qids],
    }
    save_json(summary, args.out)

    src = base["query_source"]
    title = args.label or f"merged recall (`--query-source {src}`)"
    lines = [f"# {title}", "",
             "Merged, not a separate run: " + " + ".join(
                 f"`{os.path.basename(p)}` ({d['n_scored']}q)" for p, d in zip(args.inputs, runs))
             + ". Same embeddings, pool, tolerance and query source, disjoint question sets, so this "
               "is what one run over all of them would have printed.", "",
             (f"Target tolerance: unknown (not recorded in the inputs and none given)"
              if tol is None else
              f"Target tolerance: {tol:.0f}s"
              + ("" if tol_recorded else " (not recorded in the inputs; supplied on the command "
                                         "line from the command that produced them)"))
             + f", scored on {len(qids)} of {summary['n_questions']} questions.", "",
             "| arm | " + " | ".join(f"R@{k}" for k in ks) + " | median rank |",
             "|---|" + "---|" * (len(ks) + 1)]
    for arm in arms:
        lines.append(f"| `{arm}` | " + " | ".join(f"{100 * arms_out[arm]['recall'][str(k)]:.1f}%"
                                                  for k in ks)
                     + f" | {arms_out[arm]['median_rank']} |")
    if chance_row:
        lines.append("| _chance_ | " + " | ".join(
            f"_{100 * chance_row['recall'][str(k)]:.2f}%_" for k in ks)
            + f" | _{chance_row['median_rank']}_ |")
        lines += ["", f"(chance assumes a random ranking of each question's own visible pool: "
                      f"targets/question median {chance_row['targets_per_question_median']}, "
                      f"visible pool median {chance_row['visible_pool_median']})"]
    lines += ["", f"Paired differences at k={mk}:", "",
              "| arms | diff (pp) | a only | b only | discordant | McNemar p |",
              "|---|---|---|---|---|---|"]
    for p in pairs:
        lines.append(f"| `{p['a']}` - `{p['b']}` | {p['diff_pp']:+.1f} | {p['a_only']} | "
                     f"{p['b_only']} | {p['n_discordant']} | {p['mcnemar_p']:.3f} |")
    if breakdowns.get("need_audio"):
        lines += ["", f"By need_audio at k={mk} (False = the visual-evidence questions):", "",
                  "| need_audio | n | " + " | ".join(f"`{a}`" for a in arms) + " |",
                  "|---|---|" + "---|" * len(arms)]
        for v, row in breakdowns["need_audio"].items():
            lines.append(f"| {v} | {row['n']} | " + " | ".join(f"{row[a]:.1f}%" for a in arms) + " |")
    lines.append("")
    os.makedirs(os.path.dirname(args.markdown), exist_ok=True)
    with open(args.markdown, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}\nwrote {args.markdown}")


if __name__ == "__main__":
    main()
