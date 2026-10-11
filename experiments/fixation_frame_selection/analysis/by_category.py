#!/usr/bin/env python3
"""Per-category recall@3 for the fixation-selection arms, with paired differences.

Categories are the question `type` (5) and `need_audio`. Reads results/recall_pool_all_subset_r005_tol*.json
(per_question hit@3 / rank), so the numbers match the main tables. Only scored questions count (a
question whose target clip is missing from the pool has no rank and is skipped).

    python analysis/by_category.py     # -> analysis/tables/question_query/recall_by_category.md
"""
import json, os, statistics, sys
HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(EXP, "..", "gaze_crop"))
from gaze_common import QUESTIONS_PATH, load_json  # noqa: E402

ARMS = ["full", "fixsel_ctl", "fixsel", "center_05", "gazef_05", "fixsel_gazef_05"]
LABEL = {"full": "full", "fixsel_ctl": "fixsel_ctl", "fixsel": "fixsel", "center_05": "center",
         "gazef_05": "gazef", "fixsel_gazef_05": "fixsel_gazef"}
PAIRS = [("fixsel", "full"), ("fixsel", "fixsel_ctl"), ("fixsel_gazef_05", "gazef_05"), ("fixsel_gazef_05", "full")]
TOLS = ("0", "30", "60")

qs = {e["ID"]: e for e in load_json(QUESTIONS_PATH)["entries"]}
out = ["# 카테고리별 recall@3 (fixation frame selection)", "",
       "`type` 5개와 `need_audio`별 recall@3(%)이다. 채점된 문항만 센다(정답 클립이 풀에 없으면 제외). "
       "차이는 %p이고 `+a/−b`는 짝지은 문항 중 앞 arm만 맞힌 수 / 뒤 arm만 맞힌 수다. "
       "카테고리당 문항이 60~125개라 한 문항이 0.8~1.6 %p이므로 작은 차이는 노이즈다.", ""]
for tol in TOLS:
    d = load_json(os.path.join(EXP, "results", f"recall_pool_all_subset_r005_tol{tol}.json"))
    rows = d["per_question"]
    groups = {}
    for r in rows:
        q = qs[r["ID"]]
        for g in (f"type={q['type']}", "need_audio=" + ("True" if q.get("need_audio") else "False"), "ALL"):
            groups.setdefault(g, []).append(r)
    order = sorted(g for g in groups if g.startswith("type=")) + ["need_audio=False", "need_audio=True", "ALL"]
    out += [f"## tol {tol}s", "", "### recall@3 (%)", "",
            "| 카테고리 | n | " + " | ".join(LABEL[a] for a in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for g in order:
        rs = groups[g]
        cells = [f"{100 * sum(r['hit@3'].get(a, 0) for r in rs) / len(rs):.1f}" for a in ARMS]
        out.append(f"| {g} | {len(rs)} | " + " | ".join(cells) + " |")
    out += ["", "### 짝지은 차이 (%p, +앞 arm만/−뒤 arm만)", "",
            "| 카테고리 | " + " | ".join(f"{LABEL[a]} − {LABEL[b]}" for a, b in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
    for g in order:
        rs = groups[g]
        cells = []
        for a, b in PAIRS:
            w = sum(1 for r in rs if r["hit@3"].get(a) and not r["hit@3"].get(b))
            l = sum(1 for r in rs if r["hit@3"].get(b) and not r["hit@3"].get(a))
            cells.append(f"{100 * (w - l) / len(rs):+.1f} (+{w}/−{l})")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    out += ["", "### 정답 순위 중앙값 (낮을수록 좋음)", "",
            "| 카테고리 | " + " | ".join(LABEL[a] for a in ARMS) + " |", "|---|" + "---|" * len(ARMS)]
    for g in order:
        rs = groups[g]
        cells = []
        for a in ARMS:
            v = [r["rank"][a] for r in rs if r["rank"].get(a) is not None]
            cells.append(f"{statistics.median(v):.0f}" if v else "–")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    out.append("")
p = os.path.join(HERE, "tables", "question_query", "recall_by_category.md")
open(p, "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out))
