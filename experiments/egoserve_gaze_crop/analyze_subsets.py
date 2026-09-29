#!/usr/bin/env python3
"""
채점 결과(results/recall_egoserve_{query}_tol{T}.json)를 정답의 성격별 그룹으로 다시 나눠 본다.

그룹 정의는 build_egoserve_pool.py 의 group_of 참고. 추가로 두 묶음을 만든다.
  visual_findable  own_action + long_ago              시각으로 찾을 수 있고 충분히 먼 것
  other            heard_speech + just_before + minutes_before

n 이 작아서(7~44) 검정은 순위 Wilcoxon 만 한다(n>=10). 셀은 top-k 안에 정답이 든 질의 수 / 채점된 질의 수.

    python experiments/egoserve_gaze_crop/analyze_subsets.py
"""
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = ["full", "center_05", "gazef_05_randf", "gazef_05"]
KS = (3, 10, 20)
GROUPS = {
    "all": None,
    "visual_findable": {"own_action", "long_ago"},
    "  own_action": {"own_action"},
    "  long_ago": {"long_ago"},
    "other": {"heard_speech", "just_before", "minutes_before"},
    "  heard_speech": {"heard_speech"},
    "  just_before": {"just_before"},
    "  minutes_before": {"minutes_before"},
}


def main():
    pool = json.load(open(os.path.join(HERE, "pool_egoserve.json")))
    grp = {q["ID"]: q["group"] for q in pool["questions"]}
    qs = sys.argv[1] if len(sys.argv) > 1 else "pastobs"
    out = [f"# EgoServe retrieval by answer type (query = description of the target past scene)\n",
           "cells = queries with the target in top-k / scored queries; median = median rank of the target\n"]
    for tol in (0, 30, 60):
        d = json.load(open(os.path.join(HERE, "results", f"recall_egoserve_{qs}_tol{tol}.json")))
        pq = [q for q in d["per_question"] if q.get("rank")]
        out.append(f"\n## tol {tol}s\n")
        out.append("| group | n | arm | R@3 | R@10 | R@20 | median rank |")
        out.append("|---|---|---|---|---|---|---|")
        for name, gs in GROUPS.items():
            sub = [q for q in pq if gs is None or grp[q["ID"]] in gs]
            for i, a in enumerate(ARMS):
                r = [q["rank"][a] for q in sub]
                cells = " | ".join(f"{sum(x <= k for x in r)}/{len(r)}" for k in KS)
                out.append(f"| {name if i == 0 else ''} | {len(sub) if i == 0 else ''} | `{a}` | {cells} | {int(np.median(r))} |")
        out.append("\ngazef vs other arms, rank (better:worse questions, Wilcoxon p on log rank; n>=10 only):\n")
        out.append("| group | n | vs full | vs center | vs randf |")
        out.append("|---|---|---|---|---|")
        for name, gs in GROUPS.items():
            sub = [q for q in pq if gs is None or grp[q["ID"]] in gs]
            row = []
            for b in ("full", "center_05", "gazef_05_randf"):
                u = sum(q["rank"]["gazef_05"] < q["rank"][b] for q in sub)
                w = sum(q["rank"]["gazef_05"] > q["rank"][b] for q in sub)
                p = wilcoxon(np.log([q["rank"][b] for q in sub]) - np.log([q["rank"]["gazef_05"] for q in sub])).pvalue if len(sub) >= 10 else float("nan")
                row.append(f"{u}:{w} (p={p:.2f})" if p == p else f"{u}:{w}")
            out.append(f"| {name.strip()} | {len(sub)} | " + " | ".join(row) + " |")
    path = os.path.join(HERE, "analysis", "tables", f"subsets_{qs}.md")
    open(path, "w").write("\n".join(out) + "\n")
    print("\n".join(out))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
