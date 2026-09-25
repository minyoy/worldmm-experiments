#!/usr/bin/env python3
"""
Aggregate results/{condition}.json into the tables PLAN.md asks for.

    accuracy per condition: overall, by need_audio, by type, by gap bin
    paired deltas with exact McNemar p-values and flip counts
        Exp.1  C-A, D-B, C-B
        Exp.2  D-E, E-B, D-E', E'-B
        Exp.3  C1-C2, C1-C3, D1-D2, D1-D3   (relevant / irrelevant mask drops)
    retrieval diagnostics for E and E': visual-search rate, query kind, recall@k against target clips

Writes analysis/summary.md and analysis/summary.json. Pure Python.

Usage (from repo root):
    python experiments/visual_bottleneck/analyze.py
"""

import argparse
import collections
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS_DIR, QA_TYPES, RESULTS_DIR, load_json, save_json  # noqa: E402

CONDITION_ORDER = ["A", "B", "C", "D", "E", "E_prime", "C1", "C2", "C3", "D1", "D2", "D3"]
LABEL = {
    "A": "Question only", "B": "Text memory", "C": "Oracle visual", "D": "Text + Oracle visual",
    "E": "Text + Retrieved visual (WorldMM)", "E_prime": "B ctx + Retrieved visual",
    "C1": "Oracle16 original", "C2": "Oracle16 relevant-mask", "C3": "Oracle16 irrelevant-mask",
    "D1": "Text + Oracle16 original", "D2": "Text + Oracle16 relevant-mask", "D3": "Text + Oracle16 irrelevant-mask",
}
PAIRS = [
    ("Exp.1", "C", "A", "visual evidence alone helps?"),
    ("Exp.1", "D", "B", "oracle visual gain on top of text"),
    ("Exp.1", "C", "B", "which modality is more usable (diagnostic)"),
    ("Exp.2", "E", "B", "retrieved visual gain on top of text (paper E+S+V - E+S)"),
    ("Exp.2", "D", "E", "retrieval bottleneck size (oracle - retrieved)"),
    ("Exp.2", "D", "E_prime", "oracle - retrieved with identical text context"),
    ("Exp.2", "E_prime", "B", "retrieved visual gain, text fixed"),
    ("Exp.3", "C1", "C2", "relevant mask drop (visual only)"),
    ("Exp.3", "C1", "C3", "irrelevant mask drop (visual only)"),
    ("Exp.3", "D1", "D2", "relevant mask drop (text + visual)"),
    ("Exp.3", "D1", "D3", "irrelevant mask drop (text + visual)"),
]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b: X right & Y wrong, c: X wrong & Y right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def acc(rows: List[Dict[str, Any]]) -> Optional[float]:
    return round(100 * sum(int(r["evaluate"]) for r in rows) / len(rows), 1) if rows else None


def fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}"


def load_results(results_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out = {}
    for cond in CONDITION_ORDER:
        p = os.path.join(results_dir, f"{cond}.json")
        if os.path.exists(p):
            out[cond] = {str(r["ID"]): r for r in load_json(p)}
    return out


def breakdown(res: Dict[str, Dict[str, Dict[str, Any]]], key, values, title: str, md: List[str], js: Dict) -> None:
    conds = [c for c in CONDITION_ORDER if c in res]
    md.append(f"\n### {title}\n")
    md.append("| " + " | ".join(["subset", "n"] + conds) + " |")
    md.append("|" + "---|" * (len(conds) + 2))
    js[title] = {}
    for v in values:
        row_ids = None
        cells = []
        for c in conds:
            rows = [r for r in res[c].values() if key(r) == v]
            row_ids = row_ids if row_ids is not None else len(rows)
            cells.append(fmt(acc(rows)))
            js[title].setdefault(str(v), {})[c] = acc(rows)
        md.append(f"| {v} | {row_ids or 0} | " + " | ".join(cells) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    ap.add_argument("--out-dir", default=ANALYSIS_DIR)
    args = ap.parse_args()

    res = load_results(args.results_dir)
    if not res:
        sys.exit(f"no results in {args.results_dir}")
    md: List[str] = ["# Visual bottleneck — results summary\n"]
    js: Dict[str, Any] = {"conditions": {}}

    # ---- overall ------------------------------------------------------------
    md.append("## Accuracy (%)\n")
    md.append("| condition | label | n | acc |")
    md.append("|---|---|---|---|")
    for c in CONDITION_ORDER:
        if c in res:
            rows = list(res[c].values())
            n_err = sum(1 for r in rows if r["response"] == "Error")
            md.append(f"| {c} | {LABEL[c]} | {len(rows)} | {fmt(acc(rows))}" + (f" ({n_err} errors)" if n_err else "") + " |")
            js["conditions"][c] = {"n": len(rows), "acc": acc(rows), "errors": n_err}

    breakdown(res, lambda r: bool(r["need_audio"]), [False, True], "by need_audio", md, js)
    breakdown(res, lambda r: r["type"], QA_TYPES, "by type", md, js)
    breakdown(res, lambda r: r.get("gap_bin"), ["0", "1", "2+"], "by gap (query day - target day)", md, js)

    # ---- paired comparisons -------------------------------------------------
    md.append("\n## Paired comparisons\n")
    md.append("delta = acc(X) - acc(Y) on questions present in both. b = X right & Y wrong, c = X wrong & Y right. "
              "p = exact McNemar (two-sided).\n")
    md.append("| exp | X - Y | n | acc X | acc Y | delta | b | c | p | meaning |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    js["pairs"] = []
    for exp, x, y, meaning in PAIRS:
        if x not in res or y not in res:
            continue
        ids = sorted(set(res[x]) & set(res[y]), key=int)
        if not ids:
            continue
        xs = [res[x][i] for i in ids]
        ys = [res[y][i] for i in ids]
        b = sum(1 for i in ids if res[x][i]["evaluate"] and not res[y][i]["evaluate"])
        c = sum(1 for i in ids if not res[x][i]["evaluate"] and res[y][i]["evaluate"])
        p = mcnemar_exact(b, c)
        d = round(acc(xs) - acc(ys), 1)
        md.append(f"| {exp} | {x} - {y} | {len(ids)} | {fmt(acc(xs))} | {fmt(acc(ys))} | {d:+.1f} | {b} | {c} | {p:.3f} | {meaning} |")
        js["pairs"].append({"exp": exp, "x": x, "y": y, "n": len(ids), "acc_x": acc(xs), "acc_y": acc(ys),
                            "delta": d, "b": b, "c": c, "p": round(p, 4)})

        # delta by need_audio for the core pairs
        if (x, y) in {("D", "B"), ("E", "B"), ("D", "E"), ("C", "A")}:
            for na in (False, True):
                sub = [i for i in ids if bool(res[x][i]["need_audio"]) == na]
                if sub:
                    dx = acc([res[x][i] for i in sub]); dy = acc([res[y][i] for i in sub])
                    md.append(f"|  | ↳ need_audio={na} | {len(sub)} | {fmt(dx)} | {fmt(dy)} | {dx - dy:+.1f} | | | | |")

    # ---- retrieval diagnostics ---------------------------------------------
    md.append("\n## Retrieval diagnostics (E, E')\n")
    js["retrieval"] = {}
    for c in ("E", "E_prime"):
        if c not in res:
            continue
        rows = list(res[c].values())
        used = [r for r in rows if r.get("visual_used")]
        hit = [r for r in used if set(r.get("retrieved_clips", [])) & set(r.get("target_clip_keys", []))]
        kinds = collections.Counter(r.get("visual_query_kind") for r in used)
        rounds = [r.get("num_rounds", 0) for r in rows]
        acc_used, acc_unused = acc(used), acc([r for r in rows if not r.get("visual_used")])
        md.append(f"- **{c}**: visual search used in {len(used)}/{len(rows)} questions "
                  f"(query kinds {dict(kinds)}); recall@k vs target clips {len(hit)}/{len(used)} "
                  f"= {fmt(100 * len(hit) / len(used) if used else None)}%; "
                  f"acc when visual used {fmt(acc_used)} vs not used {fmt(acc_unused)}; "
                  f"mean rounds {sum(rounds) / len(rounds):.2f}")
        js["retrieval"][c] = {"n": len(rows), "visual_used": len(used), "recall_hits": len(hit),
                              "query_kinds": dict(kinds), "acc_visual_used": acc_used, "acc_visual_unused": acc_unused,
                              "mean_rounds": round(sum(rounds) / len(rounds), 3) if rows else None}

    # ---- multiscale filter diagnostics (needs filter_calls in results) -------
    for c in ("B", "E"):
        rows = [r for r in res.get(c, {}).values() if r.get("target_in_filter_candidates") is not None]
        if not rows:
            continue
        n = len(rows)
        cand = sum(1 for r in rows if r["target_in_filter_candidates"])
        sel = sum(1 for r in rows if r["target_in_filter_selected"])
        lost = sum(1 for r in rows if r["target_in_filter_candidates"] and not r["target_in_filter_selected"])
        n_calls = sum(len(r.get("filter_calls", [])) for r in rows)
        gran = collections.Counter(s["granularity"] for r in rows for call in r.get("filter_calls", []) for s in call["selected"])
        md.append(f"- **{c} multiscale filter**: target among HippoRAG candidates {cand}/{n} = {100*cand/n:.1f}%; "
                  f"kept by the filter {sel}/{n} = {100*sel/n:.1f}%; seen-but-dropped {lost}/{n} = {100*lost/n:.1f}% "
                  f"({n_calls} filter calls; kept granularity mix {dict(gran)})")
        js["retrieval"].setdefault(c, {}).update({"filter_n": n, "target_in_candidates": cand,
                                                  "target_in_selected": sel, "seen_but_dropped": lost})

    # ---- interpretation hint ------------------------------------------------
    g = {(p["x"], p["y"]): p["delta"] for p in js["pairs"]}
    if ("D", "B") in g and ("E", "B") in g:
        oracle_gain, retr_gain = g[("D", "B")], g[("E", "B")]
        md.append("\n## Reading (PLAN.md section 1)\n")
        md.append(f"- oracle visual gain D-B = {oracle_gain:+.1f}, retrieved visual gain E-B = {retr_gain:+.1f}, "
                  f"retrieval loss D-E = {g.get(('D', 'E'), float('nan')):+.1f}")
        if ("C", "A") in g:
            md.append(f"- visual-only gain C-A = {g[('C', 'A')]:+.1f}; if this is large while D-B is small, "
                      f"consider text dominance / modality fusion")
        md.append("- large / large: both fine · large / small: retrieval bottleneck · small / small: utilization bottleneck")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    save_json(js, os.path.join(args.out_dir, "summary.json"))
    print("\n".join(md))
    print(f"\nwrote {args.out_dir}/summary.md and summary.json")


if __name__ == "__main__":
    main()
