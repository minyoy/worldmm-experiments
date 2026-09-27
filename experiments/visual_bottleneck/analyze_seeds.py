#!/usr/bin/env python3
"""
Average the single-shot conditions over repeated sampled runs and test the comparisons on the averages.

Runs (a condition is used from a run only when that run covers all subset questions):
    run0   results/                 the original run; B.json stands in for B_replay (same final prompt)
    seed1  results_seeds/seed1/     run_seeds.sh
    seed2  results_seeds/seed2/

Per question, each condition gets a score in {0, 1/3, 2/3, 1} (mean correctness over its runs).
A comparison X - Y uses d_q = score_X(q) - score_Y(q):
    delta   mean of d_q  (= mean accuracy of X minus mean accuracy of Y)
    p       exact-in-spirit sign-flip permutation test on d_q, two-sided. Under "the frames change
            nothing", X and Y runs of the same question are exchangeable, so each d_q can flip sign.
    per-run deltas are shown too, to see how much a single run could have said.
Also reported: majority-vote (>=2 of 3) McNemar, and how often a condition's answer flips between runs.

Standard library only (runs on the login node).

    python experiments/visual_bottleneck/analyze_seeds.py
"""

import argparse
import json
import os
import random
import statistics
import sys
from math import comb
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ANALYSIS_DIR, EXP_DIR, QA_TYPES, RESULTS_DIR, load_json, load_subset  # noqa: E402

SEEDS_DIR = os.path.join(EXP_DIR, "results_seeds")
RUNS = [
    ("run0", RESULTS_DIR, {"B_replay": "B"}),
    ("seed1", os.path.join(SEEDS_DIR, "seed1"), {}),
    ("seed2", os.path.join(SEEDS_DIR, "seed2"), {}),
]
CONDITIONS = [
    ("A", "Question only"),
    ("B_replay", "Text memory (B context, answered once)"),
    ("C", "Oracle visual"),
    ("D", "Text + Oracle visual"),
    ("E_prime", "B context + Retrieved visual"),
]
COMPARISONS = [
    ("C", "A", "visual evidence alone helps?"),
    ("D", "B_replay", "oracle visual gain on top of text"),
    ("C", "B_replay", "which modality is more usable (diagnostic)"),
    ("E_prime", "B_replay", "retrieved visual gain, text fixed"),
    ("D", "E_prime", "oracle - retrieved, text fixed"),
]
N_PERM = 50_000


def load_run(run_dir: str, cond: str, alias: Dict[str, str], ids: List[str]) -> Optional[Dict[str, bool]]:
    path = os.path.join(run_dir, f"{alias.get(cond, cond)}.json")
    if not os.path.exists(path):
        return None
    got = {str(r["ID"]): bool(r["evaluate"]) for r in load_json(path)}
    return got if all(i in got for i in ids) else {"__partial__": len([i for i in ids if i in got])}


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def sign_flip_p(d: List[float], n_perm: int = N_PERM, seed: int = 0) -> float:
    d = [x for x in d if x != 0]
    if not d:
        return 1.0
    obs = abs(sum(d))
    rng = random.Random(seed)
    hits = sum(1 for _ in range(n_perm)
               if abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - 1e-12)
    return (hits + 1) / (n_perm + 1)


def fmt(x: float) -> str:
    return f"{x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-md", default=os.path.join(ANALYSIS_DIR, "seeds_summary.md"))
    ap.add_argument("--out-json", default=os.path.join(ANALYSIS_DIR, "seeds_summary.json"))
    args = ap.parse_args()

    subset = load_subset()
    meta = {e["ID"]: e for e in subset["entries"]}
    ids = [e["ID"] for e in subset["entries"]]

    runs: Dict[str, List[Dict[str, bool]]] = {}
    run_names: Dict[str, List[str]] = {}
    status: Dict[str, List[str]] = {}
    for cond, _ in CONDITIONS:
        runs[cond], run_names[cond], status[cond] = [], [], []
        for name, d, alias in RUNS:
            r = load_run(d, cond, alias, ids)
            if r is None:
                status[cond].append(f"{name}: missing")
            elif "__partial__" in r:
                status[cond].append(f"{name}: partial {r['__partial__']}/{len(ids)}")
            else:
                runs[cond].append(r)
                run_names[cond].append(name)

    score = {c: {q: statistics.mean(r[q] for r in runs[c]) for q in ids} for c in runs if runs[c]}

    def acc(rs: Dict[str, bool], qs: List[str]) -> float:
        return 100 * sum(rs[q] for q in qs) / len(qs)

    out: List[str] = ["# Visual bottleneck: repeated sampled runs", ""]
    out += [f"Questions: {len(ids)}. Generation is sampled (Qwen3-VL default: do_sample, temperature 0.7); "
            "run0 is the original run, seed1/seed2 are repeats from `run_seeds.sh`. "
            "B_replay's run0 is `results/B.json` (identical final-answer prompt).", ""]
    incomplete = {c: s for c, s in status.items() if s}
    if incomplete:
        out += ["**Not used (incomplete):** " + "; ".join(f"{c} ({', '.join(s)})" for c, s in incomplete.items()), ""]

    # ---- accuracy per run -----------------------------------------------------------------------
    out += ["## Accuracy (%)", "",
            "| condition | label | runs | per run | mean | sd | answers flip across runs |",
            "|---|---|---|---|---|---|---|"]
    summary: Dict[str, Dict] = {"conditions": {}, "comparisons": [], "subsets": {}}
    for cond, label in CONDITIONS:
        rs = runs[cond]
        if not rs:
            out.append(f"| {cond} | {label} | 0 | - | - | - | - |")
            continue
        per = [acc(r, ids) for r in rs]
        sd = statistics.stdev(per) if len(per) > 1 else float("nan")
        flips = sum(1 for q in ids if len({r[q] for r in rs}) > 1) if len(rs) > 1 else None
        out.append(f"| {cond} | {label} | {len(rs)} | {' / '.join(fmt(p) for p in per)} | **{fmt(statistics.mean(per))}** "
                   f"| {fmt(sd) if len(per) > 1 else '-'} | {f'{flips}/{len(ids)}' if flips is not None else '-'} |")
        summary["conditions"][cond] = {"runs": run_names[cond], "acc": per, "mean": statistics.mean(per),
                                       "sd": sd if len(per) > 1 else None, "flips": flips}

    # ---- paired comparisons ---------------------------------------------------------------------
    def compare(x: str, y: str, qs: List[str]) -> Optional[Dict]:
        if x not in score or y not in score:
            return None
        d = [score[x][q] - score[y][q] for q in qs]
        per_run = [acc(runs[x][i], qs) - acc(runs[y][i], qs) for i in range(min(len(runs[x]), len(runs[y])))]
        res = {"n": len(qs), "acc_x": 100 * statistics.mean(score[x][q] for q in qs),
               "acc_y": 100 * statistics.mean(score[y][q] for q in qs),
               "delta": 100 * statistics.mean(d), "per_run": per_run,
               "better": sum(1 for v in d if v > 0), "worse": sum(1 for v in d if v < 0),
               "p_perm": sign_flip_p(d)}
        if len(runs[x]) % 2 == 1 and len(runs[y]) % 2 == 1:
            mx = {q: score[x][q] > 0.5 for q in qs}
            my = {q: score[y][q] > 0.5 for q in qs}
            b = sum(1 for q in qs if mx[q] and not my[q])
            c = sum(1 for q in qs if not mx[q] and my[q])
            res.update(maj_b=b, maj_c=c, maj_p=mcnemar_p(b, c))
        return res

    out += ["", "## Paired comparisons (mean over runs)", "",
            "delta = mean acc(X) - mean acc(Y). better/worse = questions whose run-averaged score is higher/lower. "
            "p = two-sided sign-flip permutation test on per-question score differences "
            f"({N_PERM:,} permutations). Majority = correct in >=2 of 3 runs, exact McNemar.", "",
            "| X - Y | meaning | acc X | acc Y | **delta** | per-run deltas | better / worse | p | majority b / c | majority p |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    by_audio = {v: [q for q in ids if bool(meta[q]["need_audio"]) is v] for v in (False, True)}
    for x, y, meaning in COMPARISONS:
        r = compare(x, y, ids)
        if r is None:
            out.append(f"| {x} - {y} | {meaning} | | | not enough runs | | | | | |")
            continue
        maj = (f"{r['maj_b']} / {r['maj_c']}", f"{r['maj_p']:.3f}") if "maj_p" in r else ("-", "-")
        out.append(f"| {x} - {y} | {meaning} | {fmt(r['acc_x'])} | {fmt(r['acc_y'])} | **{r['delta']:+.1f}** "
                   f"| {' / '.join(f'{v:+.1f}' for v in r['per_run'])} | {r['better']} / {r['worse']} "
                   f"| {r['p_perm']:.3f} | {maj[0]} | {maj[1]} |")
        summary["comparisons"].append({"x": x, "y": y, **r})
        for v, qs in by_audio.items():
            s = compare(x, y, qs)
            out.append(f"| ↳ need_audio={v} | | {fmt(s['acc_x'])} | {fmt(s['acc_y'])} | {s['delta']:+.1f} "
                       f"| | {s['better']} / {s['worse']} | {s['p_perm']:.3f} | | |")
            summary["comparisons"].append({"x": x, "y": y, "subset": f"need_audio={v}", **s})

    # ---- subsets --------------------------------------------------------------------------------
    avail = [c for c, _ in CONDITIONS if c in score]
    groups = [("need_audio", [(str(v), qs) for v, qs in by_audio.items()]),
              ("type", [(t, [q for q in ids if meta[q]["type"] == t]) for t in QA_TYPES]),
              ("gap", [(g, [q for q in ids if meta[q]["gap_bin"] == g])
                       for g in sorted({meta[q]["gap_bin"] for q in ids}, key=str)])]
    for gname, rows in groups:
        out += ["", f"### Mean accuracy by {gname}", "",
                "| subset | n | " + " | ".join(avail) + " |", "|---|---|" + "---|" * len(avail)]
        for label, qs in rows:
            if not qs:
                continue
            vals = [100 * statistics.mean(score[c][q] for q in qs) for c in avail]
            out.append(f"| {label} | {len(qs)} | " + " | ".join(fmt(v) for v in vals) + " |")
            summary["subsets"].setdefault(gname, {})[label] = dict(zip(avail, vals), n=len(qs))

    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(text)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(text)
    print(f"written: {args.out_md}\n         {args.out_json}")


if __name__ == "__main__":
    main()
