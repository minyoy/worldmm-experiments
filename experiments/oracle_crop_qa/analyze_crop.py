"""Summarize the oracle-frame crop experiment (oracle_crop.py). Analyses fixed in PLAN.md before any result was seen:
  (1) all questions with a target clip (primary)   (2) need_audio=False   (3) >=1 off-center timestamp (offcenter only)
Primary comparisons (Holm over three): gaze - center, dual_gaze - dual_center, dual_offcenter - full.

Reads  run_data/results/<cond>/, run_data/frames/, ../visual_routing_qa/run_data/results/oracle_full/ (reproducibility)
Writes run_data/analysis/summary.md, per_question.csv, excluded.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDS = ("full", "gaze", "center", "dual_center", "dual_gaze", "dual_offcenter", "dual_offcenter_center")
PRIMARY = (("gaze", "center"), ("dual_gaze", "dual_center"), ("dual_offcenter", "full"))
SECONDARY = (("gaze", "full"), ("center", "full"), ("dual_center", "full"), ("dual_gaze", "full"),
             ("dual_offcenter_center", "full"), ("dual_gaze", "gaze"), ("dual_center", "center"),
             ("dual_offcenter", "dual_gaze"), ("dual_offcenter", "dual_offcenter_center"))
OFFCENTER = (("dual_offcenter", "full"), ("dual_offcenter_center", "full"), ("dual_offcenter", "dual_offcenter_center"),
             ("dual_offcenter", "dual_gaze"))


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def boot_ci(deltas: list[int], seed: int = 20261012, reps: int = 10000) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(100 * sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps) - 1]


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def pair_rows(records: list[dict], pairs, adjust: bool) -> list[str]:
    rows, ps = [], []
    for a, b in pairs:
        d = [r[a] - r[b] for r in records]
        lo, hi = boot_ci(d)
        p = mcnemar(d.count(1), d.count(-1))
        ps.append(p)
        rows.append([f"{a} − {b}", f"{100 * sum(d) / len(d):+.1f}", f"[{lo:+.1f}, {hi:+.1f}]",
                     f"{d.count(1)} / {d.count(-1)}", f"{p:.3f}"])
    head = "| Pair | Δ pp | 95% CI | A only / B only | exact p |"
    sep = "|---|---:|---|---:|---:|"
    if adjust:
        head, sep = head + " Holm p |", sep + "---:|"
        for row, q in zip(rows, holm(ps)):
            row.append(f"{q:.3f}")
    return [head, sep] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def acc_table(records: list[dict]) -> list[str]:
    n = len(records)
    lines = ["| Condition | Correct | Rate | Mean images |", "|---|---:|---:|---:|"]
    for c in CONDS:
        k = sum(r[c] for r in records)
        lines.append(f"| {c} | {k} | {100 * k / n:.1f}% | {sum(r['img_' + c] for r in records) / n:.1f} |")
    return lines + [""]


def section(title: str, records: list[dict], primary: bool) -> list[str]:
    lines = [f"## {title}  (N={len(records)})", ""] + acc_table(records)
    if primary:
        lines += ["**Primary (Holm over these three)**", ""] + pair_rows(records, PRIMARY, True)
    lines += ["**Secondary (descriptive, not corrected)**", ""] + pair_rows(records, SECONDARY, False)
    return lines


def type_section(records: list[dict]) -> list[str]:
    types = sorted({r["type"] for r in records}, key=lambda t: (-sum(r["type"] == t for r in records), t))
    lines = ["## By question type (descriptive, post-hoc, not tested)", "",
             "| Type | N | " + " | ".join(CONDS) + " |", "|---|---:|" + "---:|" * len(CONDS)]
    for t in types:
        s = [r for r in records if r["type"] == t]
        lines.append(f"| {t} | {len(s)} | " + " | ".join(f"{100 * sum(r[c] for r in s) / len(s):.1f}%" for c in CONDS) + " |")
    return lines + [""]


def repro_section(records: list[dict], got: dict, old_dir: Path) -> list[str]:
    """New `full` vs the gaze-prompting run's `oracle_full`: identical inputs on single-clip questions."""
    same_in, same_resp, same_corr, n = 0, 0, 0, 0
    for r in records:
        p = old_dir / f"{r['ID']}.json"
        if r["n_target_clips"] != 1 or not p.exists():
            continue
        old, new = read_json(p), got[r["ID"]]["full"]
        n += 1
        same_in += old.get("frame_ids") == [x.split(":", 1)[1] for x in new["frame_ids"]]
        same_resp += old.get("response") == new.get("response")
        same_corr += bool(old.get("correct")) == bool(new.get("correct"))
    if not n:
        return []
    return ["## Reproducibility: new `full` vs previous `oracle_full` (single target clip)", "",
            f"Questions: {n}. Same frame ids: {same_in}. Same response: {same_resp}. Same correctness: {same_corr}.",
            "Different responses on identical frames measure run-to-run noise (image order, batching, kernels).", ""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "run_data")
    ap.add_argument("--old", type=Path, default=HERE.parent / "visual_routing_qa/run_data/results/oracle_full")
    args = ap.parse_args()
    manifest = read_json(args.out / "crop_manifest.json")
    records, skipped, got = [], {}, {}
    for qid in map(str, manifest["question_ids"]):
        g = {}
        for c in CONDS:
            p = args.out / "results" / c / f"{qid}.json"
            if not p.exists():
                skipped[qid] = f"missing {c}"
                break
            d = read_json(p)
            if d.get("status") != "ok":
                skipped[qid] = f"{c}: {d.get('error', 'no status')}"
                break
            g[c] = d
        if qid in skipped:
            continue
        if len({d["text_hash"] for d in g.values()}) != 1:
            raise ValueError(f"{qid}: text contexts differ across conditions")
        got[qid] = g
        f = g["full"]
        r = {"ID": qid, "need_audio": f["need_audio"], "type": f["type"], "n_target_clips": len(f["target_keys"]),
             "n_timestamps": f["n_timestamps"], "n_offcenter": f["n_offcenter"]}
        for c in CONDS:
            r[c] = int(bool(g[c]["correct"]))
            r["img_" + c] = g[c]["num_images"]
        records.append(r)
    if not records:
        raise ValueError("No question has all conditions finished")
    visual = [r for r in records if not r["need_audio"]]
    off = [r for r in records if r["n_offcenter"] >= 1]
    off5 = [r for r in records if r["n_offcenter"] >= 5]
    lines = ["# Oracle-frame crop QA", "",
             f"Questions with all {len(CONDS)} conditions finished: {len(records)}/{len(manifest['question_ids'])}; "
             f"excluded: {len(skipped)}.", "",
             f"Target clips (tol=0), 1 fps, <= {manifest['max_timestamps']} timestamps; crop ratio {manifest['crop_ratio']}; "
             f"off-center = gaze distance from (0.5, 0.5) >= {manifest['offcenter_dist']:.3f}. Same timestamps, text "
             "memory, QA model and greedy decoding in every condition. Analyses 1-3 were fixed before the results.", ""]
    lines += section("Analysis 1 (primary): all questions", records, True)
    lines += section("Analysis 2: need_audio=False", visual, False)
    lines += [f"## Analysis 3: >=1 off-center timestamp  (N={len(off)})", ""] + acc_table(off) + pair_rows(off, OFFCENTER, False)
    if off5:
        lines += [f"## >=5 off-center timestamps (descriptive)  (N={len(off5)})", ""] + acc_table(off5) + pair_rows(off5, OFFCENTER, False)
    lines += type_section(records)
    lines += repro_section(records, got, args.old)
    analysis = args.out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ["ID", "need_audio", "type", "n_target_clips", "n_timestamps", "n_offcenter", *CONDS,
              *("img_" + c for c in CONDS)]
    with (analysis / "per_question.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    (analysis / "excluded.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    print(analysis / "summary.md")


if __name__ == "__main__":
    main()
