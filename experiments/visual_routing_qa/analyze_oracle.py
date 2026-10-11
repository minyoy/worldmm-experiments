"""Summarize the oracle-frame gaze-prompting experiment (oracle.py).

Two analyses, both fixed before any result was seen:
  (1) all questions with a target clip (primary)
  (2) need_audio=False questions only (visual-evidence subset)
Primary comparisons: oracle_{text,mark,saliency} - oracle_full, paired. The retrieval-based `full` / `gaze` results of
run.py are shown for reference (how much the oracle frames themselves matter).

Reads  run_data/results/oracle_*/, run_data/oracle_manifest.json, run_data/analysis/per_question.csv
Writes run_data/analysis/oracle_summary.md and oracle_per_question.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDS = ("oracle_full", "oracle_text", "oracle_mark", "oracle_saliency")
REFERENCE = ("full", "gaze")                    # retrieved-frame conditions from run.py (per_question.csv)
PRIMARY = (("oracle_text", "oracle_full"), ("oracle_mark", "oracle_full"), ("oracle_saliency", "oracle_full"))
SECONDARY = (("oracle_mark", "oracle_text"), ("oracle_saliency", "oracle_text"), ("oracle_saliency", "oracle_mark"))
VS_RETRIEVED = (("oracle_full", "full"), ("oracle_full", "gaze"))


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def boot_ci(deltas: list[int], seed: int = 20261002) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(100 * sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(5000))
    return means[124], means[4874]


def pair_table(records: list[dict], pairs, title: str) -> list[str]:
    lines = [f"**{title}**", "", "| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |",
             "|---|---:|---:|---:|---:|"]
    for a, b in pairs:
        d = [int(r[a]) - int(r[b]) for r in records]
        lo, hi = boot_ci(d)
        lines.append(f"| {a} − {b} | {100 * sum(d) / len(d):+.1f} | {d.count(1)} | {d.count(-1)} | [{lo:+.1f}, {hi:+.1f}] |")
    return lines + [""]


def section(name: str, records: list[dict], with_reference: bool) -> list[str]:
    n = len(records)
    conds = CONDS + (REFERENCE if with_reference else ())
    lines = [f"## {name}  (N={n})", "", "| Condition | Correct | Rate |", "|---|---:|---:|"]
    for c in conds:
        k = sum(int(r[c]) for r in records)
        lines.append(f"| {c} | {k} | {100 * k / n:.1f}% |")
    lines.append("")
    lines += pair_table(records, PRIMARY, "Primary (gaze prompting vs no gaze information, same oracle frames)")
    lines += pair_table(records, SECONDARY, "Between prompting methods")
    if with_reference:
        lines += pair_table(records, VS_RETRIEVED, "Oracle frames vs retrieved frames (run.py `full` / `gaze`)")
    return lines


def pooled_section(name: str, records: list[dict]) -> list[str]:
    """Post-hoc: mean improvement of the three prompting conditions over oracle_full, per question.

    The three conditions share the questions and the baseline, so their gains are correlated and are not three
    independent confirmations. Bootstrap CI over questions; sign-flip permutation test of mean = 0.
    """
    prompts = CONDS[1:]
    v = [sum(int(r[c]) - int(r["oracle_full"]) for c in prompts) / len(prompts) for r in records]
    lo, hi = boot_ci(v)
    rng = random.Random(1)
    obs = abs(sum(v))
    hits = sum(abs(sum(x if rng.random() < 0.5 else -x for x in v)) >= obs - 1e-12 for _ in range(20000))
    return [f"| {name} | {len(records)} | {100 * sum(v) / len(v):+.2f} | [{lo:+.2f}, {hi:+.2f}] | {hits / 20000:.3f} |"]


def corr(a: list[float], b: list[float]) -> float:
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def pooled_sections(records: list[dict], visual: list[dict]) -> list[str]:
    lines = ["## Pooled effect of the three prompting methods (post-hoc)", "",
             "Per-question mean of (oracle_text, oracle_mark, oracle_saliency) minus oracle_full. The three methods are "
             "not independent confirmations: they share the questions and the baseline.", "",
             "| Analysis | N | Mean gain (pp) | 95% CI | sign-flip p |", "|---|---:|---:|---|---:|"]
    lines += pooled_section("all questions with a target clip", records)
    lines += pooled_section("need_audio=False", visual)
    d = {c: [int(r[c]) - int(r["oracle_full"]) for r in records] for c in CONDS[1:]}
    lines += ["", "Correlation of per-question changes: "
              f"text-mark {corr(d['oracle_text'], d['oracle_mark']):.2f}, "
              f"text-saliency {corr(d['oracle_text'], d['oracle_saliency']):.2f}, "
              f"mark-saliency {corr(d['oracle_mark'], d['oracle_saliency']):.2f}.",
              f"Questions improved by at least one method: {sum(any(d[c][i] > 0 for c in d) for i in range(len(records)))}; "
              f"worsened by at least one: {sum(any(d[c][i] < 0 for c in d) for i in range(len(records)))} of {len(records)}.", ""]
    return lines


def rate(records: list[dict], c: str) -> str:
    return f"{100 * sum(int(r[c]) for r in records) / len(records):.1f}%"


def type_sections(records: list[dict]) -> list[str]:
    """Descriptive breakdown by question type. Made after the main results were seen: no tests, no multiplicity
    correction; with ~15 type x method intervals about one is expected to exclude 0 by chance."""
    types = sorted({r["type"] for r in records}, key=lambda t: (-sum(r["type"] == t for r in records), t))
    by = {t: [r for r in records if r["type"] == t] for t in types}
    lines = ["## By question type (descriptive; made after the results were seen, not tested)", "",
             "Per-type samples are small (61-122); intervals are wide and are not corrected for the number of "
             "type x method comparisons.", "",
             "### Composition", "",
             "| Type | N | need_audio=False | need_audio=True | >1 target clips |", "|---|---:|---:|---:|---:|"]
    for t in types:
        s = by[t]
        lines.append(f"| {t} | {len(s)} | {sum(not r['need_audio'] for r in s)} | {sum(bool(r['need_audio']) for r in s)} "
                     f"| {sum(r['n_target_clips'] > 1 for r in s)} |")
    lines += ["", "### Accuracy", "", "| Type | N | " + " | ".join(CONDS) + " |", "|---|---:|" + "---:|" * len(CONDS)]
    for t in types:
        lines.append(f"| {t} | {len(by[t])} | " + " | ".join(rate(by[t], c) for c in CONDS) + " |")
    lines += ["", "### Difference from oracle_full (percentage points, [95% CI], gain/loss questions)", "",
              "| Type | N | oracle_text | oracle_mark | oracle_saliency |", "|---|---:|---|---|---|"]
    for t in types:
        cells = []
        for c in CONDS[1:]:
            d = [int(r[c]) - int(r["oracle_full"]) for r in by[t]]
            lo, hi = boot_ci(d)
            cells.append(f"{100 * sum(d) / len(d):+.1f} [{lo:+.1f}, {hi:+.1f}] ({d.count(1)}/{d.count(-1)})")
        lines.append(f"| {t} | {len(by[t])} | " + " | ".join(cells) + " |")
    lines += ["", "### Type x evidence kind (accuracy)", "",
              "| Type | need_audio | N | " + " | ".join(CONDS) + " |", "|---|---|---:|" + "---:|" * len(CONDS)]
    for t in types:
        for na in (False, True):
            s = [r for r in by[t] if bool(r["need_audio"]) == na]
            if s:
                lines.append(f"| {t} | {na} | {len(s)} | " + " | ".join(rate(s, c) for c in CONDS) + " |")
    lines += ["", "### By number of target clips (accuracy, mean difference from oracle_full in percentage points)", "",
              "| Group | N | " + " | ".join(CONDS) + " | Δ text / mark / saliency |", "|---|---:|" + "---:|" * len(CONDS) + "---|"]
    groups = (("1 target clip", [r for r in records if r["n_target_clips"] == 1]),
              (">1 target clips", [r for r in records if r["n_target_clips"] > 1]),
              ("HabitInsight, 1 clip", [r for r in by.get("HabitInsight", []) if r["n_target_clips"] == 1]),
              ("HabitInsight, >1 clips", [r for r in by.get("HabitInsight", []) if r["n_target_clips"] > 1]))
    for name, s in groups:
        if s:
            deltas = " / ".join(f"{100 * sum(int(r[c]) - int(r['oracle_full']) for r in s) / len(s):+.1f}" for c in CONDS[1:])
            lines.append(f"| {name} | {len(s)} | " + " | ".join(rate(s, c) for c in CONDS) + f" | {deltas} |")
    return lines + [""]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "run_data")
    args = ap.parse_args()
    manifest = read_json(args.out / "oracle_manifest.json")
    order = list(map(str, manifest["question_ids"]))
    ref = {}
    ref_path = args.out / "analysis" / "per_question.csv"
    if ref_path.exists():
        with ref_path.open(encoding="utf-8") as f:
            ref = {r["ID"]: r for r in csv.DictReader(f)}
    records, skipped = [], {}
    for qid in order:
        got = {}
        for c in CONDS:
            p = args.out / "results" / c / f"{qid}.json"
            if not p.exists():
                skipped[qid] = f"missing {c}"
                break
            d = read_json(p)
            if d.get("status") != "ok":
                skipped[qid] = f"{c}: {d.get('error', 'no status')}"
                break
            got[c] = d
        if qid in skipped:
            continue
        if len({d["text_hash"] for d in got.values()}) != 1:
            raise ValueError(f"{qid}: text contexts differ across conditions")
        r = {"ID": qid, "need_audio": got["oracle_full"]["need_audio"], "type": got["oracle_full"]["type"],
             "n_target_clips": len(got["oracle_full"]["target_keys"]),
             "n_images_full": got["oracle_full"]["num_images"]}
        for c in CONDS:
            r[c] = int(bool(got[c]["correct"]))
        for c in REFERENCE:
            r[c] = int(ref[qid][f"qa_{c}"]) if qid in ref else None
        records.append(r)
    if not records:
        raise ValueError("No question has all four oracle conditions finished")
    visual = [r for r in records if not r["need_audio"]]
    lines = ["# Oracle-frame gaze prompting", "",
             f"Questions with all four conditions finished: {len(records)}/{len(order)}; excluded: {len(skipped)}.", "",
             "Frames: the question's target clip(s) at tol=0, 1 fps, at most 64 frames (proportional per clip). "
             "Same text memory, QA model and greedy decoding in every condition; only the way the gaze is conveyed "
             "differs. Both analyses below (all questions; need_audio=False only) were fixed before the results were seen, "
             "and the primary one is the first.", ""]
    lines += section("Analysis 1 (primary): all questions with a target clip", records, bool(ref))
    lines += section("Analysis 2: need_audio=False (visual-evidence) questions", visual, bool(ref))
    lines += pooled_sections(records, visual)
    lines += type_sections(records)
    analysis = args.out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "oracle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = ["ID", "need_audio", "type", "n_target_clips", "n_images_full", *CONDS, *REFERENCE]
    with (analysis / "oracle_per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in records)
    (analysis / "oracle_excluded.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    print(analysis / "oracle_summary.md")


if __name__ == "__main__":
    main()
