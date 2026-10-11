"""Summarize paired QA accuracy and visual Hit@3 for full, gaze, routing, oracle."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

ARMS = ("full", "gaze", "routing", "oracle")
PAIRS = (("gaze", "full"), ("routing", "full"), ("routing", "gaze"),
         ("oracle", "routing"))


def read_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def boot_ci(deltas: list[int], seed: int = 20261002) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    if not n:
        return float("nan"), float("nan")
    means = sorted(100 * sum(deltas[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(5000))
    return means[124], means[4874]


def metric_table(records: dict, ids: list[str], metric: str) -> list[str]:
    lines = [f"### {metric}", "", "| Condition | Successes | N | Rate |", "|---|---:|---:|---:|"]
    for arm in ARMS:
        successes = sum(bool(records[qid][arm][metric]) for qid in ids)
        lines.append(f"| {arm} | {successes} | {len(ids)} | {100*successes/len(ids):.1f}% |")
    lines.extend(["", "| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |",
                  "|---|---:|---:|---:|---:|"])
    for a, b in PAIRS:
        delta = [int(bool(records[qid][a][metric])) - int(bool(records[qid][b][metric]))
                 for qid in ids]
        lo, hi = boot_ci(delta)
        lines.append(f"| {a} − {b} | {100*sum(delta)/len(delta):+.1f} | "
                     f"{delta.count(1)} | {delta.count(-1)} | [{lo:+.1f}, {hi:+.1f}] |")
    lines.append("")
    return lines


DUAL = ("dual_gaze", "dual_retrieve")
DUAL_PAIRS = (("dual_gaze", "gaze"), ("dual_gaze", "full"), ("dual_retrieve", "full"),
              ("dual_retrieve", "gaze"), ("dual_retrieve", "dual_gaze"), ("dual_retrieve", "routing"))


def dual_table(records: dict, ids: list[str]) -> list[str]:
    """QA accuracy of the two dual-view conditions next to the single-view ones (retrieval is not repeated)."""
    arms = ("full", "gaze", "routing", "oracle") + DUAL
    lines = ["### correct (dual-view conditions)", "",
             "D `dual_gaze`: gaze Top-3 clips shown as full frame + gaze crop. "
             "E `dual_retrieve`: full Top-3 as full frames + gaze Top-3 as gaze crops. "
             "Every condition gets the same text memory and at most 64 images.", "",
             "| Condition | Successes | N | Rate |", "|---|---:|---:|---:|"]
    for arm in arms:
        k = sum(bool(records[q][arm]["correct"]) for q in ids)
        lines.append(f"| {arm} | {k} | {len(ids)} | {100 * k / len(ids):.1f}% |")
    lines += ["", "| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |",
              "|---|---:|---:|---:|---:|"]
    for a, b in DUAL_PAIRS:
        delta = [int(bool(records[q][a]["correct"])) - int(bool(records[q][b]["correct"])) for q in ids]
        lo, hi = boot_ci(delta)
        lines.append(f"| {a} − {b} | {100 * sum(delta) / len(delta):+.1f} | {delta.count(1)} | "
                     f"{delta.count(-1)} | [{lo:+.1f}, {hi:+.1f}] |")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "run_data")
    args = ap.parse_args()
    manifest = read_json(args.out / "manifest.json")
    all_ids = list(map(str, manifest["question_ids"]))
    records = {}
    skipped = {}
    for qid in all_ids:
        data = {}
        for arm in ARMS[:3]:
            path = args.out / "results" / arm / f"{qid}.json"
            if not path.exists():
                skipped[qid] = f"missing {arm}"
                break
            data[arm] = read_json(path)
            if data[arm].get("status") != "ok":
                skipped[qid] = f"{arm}: {data[arm].get('error', 'no status')}"
                break
        if qid in skipped:
            continue
        chosen = data["routing"].get("selected_arm")
        if chosen not in ("full", "gaze"):
            raise ValueError(f"{qid}: bad routing choice {chosen}")
        selected = data[chosen]
        for field in ("top3", "response", "correct", "image_hash", "text_hash", "hit_at_3"):
            if data["routing"].get(field) != selected.get(field):
                raise ValueError(f"{qid}: C != selected {chosen} for {field}")
        if data["full"]["text_hash"] != data["gaze"]["text_hash"]:
            raise ValueError(f"{qid}: A/B text contexts differ")
        if len(data["full"]["top3"]) != 3 or len(data["gaze"]["top3"]) != 3:
            raise ValueError(f"{qid}: retrieval was not Top-3")
        data["oracle"] = {
            "correct": bool(data["full"]["correct"] or data["gaze"]["correct"]),
            "hit_at_3": {tol: bool(data["full"]["hit_at_3"][tol] or
                                   data["gaze"]["hit_at_3"][tol]) for tol in ("0", "60")},
        }
        for arm in ARMS:
            data[arm]["hit0"] = data[arm]["hit_at_3"]["0"]
            data[arm]["hit60"] = data[arm]["hit_at_3"]["60"]
        records[qid] = data
    qa_ids = [qid for qid in all_ids if qid in records]
    dual_ids = []
    for qid in list(records):
        found = {c: (args.out / "results" / c / f"{qid}.json") for c in DUAL}
        if all(p.exists() for p in found.values()):
            got = {c: read_json(p) for c, p in found.items()}
            if all(g.get("status") == "ok" for g in got.values()):
                records[qid].update(got)
                dual_ids.append(qid)
    if not qa_ids:
        raise ValueError("No complete A/B/C questions to analyze")
    hit_ids = {tol: [qid for qid in qa_ids if records[qid]["full"]["target_count"][tol] > 0]
               for tol in ("0", "60")}
    lines = ["# Full / gaze / query routing / oracle results", "",
             f"QA common denominator: {len(qa_ids)}/{len(all_ids)}; excluded: {len(skipped)}.",
             f"Hit@3 denominators: tol=0 **{len(hit_ids['0'])}**, "
             f"tol=60s **{len(hit_ids['60'])}**. "
             "Hit requires at least one target clip in the common candidate pool.", "",
             "Oracle is computed separately for QA and Hit@3 as A OR B; it is an offline upper bound.", ""]
    lines += metric_table(records, qa_ids, "correct")
    for tol, metric in (("0", "hit0"), ("60", "hit60")):
        if hit_ids[tol]:
            lines += metric_table(records, hit_ids[tol], metric)
    if dual_ids:
        lines += dual_table(records, [q for q in qa_ids if q in set(dual_ids)])
        if len(dual_ids) < len(qa_ids):
            lines += [f"(dual-view tables use the {len(dual_ids)}/{len(qa_ids)} questions whose D and E both finished.)", ""]
    routes = {arm: sum(records[qid]["routing"]["selected_arm"] == arm for qid in qa_ids)
              for arm in ("full", "gaze")}
    lines += ["## Routing and complementarity", "",
              f"Qwen decisions: FULL {routes['full']}, GAZE {routes['gaze']}.",
              f"Router agreed across both set orders on {sum(bool(records[q]['routing'].get('router_agree')) for q in qa_ids)}/{len(qa_ids)} "
              "questions (disagreements fall back to FULL).", ""]
    for metric, ids in (("correct", qa_ids), ("hit0", hit_ids["0"]), ("hit60", hit_ids["60"])):
        disagree = sum(records[i]["full"][metric] != records[i]["gaze"][metric] for i in ids)
        useful = sum(records[i]["routing"][metric] and
                     records[i]["full"][metric] != records[i]["gaze"][metric] for i in ids)
        lines.append(f"- {metric}: A/B disagree on {disagree}/{len(ids)}; router picks the "
                     f"successful arm on {useful}/{disagree} discordant questions.")
    analysis = args.out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (analysis / "per_question.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "selected_arm", "qa_full", "qa_gaze",
                                                "qa_routing", "qa_oracle", "hit0_full", "hit0_gaze",
                                                "hit0_routing", "hit0_oracle", "hit60_full",
                                                "hit60_gaze", "hit60_routing", "hit60_oracle",
                                                "qa_dual_gaze", "qa_dual_retrieve"])
        writer.writeheader()
        for qid in qa_ids:
            r = records[qid]
            row = {"ID": qid, "selected_arm": r["routing"]["selected_arm"]}
            for arm in ARMS:
                for col, field in (("qa", "correct"), ("hit0", "hit0"), ("hit60", "hit60")):
                    row[f"{col}_{arm}"] = int(r[arm][field])
            for c in DUAL:
                row[f"qa_{c}"] = int(r[c]["correct"]) if c in r else ""
            writer.writerow(row)
    (analysis / "excluded.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    print(analysis / "summary.md")


if __name__ == "__main__":
    main()
