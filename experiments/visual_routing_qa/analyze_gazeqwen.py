"""Aggregate the GazeQwen oracle runs (oracle_gazeqwen.py) into tables and a per-question CSV. Standard library only.

    python analyze_gazeqwen.py                      # reads run_data/results/oracle_gq_*_{sdpa,eager}
    python analyze_gazeqwen.py --out run_data/analysis

Writes gazeqwen_summary.md and gazeqwen_per_question.csv. Every comparison is paired by question (bootstrap CI over
questions, exact sign test = McNemar). There is no multiple-comparison correction; the post hoc splits are descriptive.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDS = ("gq_q25_full", "gq_lora_only", "gq_inject_only", "gq_gaze", "gq_placebo")
TAGS = ("sdpa", "eager")
LADDER = (("gq_lora_only", "gq_q25_full", "LoRA"),
          ("gq_inject_only", "gq_lora_only", "V-JEPA injection"),
          ("gq_gaze", "gq_inject_only", "gaze content"),
          ("gq_gaze", "gq_placebo", "gaze position"),
          ("gq_gaze", "gq_lora_only", "GazeQwen vs LoRA only"))
BOOT = 4000


def load(results: Path, tag: str) -> dict:
    out = {}
    for cond in CONDS:
        out[cond] = {}
        for f in (results / f"oracle_{cond}_{tag}").glob("*.json"):
            d = json.loads(f.read_text(encoding="utf-8"))
            out[cond][str(d["ID"])] = d
    return out


def exact_p(up: int, down: int) -> float:
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    return min(1.0, 2 * sum(math.comb(n, x) for x in range(k + 1)) / 2 ** n)


def paired(a: list[int], b: list[int], seed: int) -> dict:
    """a - b over the same questions (0/1 correctness): mean diff in %p, bootstrap CI, up/down, exact p."""
    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(BOOT))
    up, down = sum(v > 0 for v in d), sum(v < 0 for v in d)
    return {"diff": 100 * sum(d) / n, "lo": boots[int(0.025 * BOOT)], "hi": boots[int(0.975 * BOOT) - 1],
            "up": up, "down": down, "p": exact_p(up, down)}


def fmt(c: dict) -> str:
    return f"{c['diff']:+.1f} | [{c['lo']:+.1f}, {c['hi']:+.1f}] | {c['up']} / {c['down']} | {c['p']:.3f}"


def section(lines: list[str], title: str, R: dict, ids: list[str]) -> None:
    n = len(ids)
    lines += [f"### {title} (N={n})", "", "| 조건 | 정답 | 정확도 |", "|---|---:|---:|"]
    corr = {c: [int(R[c][i]["correct"]) for i in ids] for c in CONDS}
    for c in CONDS:
        lines.append(f"| `{c}` | {sum(corr[c])} | {100 * sum(corr[c]) / n:.1f}% |")
    lines += ["", "| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |", "|---|---|---:|---|---:|---:|"]
    for k, (x, y, what) in enumerate(LADDER):
        lines.append(f"| `{x} − {y}` | {what} | {fmt(paired(corr[x], corr[y], k))} |")
    changed = lambda x, y: sum(R[x][i]["response"] != R[y][i]["response"] for i in ids)  # noqa: E731
    lines += ["", f"답이 바뀐 문항: gaze vs inject_only {changed('gq_gaze', 'gq_inject_only')}, "
                  f"gaze vs placebo {changed('gq_gaze', 'gq_placebo')}, gaze vs lora_only {changed('gq_gaze', 'gq_lora_only')}", ""]


def impl_table(lines: list[str], R: dict, ids: list[str]) -> None:
    n = len(ids)
    lines += [f"N={n}", "", "| 조건 | sdpa | eager | Δ %p | 95% CI | 개선 / 악화 | p | 답 일치 |", "|---|---:|---:|---:|---|---:|---:|---:|"]
    for k, c in enumerate(CONDS):
        a = [int(R["sdpa"][c][i]["correct"]) for i in ids]
        b = [int(R["eager"][c][i]["correct"]) for i in ids]
        same = sum(R["sdpa"][c][i]["response"] == R["eager"][c][i]["response"] for i in ids)
        lines.append(f"| `{c}` | {100 * sum(a) / n:.1f}% | {100 * sum(b) / n:.1f}% | {fmt(paired(b, a, 100 + k))} | {100 * same / n:.1f}% |")
    lines.append("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=HERE / "run_data/results")
    ap.add_argument("--out", type=Path, default=HERE / "run_data/analysis")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    R = {t: load(args.results, t) for t in TAGS}
    ok = {t: {i for i in R[t][CONDS[0]] if all(i in R[t][c] and R[t][c][i]["status"] == "ok" for c in CONDS)} for t in TAGS}
    failed = {t: sorted({i for c in CONDS for i, d in R[t][c].items() if d["status"] != "ok"}, key=int) for t in TAGS}
    order = lambda s: sorted(s, key=int)  # noqa: E731
    nfix = {i: R["sdpa"]["gq_gaze"][i].get("n_fixations", 0) for i in ok["sdpa"]}
    lines = ["# GazeQwen oracle 실험 요약", "",
             "생성: `analyze_gazeqwen.py`. 문항별 결과 JSON(`run_data/results/oracle_gq_*_{sdpa,eager}`)에서 계산. "
             "짝지은 부트스트랩(4,000회), 정확 부호검정. 다중비교 보정 없음.", "",
             f"- sdpa 완료 {len(ok['sdpa'])}문항 (실패: {failed['sdpa'] or '없음'})",
             f"- eager 완료 {len(ok['eager'])}문항 (실패: {failed['eager'] or '없음'})", ""]

    lines += ["## 1. sdpa", ""]
    section(lines, "전체", R["sdpa"], order(ok["sdpa"]))
    with_fix = order(i for i in ok["sdpa"] if nfix[i] > 0)
    lines += [f"fixation이 있는 문항 {len(with_fix)}개, 없는 문항 {len(ok['sdpa']) - len(with_fix)}개 "
              "(없는 문항에서는 `gq_gaze`가 구조적으로 `gq_lora_only`와 같다).", ""]
    section(lines, "fixation이 1개 이상인 문항", R["sdpa"], with_fix)

    lines += ["## 2. eager", ""]
    section(lines, "전체 (성공한 문항)", R["eager"], order(ok["eager"]))
    section(lines, "fixation이 1개 이상인 문항", R["eager"], order(i for i in ok["eager"] if nfix.get(i, 0) > 0))

    both = order(ok["sdpa"] & ok["eager"])
    n = len(both)
    lines += ["## 3. 구현 차이 (eager − sdpa, 같은 문항)", ""]
    impl_table(lines, R, both)
    lines += ["## 4. 같은 문항 부분집합에서의 구현별 비교", ""]
    section(lines, "sdpa, 양쪽 모두 성공한 문항", R["sdpa"], both)
    section(lines, "eager, 양쪽 모두 성공한 문항", R["eager"], both)

    visual = lambda ids: [i for i in ids if not R["sdpa"]["gq_gaze"][i]["need_audio"]]  # noqa: E731
    v_sdpa, v_eager, v_both = visual(order(ok["sdpa"])), visual(order(ok["eager"])), visual(both)
    lines += ["## 5. 시각 근거 문항 (`need_audio=False`)", "",
              "정답 근거에 대화가 필요 없다고 라벨된 문항이다(ORACLE_REPORT의 분석 2와 같은 기준). "
              "\"화면 하나로 풀린다\"는 뜻은 아니다. 사후 분할이며 보정하지 않았다.", ""]
    lines += ["### 5.1 sdpa", ""]
    section(lines, "시각 근거 전체", R["sdpa"], v_sdpa)
    section(lines, "시각 근거 + fixation 1개 이상", R["sdpa"], [i for i in v_sdpa if nfix[i] > 0])
    lines += ["### 5.2 eager", ""]
    section(lines, "시각 근거 전체 (성공한 문항)", R["eager"], v_eager)
    section(lines, "시각 근거 + fixation 1개 이상", R["eager"], [i for i in v_eager if nfix.get(i, 0) > 0])
    lines += ["### 5.3 구현 차이 (시각 근거, eager − sdpa)", ""]
    impl_table(lines, R, v_both)

    (args.out / "gazeqwen_summary.md").write_text("\n".join(lines), encoding="utf-8")

    cols = ["ID", "type", "need_audio", "answer", "n_fixations"]
    for t in TAGS:
        cols += [f"n_tokens_{t}"] + [f"{c}_{t}_{k}" for c in CONDS for k in ("response", "correct")]
    with (args.out / "gazeqwen_per_question.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in order(ok["sdpa"] | ok["eager"]):
            base = (R["sdpa"]["gq_gaze"].get(i) or R["eager"]["gq_gaze"][i])
            row = [i, base.get("type"), base.get("need_audio"), base["answer"], nfix.get(i, "")]
            for t in TAGS:
                g = R[t]["gq_gaze"].get(i)
                row.append(g.get("n_tokens", "") if g else "")
                for c in CONDS:
                    d = R[t][c].get(i)
                    good = d is not None and d["status"] == "ok"
                    row += [d["response"] if good else "", int(d["correct"]) if good else ""]
            w.writerow(row)
    print(f"wrote {args.out / 'gazeqwen_summary.md'} and gazeqwen_per_question.csv")


if __name__ == "__main__":
    main()
