#!/usr/bin/env python3
"""
Did a gaze arm beat `full` in stage 1? Prints the comparison and exits 0 if yes, 1 if no, so a
shell script can branch on it without parsing numbers out of a log.

    python experiments/gaze_crop/gate.py --arms gazef@0.5
    python experiments/gaze_crop/gate.py --min-gain-pp 2.0 --quiet && echo "worth stage 2"

The winning arm's name is printed as  BEST=<arm>  so run_stage2.sh can carry that one forward.
Exit code 2 means the question could not be answered (missing file, missing arm).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import RESULTS_DIR, load_json  # noqa: E402


def canon(n: str) -> str:
    return n.replace("@", "_").replace(".", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default=os.path.join(RESULTS_DIR, "recall_stage1_question_tol0.json"))
    ap.add_argument("--arms", nargs="+", default=["gazef@0.5"],
                    help="candidate gaze arms; the best one is reported")
    ap.add_argument("--baseline", default="full")
    ap.add_argument("--min-gain-pp", type=float, default=0.0,
                    help="required recall gain over the baseline, in percentage points")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        print(f"gate: {args.results} not found -- run stage 1 first")
        return 2
    d = load_json(args.results)
    arms, k = d["arms"], str(d["main_k"])
    if args.baseline not in arms:
        print(f"gate: no '{args.baseline}' arm in {args.results}")
        return 2
    base = 100 * arms[args.baseline]["recall"][k]

    rows = []
    for want in args.arms:
        key = next((n for n in arms if canon(n) == canon(want)), None)
        if key is None:
            continue
        rows.append((want, 100 * arms[key]["recall"][k] - base))
    if not rows:
        print(f"gate: none of {args.arms} are in {args.results} (have {sorted(arms)})")
        return 2

    rows.sort(key=lambda r: -r[1])
    best, gain = rows[0]
    if not args.quiet:
        print(f"recall@{k}: {args.baseline} {base:.1f}%")
        for name, g in rows:
            print(f"  {name}: {base + g:.1f}%  ({g:+.1f}pp vs {args.baseline})")
        # the pair test from recall_eval, when it is there, says whether the gap is noise
        for p in d.get("pairs", []):
            if canon(p["a"]) == canon(best) and canon(p["b"]) == args.baseline:
                if "mcnemar_p" in p:
                    n_d = p.get("n_discordant", p["a_only"] + p["b_only"])
                    print(f"  paired test for {best} - {args.baseline}: "
                          f"{p['b_only']} vs {p['a_only']} of {n_d} discordant, "
                          f"p={p['mcnemar_p']:.3f}"
                          + ("" if p["mcnemar_p"] < 0.05 else "  <- not significant"))
                elif "ci95_pp" in p:      # results file from before the bootstrap was dropped
                    print(f"  paired 95% CI for {best} - {args.baseline}: "
                          f"[{p['ci95_pp'][0]:+.1f}, {p['ci95_pp'][1]:+.1f}]  (old bootstrap CI; "
                          f"unreliable at small discordant counts)")
    print(f"BEST={best}")
    ok = gain > args.min_gain_pp
    if not args.quiet:
        print(f"gate: {'PASS' if ok else 'FAIL'} "
              f"(best gain {gain:+.1f}pp vs threshold {args.min_gain_pp:+.1f}pp)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
