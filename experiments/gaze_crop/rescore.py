#!/usr/bin/env python3
"""
Re-score a finished run's stored responses with both parsers. No GPU, no re-run: answer parsing is
post-hoc, so "did this arm win because of the parser?" is a one-second question.

The two scorers live in ../visual_bottleneck/eval_egolife.py: the original `evaluate_prediction`
(exact choice text, or a leading letter) and `evaluate_prediction_lenient` (falls back to the last
explicitly chosen / bolded letter, else the single choice quoted verbatim). They are lifted out by
source range so this script does not import torch.

    python experiments/gaze_crop/rescore.py experiments/gaze_crop/results_qa/*/E_prime.json
    python experiments/gaze_crop/rescore.py --show 10 <file>   # list the questions that changed

Measured on the visual-bottleneck runs: E' 45.8% under both parsers (0 of 120 questions change), B
40.8% -> 41.7% (1 question). Note that eval_egolife.py's --robust-reasoning also loosens the
round-decision JSON parsing, which changes the retrieval loop and therefore the generated answers --
that is a different run, not a different scoring of the same run. E' does not use the loop, so for
E' the flag is scoring-only, and this script measures all of its effect.
"""

import argparse
import ast
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple  # noqa: F401  (the lifted annotations need these)

VB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visual_bottleneck"))
EVAL_SRC = os.path.join(VB, "eval_egolife.py")
WANTED = ("normalize", "extract_choice_letter", "evaluate_prediction",
          "extract_choice_letter_lenient", "evaluate_prediction_lenient")


def load_scorers() -> Dict[str, Any]:
    with open(EVAL_SRC, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    ns: Dict[str, Any] = {"re": re, "Any": Any, "Dict": Dict, "List": List,
                          "Optional": Optional, "Tuple": Tuple}
    found = set()
    for node in tree.body:
        # module-level UPPER_CASE constants the lifted functions close over
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]), EVAL_SRC, "exec"), ns)
            except Exception:
                pass
        if isinstance(node, ast.FunctionDef) and node.name in WANTED:
            exec(compile(ast.Module(body=[node], type_ignores=[]), EVAL_SRC, "exec"), ns)
            found.add(node.name)
    missing = set(WANTED) - found
    if missing:
        raise SystemExit(f"could not lift {sorted(missing)} from {EVAL_SRC}; has it been refactored?")
    return ns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="results JSON files written by eval_egolife.py")
    ap.add_argument("--show", type=int, default=3, help="how many changed questions to print")
    args = ap.parse_args()
    ns = load_scorers()

    for path in args.results:
        rows = json.load(open(path, encoding="utf-8"))
        strict = lenient = 0
        changed: List[Tuple[str, str, str, str]] = []
        for r in rows:
            s = ns["evaluate_prediction"](r["response"], r["answer"], r["choices"])
            ok, letter = ns["evaluate_prediction_lenient"](r["response"], r["answer"], r["choices"])
            strict += int(s)
            lenient += int(ok)
            if ok != s:
                changed.append((str(r["ID"]), "gained" if ok else "LOST", str(letter), r["answer"]))
        n = len(rows) or 1
        # stored 'evaluate' is whatever parser produced the file; flag a mismatch rather than hide it
        stored = sum(int(r.get("evaluate", 0)) for r in rows)
        note = "" if stored in (strict, lenient) else f"  [!] stored evaluate={stored}, matches neither"
        print(f"{path}\n  n={n}  strict {strict}/{n} = {100*strict/n:.1f}%   "
              f"lenient {lenient}/{n} = {100*lenient/n:.1f}%   "
              f"({100*(lenient-strict)/n:+.1f}pp, {len(changed)} changed){note}")
        for c in changed[:args.show]:
            print(f"    ID {c[0]:>4}: {c[1]} (lenient letter {c[2]}, gold {c[3]})")


if __name__ == "__main__":
    main()
