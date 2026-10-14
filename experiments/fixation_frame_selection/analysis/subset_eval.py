#!/usr/bin/env python3
"""recall@3 on a-priori "gaze-suited" subsets of EgoLifeQA (criteria fixed BEFORE looking at results).

  V      need_audio == False                                  (answerable from pixels)
  F      wearer is inside a fixation at the annotated target moment (embed_fix_arms detector, defaults)
  subsets: V, V&F, and their complements. The first annotated moment is used for F.

F is a property of the gaze at the target moment, not of what any arm retrieved. It is still correlated
with what fixsel keeps, so a gain on V&F says "where attention and evidence coincide", not "fixsel works".

    python analysis/subset_eval.py     # -> analysis/tables/question_query/subset_eval.md
"""
import contextlib, os, sys
from unittest.mock import MagicMock
try:
    import matplotlib.pyplot  # noqa: F401
except ImportError:
    for m in ("matplotlib", "matplotlib.pyplot", "matplotlib.patches", "matplotlib.cm", "matplotlib.colors"):
        sys.modules[m] = MagicMock()
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
GC = os.path.abspath(os.path.join(EXP, "..", "gaze_crop"))
for p in (EXP, GC, os.path.join(EXP, "streamgaze")):
    sys.path.insert(0, p)
from gaze_common import GAZE_PATH, QUESTIONS_PATH, exact_paired_p, load_gaze, load_json  # noqa: E402
from embed_fix_arms import FIX_DEFAULTS, fixation_of  # noqa: E402
from fixation_sweep import clip_fixations  # noqa: E402
spec = importlib.util.spec_from_file_location("gcc", os.path.join(GC, "tools", "build_gaze_cases.py"))
gcc = importlib.util.module_from_spec(spec); spec.loader.exec_module(gcc)

ARMS = ["full", "fixsel_ctl", "fixsel", "center_05", "gazef_05", "fixsel_gazef_05"]
LAB = {"full": "full", "fixsel_ctl": "fixsel_ctl", "fixsel": "fixsel", "center_05": "center", "gazef_05": "gazef", "fixsel_gazef_05": "fixsel_gazef"}
PAIRS = [("gazef_05", "full"), ("gazef_05", "center_05"), ("fixsel", "fixsel_ctl"), ("fixsel", "full"),
         ("fixsel_gazef_05", "gazef_05"), ("fixsel_gazef_05", "full")]
qs = {e["ID"]: e for e in load_json(QUESTIONS_PATH)["entries"]}
clips = load_json(gcc.POOL_ALL)["clips"]
gaze = load_gaze(GAZE_PATH)
feat = {}
for qid, q in qs.items():
    tc = gcc.target_clip(q, clips)
    if not tc or tc[0]["key"] not in gaze:
        continue
    clip, off, _ = tc
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        fx = clip_fixations(gaze[clip["key"]].get("samples") or [], radius_thresh=FIX_DEFAULTS["radius"],
                            duration_thresh=FIX_DEFAULTS["min_dur"], gap_thresh=FIX_DEFAULTS["gap"],
                            dropout_thresh=FIX_DEFAULTS["dropout"])
    feat[qid] = {"V": not q.get("need_audio"), "F": fixation_of(sorted(fx, key=lambda f: f["start_time"]), off) is not None}

SETS = [("전체 (위치 파악된 문항)", lambda f: True),
        ("V", lambda f: f["V"]), ("not V (오디오 필요)", lambda f: not f["V"]),
        ("V∧F", lambda f: f["V"] and f["F"]), ("V∧not F", lambda f: f["V"] and not f["F"])]
L = ["# 사전 정의 부분집합별 recall@3 (EgoLifeQA 500)", "",
     "기준은 결과를 보기 전에 고정: V=오디오 불필요, F=정답 순간 착용자가 fixation 중. "
     "F는 첫 번째 주석 시점 기준이고 정답 클립을 못 찾은 문항은 제외했다. "
     "F는 정답 순간의 시선 속성이지 arm이 찾은 것이 아니지만 fixsel이 남기는 프레임과 상관이 있어, "
     "V∧F의 이득은 '주의와 근거가 겹치는 곳'의 이득으로 읽어야 한다.", "",
     "V∧F에서 `fixsel_gazef − gazef`가 가장 큰 것은 gaze crop에 fixation 프레임 선택을 추가한 차이다. "
     "gaze 위치 자체의 효과는 `gazef − center`로 비교하며, V∧F에서는 이 차이가 양수가 아니다.", ""]
for tol in ("0", "30", "60"):
    rs = {r["ID"]: r for r in load_json(os.path.join(EXP, "results", f"recall_pool_all_subset_r005_tol{tol}.json"))["per_question"]}
    L += [f"## tol {tol}s", "", "| 부분집합 | n | " + " | ".join(LAB[a] for a in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    detail = []
    for name, fn in SETS:
        ids = [i for i in rs if i in feat and fn(feat[i])]
        if not ids:
            continue
        L.append(f"| {name} | {len(ids)} | " + " | ".join(f"{100 * sum(rs[i]['hit@3'].get(a, 0) for i in ids) / len(ids):.1f}" for a in ARMS) + " |")
        cells = []
        for a, b in PAIRS:
            w = sum(1 for i in ids if rs[i]["hit@3"].get(a) and not rs[i]["hit@3"].get(b))
            l = sum(1 for i in ids if rs[i]["hit@3"].get(b) and not rs[i]["hit@3"].get(a))
            cells.append(f"{100 * (w - l) / len(ids):+.1f} (+{w}/−{l}, p={exact_paired_p(w, l):.2f})")
        detail.append(f"| {name} | " + " | ".join(cells) + " |")
    L += ["", "짝지은 차이 (%p, +앞만/−뒤만, 부호검정 p)", "", "| 부분집합 | " + " | ".join(f"{LAB[a]} − {LAB[b]}" for a, b in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)] + detail + [""]
out = os.path.join(HERE, "tables", "question_query", "subset_eval.md")
open(out, "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
