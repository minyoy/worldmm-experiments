#!/usr/bin/env python3
"""
A small static website for checking, question by question, whether the wearer's gaze/fixations
line up with what the question asks -- for the two fixation-frame-selection arms. Same format as
gaze_crop/gaze_cases, with both directions side by side (the arm found the answer clip when its
baseline did not, and the reverse):

  A. fixsel  better   fixsel hit@3, full missed         (frame SELECTION alone, whole frame)
  B. fixsel  worse    full hit@3, fixsel missed
  C. fixsel_gazef better   fixsel_gazef hit@3, gazef missed   (selection on top of the gaze crop)
  D. fixsel_gazef worse    gazef hit@3, fixsel_gazef missed

The baseline is the arm that differs by one thing: fixsel only deletes frames from full's own 16
uniform instants, and fixsel_gazef only deletes frames from gazef's. Cases are ordered by the biggest
rank change (log ratio). Each case shows the 16 embedded instants: frames inside a fixation (kept,
numbered by fixation) and frames outside every fixation (dropped, dimmed), the gaze point, the
fixation centroid, and for C/D the gazef crop box -- next to the question, choices and the EgoLife
reason, so "is the gaze where the question points?" can be judged by eye. The frame at the annotated
target moment is shown too, with whether the wearer was fixating then.

Frame times, fixation detection, kept/dropped and crop boxes are embed_fix_arms.py's own
(select_frames "subset", FIX_DEFAULTS, gaze_at, crop_box), so the page shows what the arms embedded.

Needs decord + PIL + pandas, CPU only (e.g. the `eyewo` conda env). The detector's package imports
matplotlib for plotting only; a stub stands in if it is not installed.

    python experiments/fixation_frame_selection/build_gaze_cases.py                # -> gaze_cases/index.html
    python experiments/fixation_frame_selection/build_gaze_cases.py --tol 0
    scp -r server:.../experiments/fixation_frame_selection/gaze_cases .  &&  open gaze_cases/index.html

The page keeps a per-question verdict (related / not / unsure) and a note in the browser, and exports
them as JSON. Frames are EgoLife video, so the folder is git-ignored.
"""

import argparse
import contextlib
import importlib.util
import json
import math
import os
import sys
from typing import Any, Dict, List

try:
    import matplotlib.pyplot  # noqa: F401
except ImportError:  # the detector imports it for its plotting helpers, which this script never calls
    from unittest.mock import MagicMock
    for _m in ("matplotlib", "matplotlib.pyplot", "matplotlib.patches", "matplotlib.cm", "matplotlib.colors"):
        sys.modules[_m] = MagicMock()

HERE = os.path.dirname(os.path.abspath(__file__))
GAZE_CROP = os.path.abspath(os.path.join(HERE, "..", "gaze_crop"))
sys.path.insert(0, HERE)
sys.path.insert(0, GAZE_CROP)
sys.path.insert(0, os.path.join(HERE, "streamgaze"))

from gaze_common import (  # noqa: E402
    GAZE_PATH, NFRAMES, QUESTIONS_PATH, VIDEO_ROOT, crop_box, decord_threads, gaze_at, load_gaze,
    load_json, resolve_video_path,
)
from embed_arms import frame_indices_uniform  # noqa: E402
from embed_fix_arms import FIX_DEFAULTS, fixation_of, select_frames  # noqa: E402
from fixation_sweep import clip_fixations  # noqa: E402

_spec = importlib.util.spec_from_file_location("gaze_crop_cases", os.path.join(GAZE_CROP, "build_gaze_cases.py"))
_gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gc)
target_clip, POOL_ALL, RATIO = _gc.target_clip, _gc.POOL_ALL, _gc.RATIO

RESULTS = os.path.join(HERE, "results")
ARMS = ["full", "fixsel_ctl", "fixsel", "center_05", "gazef_05", "fixsel_gazef_05"]
# key -> (arm, baseline, direction); direction "up" = arm hit and baseline missed
SETS = {
    "A": ("fixsel", "full", "up"),
    "B": ("fixsel", "full", "down"),
    "C": ("fixsel_gazef_05", "gazef_05", "up"),
    "D": ("fixsel_gazef_05", "gazef_05", "down"),
}
CROP_SETS = ("C", "D")      # sets whose arm crops on the gaze, so the page draws the box and the crop


def load_ranks() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{tol: {qid: {"rank": {...}, "hit": {...}}}} for tol 0 and 60."""
    out = {}
    for t in ("0", "60"):
        d = load_json(os.path.join(RESULTS, f"recall_pool_all_subset_r005_tol{t}.json"))
        out[t] = {q["ID"]: {"rank": q["rank"], "hit": q["hit@3"]} for q in d["per_question"]}
    return out


def pick(ranks: Dict[str, Dict[str, Any]], arm: str, base: str, direction: str) -> List[str]:
    """Questions where arm hit@3 and base did not (up) or the reverse (down), biggest rank change first."""
    win, lose = (arm, base) if direction == "up" else (base, arm)
    qs = [q for q, r in ranks.items() if r["hit"].get(win) and not r["hit"].get(lose)]
    return sorted(qs, key=lambda q: -math.log(ranks[q]["rank"][lose] / ranks[q]["rank"][win]))


def render_case(q, clip, offset, gaze_entry, out_img: str, size: int):
    from decord import VideoReader, cpu
    from PIL import Image

    path = resolve_video_path(clip["video_path"], VIDEO_ROOT)
    if not os.path.exists(path):
        print(f"  [{q['ID']}] video missing: {path}")
        return None
    vr = VideoReader(path, ctx=cpu(0), num_threads=decord_threads())
    total, fps = len(vr), float(vr.get_avg_fps()) or 30.0
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        fixes = clip_fixations(gaze_entry.get("samples") or [], radius_thresh=FIX_DEFAULTS["radius"],
                               duration_thresh=FIX_DEFAULTS["min_dur"], gap_thresh=FIX_DEFAULTS["gap"],
                               dropout_thresh=FIX_DEFAULTS["dropout"])
    fixes = sorted(fixes, key=lambda f: f["start_time"])
    fix_no = {id(f): n + 1 for n, f in enumerate(fixes)}
    idx = frame_indices_uniform(total, NFRAMES)
    kept = {i for i, _ in select_frames(fixes, total, fps, NFRAMES, "subset")}
    ti = int(min(max(round(offset * fps), 0), total - 1))
    want = sorted(set(idx) | {ti})
    arr = {i: a for i, a in zip(want, vr.get_batch(want).asnumpy())}
    h, w = next(iter(arr.values())).shape[:2]

    def norm(b):
        return [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]

    frames = []
    for n, i in enumerate([ti] + list(idx)):
        t = i / fps
        p = gaze_at(gaze_entry, t)
        fx = fixation_of(fixes, t)
        name = f"{q['ID']}_" + ("tg" if n == 0 else f"{n - 1:02d}") + ".jpg"
        im = Image.fromarray(arr[i]).convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(os.path.join(out_img, name), quality=86)
        fr = {"src": f"img/{name}", "t": round(t, 2), "target": n == 0, "embedded": n > 0 or ti in idx,
              "kept": i in kept if n > 0 else None, "fix": fix_no[id(fx)] if fx else 0}
        if fx:
            fr.update(cx=round(fx["center_x"], 4), cy=round(fx["center_y"], 4))
        if p:
            fr.update(gx=round(p[0], 4), gy=round(p[1], 4), box=norm(crop_box(w, h, p[0], p[1], RATIO)))
        frames.append(fr)
    near = min(idx, key=lambda i: abs(i - ti))      # embedded instant closest to the target moment
    return {"w": w, "h": h, "frames": frames, "n_kept": len(kept), "n_fix": len(fixes),
            "target_in_fix": fixation_of(fixes, ti / fps) is not None, "near_target_kept": near in kept,
            "fixations": [{"s": round(f["start_time"], 2), "e": round(f["end_time"], 2), "n": fix_no[id(f)]} for f in fixes]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", choices=("0", "60"), default="60", help="which scoring picks the cases")
    ap.add_argument("--n", type=int, default=30, help="max cases per set")
    ap.add_argument("--size", type=int, default=560, help="longest side of saved frames, px")
    ap.add_argument("--out", default=os.path.join(HERE, "gaze_cases"))
    args = ap.parse_args()

    questions = {e["ID"]: e for e in load_json(QUESTIONS_PATH)["entries"]}
    clips = load_json(POOL_ALL)["clips"]
    gaze = load_gaze(GAZE_PATH)
    ranks = load_ranks()

    located = {}
    for qid, q in questions.items():
        tc = target_clip(q, clips)
        if tc and tc[0]["key"] in gaze:
            located[qid] = tc

    sets, n_cand = {}, {}
    for key, (arm, base, direction) in SETS.items():
        cand = pick(ranks[args.tol], arm, base, direction)
        n_cand[key] = len(cand)
        sets[key] = [q for q in cand if q in located][:args.n]
        print(f"set {key} ({arm} {'beats' if direction == 'up' else 'loses to'} {base}): "
              f"{len(cand)} candidates at tol {args.tol}, keeping {len(sets[key])}")

    img_dir = os.path.join(args.out, "img")
    os.makedirs(img_dir, exist_ok=True)
    todo = list(dict.fromkeys(q for s in sets.values() for q in s))
    cases = {}
    for n, qid in enumerate(todo, 1):
        q = questions[qid]
        clip, offset, n_moments = located[qid]
        r = render_case(q, clip, offset, gaze[clip["key"]], img_dir, args.size)
        if r is None:
            continue
        r.update({
            "id": qid, "type": q.get("type"), "need_audio": bool(q.get("need_audio")),
            "question": q.get("question"), "choices": q.get("choices") or {}, "answer": q.get("answer"),
            "reason": q.get("reason"), "keywords": q.get("keywords"),
            "clip": clip["key"], "target_offset": round(offset, 2), "n_moments": n_moments,
            "ranks": {t: ranks[t].get(qid) for t in ("0", "60")},
        })
        cases[qid] = r
        print(f"  {n}/{len(todo)} Q{qid} {clip['key']} @ {offset:.1f}s  kept {r['n_kept']}/{NFRAMES}")

    data = {"tol": args.tol, "ratio": RATIO, "fix_radius": FIX_DEFAULTS["radius"], "arms": ARMS, "cases": cases, "crop_sets": list(CROP_SETS),
            "sets": {k: {"arm": a, "base": b, "dir": d, "ids": [q for q in sets[k] if q in cases],
                         "n_candidates": n_cand[k]} for k, (a, b, d) in SETS.items()}}
    with open(os.path.join(args.out, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.GAZE_CASES = " + json.dumps(data, ensure_ascii=False) + ";\n")
    with open(os.path.join(HERE, "gaze_cases_template.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    counts = ", ".join("%d %s" % (len(v["ids"]), k) for k, v in data["sets"].items())
    print(f"wrote {args.out}/index.html  ({counts} cases, {len(os.listdir(img_dir))} frames)")


if __name__ == "__main__":
    main()
