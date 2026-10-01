#!/usr/bin/env python3
"""
A small static website for eyeballing questions, one case at a time:

  A. gazef-only hits -- questions where the gaze crop put the answer clip in the top 3 and the
     whole frame did not (hit@3 gazef=1, full=0 in results/recall_arms_tol<T>.json). Shown with the
     frame, the wearer's gaze point, the gazef crop box, and the crop itself next to the full frame,
     plus every arm's rank so "gaze helped" can be told apart from "cropping helped" (center hit too).
  B. off-centre gaze -- questions whose target clip has the gaze furthest outside the centre crop
     box across the 16 embedded frames. Shown with the frame, the gaze point, question and choices.

  C. gaze-hurt misses -- the mirror of A: questions where the whole frame put the answer clip in the
     top 3 and the gaze crop did not (hit@3 full=1, gazef=0). Same view as A, ordered by the biggest
     rank loss. If center also missed, the crop itself (not the gaze) is what lost the clip.

Each case shows the frame at the annotated target moment plus the 16 frames the embedding actually
used (click to switch). Frame times, gaze lookup and crop boxes are embed_arms.py's own
(frame_indices_uniform, gaze_at, crop_box), so the boxes are what gazef embedded.

Runs where the videos and gaze live (needs decord + PIL, CPU only). Writes a folder that opens from
disk -- no server needed:

    python experiments/gaze_crop/build_gaze_cases.py                  # -> gaze_cases/index.html
    python experiments/gaze_crop/build_gaze_cases.py --tol 0          # A from the strict scoring
    scp -r server:.../experiments/gaze_crop/gaze_cases .  &&  open gaze_cases/index.html

The page keeps a per-question verdict (helps / doesn't / unsure) and a note in the browser, and
exports them as JSON. Frames are EgoLife video, so the folder is git-ignored.
"""

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gaze_common import (  # noqa: E402
    GAZE_PATH, NFRAMES, QUESTIONS_PATH, RESULTS_DIR, VIDEO_ROOT, clip_span, crop_box, decord_threads,
    gaze_at, load_gaze, load_json, resolve_video_path, target_spans,
)
from embed_arms import frame_indices_uniform  # noqa: E402

RATIO = 0.5
POOL_ALL = os.path.join(HERE, "pool_all.json")
ARMS = ["full", "center_05", "gazef_05", "gazef_05_randf"]
MAX_GAP_SEC = 15.0      # a target moment in the gap between clips snaps to a clip this close


# ---------------------------------------------------------------- selection

def load_ranks() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{tol: {qid: {"rank": {...}, "hit@3": {...}}}} for tol 0 and 60."""
    out = {}
    for t in ("0", "60"):
        d = load_json(os.path.join(RESULTS_DIR, f"recall_arms_tol{t}.json"))
        out[t] = {q["ID"]: {"rank": q["rank"], "hit": q["hit@3"]} for q in d["per_question"]}
    return out


def target_clip(q: Dict[str, Any], clips: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], float, int]]:
    """(clip, target offset in seconds inside it, number of annotated moments) for the FIRST
    annotated moment, or None. A moment that falls in a gap between clips snaps to the nearest clip
    within MAX_GAP_SEC (13 questions have no clip at tol 0 for exactly this reason)."""
    date = (q.get("target_time") or {}).get("date")
    spans = target_spans(q.get("target_time"), date)
    if not spans:
        return None
    m = spans[0][0]
    best, best_d = None, None
    for c in clips:
        a, b = clip_span(c)
        d = 0.0 if a <= m <= b else min(abs(m - a), abs(m - b))
        if best_d is None or d < best_d:
            best, best_d = c, d
    if best is None or best_d > MAX_GAP_SEC:
        return None
    a, b = clip_span(best)
    return best, min(max(m - a, 0.0), b - a), len(spans)


def off_centre(entry: Dict[str, Any]) -> Tuple[float, float]:
    """(fraction of the 16 embedded frame times whose gaze is outside the centre box, mean offset).
    Uses clip duration from the gaze samples, as plot_arm_examples.pick_clip does, so it needs no
    decoding; assumes a square frame (Aria RGB is), which the page recomputes exactly afterwards."""
    s = entry.get("samples") or []
    if len(s) < 2:
        return 0.0, 0.0
    dur = max(float(x[0]) for x in s)
    pts = [gaze_at(entry, dur * i / (NFRAMES - 1)) for i in range(NFRAMES)]
    pts = [p for p in pts if p]
    if not pts:
        return 0.0, 0.0
    half = RATIO / 2
    out = sum(1 for x, y in pts if abs(x - 0.5) > half or abs(y - 0.5) > half)
    return out / len(pts), sum(math.hypot(x - 0.5, y - 0.5) for x, y in pts) / len(pts)


# ---------------------------------------------------------------- frames

def render_case(q, clip, offset, gaze_entry, out_img: str, size: int) -> Optional[Dict[str, Any]]:
    from decord import VideoReader, cpu
    from PIL import Image

    path = resolve_video_path(clip["video_path"], VIDEO_ROOT)
    if not os.path.exists(path):
        print(f"  [{q['ID']}] video missing: {path}")
        return None
    vr = VideoReader(path, ctx=cpu(0), num_threads=decord_threads())
    total, fps = len(vr), float(vr.get_avg_fps()) or 30.0
    idx = frame_indices_uniform(total, NFRAMES)
    ti = int(min(max(round(offset * fps), 0), total - 1))
    want = sorted(set(idx) | {ti})
    arr = {i: a for i, a in zip(want, vr.get_batch(want).asnumpy())}
    h, w = next(iter(arr.values())).shape[:2]
    cbox = crop_box(w, h, 0.5, 0.5, RATIO)

    def norm(b):
        return [round(b[0] / w, 4), round(b[1] / h, 4), round(b[2] / w, 4), round(b[3] / h, 4)]

    frames = []
    for n, i in enumerate([ti] + list(idx)):
        p = gaze_at(gaze_entry, i / fps)
        name = f"{q['ID']}_" + ("tg" if n == 0 else f"{n - 1:02d}") + ".jpg"
        im = Image.fromarray(arr[i]).convert("RGB")
        im.thumbnail((size, size), Image.LANCZOS)
        im.save(os.path.join(out_img, name), quality=86)
        fr = {"src": f"img/{name}", "t": round(i / fps, 2), "target": n == 0,
              "embedded": n > 0 or ti in idx}
        if p:
            box = crop_box(w, h, p[0], p[1], RATIO)
            fr.update(gx=round(p[0], 4), gy=round(p[1], 4), box=norm(box),
                      in_center=cbox[0] <= p[0] * w <= cbox[2] and cbox[1] <= p[1] * h <= cbox[3])
        frames.append(fr)
    return {"w": w, "h": h, "center_box": norm(cbox), "frames": frames}


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", choices=("0", "60"), default="60",
                    help="which scoring picks set A (tol 0: 19 questions, tol 60: 38)")
    ap.add_argument("--n-a", type=int, default=30)
    ap.add_argument("--n-b", type=int, default=30)
    ap.add_argument("--n-c", type=int, default=30)
    ap.add_argument("--size", type=int, default=560, help="longest side of saved frames, px")
    ap.add_argument("--out", default=os.path.join(HERE, "gaze_cases"))
    args = ap.parse_args()

    questions = {e["ID"]: e for e in load_json(QUESTIONS_PATH)["entries"]}
    clips = load_json(POOL_ALL)["clips"]
    gaze = load_gaze(GAZE_PATH)
    ranks = load_ranks()

    # A: gazef hit@3, full missed. Order: biggest rank gain first (log ratio, so 300->5 beats 8->2).
    cand_a = [qid for qid, r in ranks[args.tol].items() if r["hit"]["gazef_05"] and not r["hit"]["full"]]
    cand_a.sort(key=lambda qid: -math.log(ranks[args.tol][qid]["rank"]["full"] /
                                          ranks[args.tol][qid]["rank"]["gazef_05"]))
    print(f"set A: {len(cand_a)} gazef-only hits at tol {args.tol}, keeping {min(len(cand_a), args.n_a)}")

    # C: full hit@3, gazef missed. Order: biggest rank loss first.
    cand_c = [qid for qid, r in ranks[args.tol].items() if r["hit"]["full"] and not r["hit"]["gazef_05"]]
    cand_c.sort(key=lambda qid: -math.log(ranks[args.tol][qid]["rank"]["gazef_05"] /
                                          ranks[args.tol][qid]["rank"]["full"]))
    print(f"set C: {len(cand_c)} gaze-hurt misses at tol {args.tol}, keeping {min(len(cand_c), args.n_c)}")

    # B: every question, by how far off-centre its target clip's gaze sits
    located: Dict[str, Tuple[Dict[str, Any], float, int]] = {}
    scored_b = []
    for qid, q in questions.items():
        tc = target_clip(q, clips)
        if not tc or tc[0]["key"] not in gaze:
            continue
        located[qid] = tc
        frac, off = off_centre(gaze[tc[0]["key"]])
        scored_b.append((frac, off, qid))
    scored_b.sort(reverse=True)
    set_b = [qid for _, _, qid in scored_b[:args.n_b]]
    print(f"set B: {len(scored_b)} questions with a located, gazed target clip; top off-centre "
          f"fraction {scored_b[0][0]:.2f} .. #{len(set_b)} {scored_b[len(set_b) - 1][0]:.2f}")

    set_a = [qid for qid in cand_a if qid in located][:args.n_a]
    set_c = [qid for qid in cand_c if qid in located][:args.n_c]
    skipped = [qid for qid in cand_a[:args.n_a] if qid not in located]
    if skipped:
        print(f"  A: no locatable target clip / gaze for {skipped}")

    img_dir = os.path.join(args.out, "img")
    os.makedirs(img_dir, exist_ok=True)
    offc = {qid: (f, o) for f, o, qid in scored_b}
    cases = {}
    todo = list(dict.fromkeys(set_a + set_b + set_c))
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
            "off_centre_frac": round(offc.get(qid, (0, 0))[0], 3),
        })
        cases[qid] = r
        print(f"  {n}/{len(todo)} Q{qid} {clip['key']} @ {offset:.1f}s")

    data = {"tol_a": args.tol, "ratio": RATIO, "arms": ARMS, "cases": cases,
            "A": [q for q in set_a if q in cases], "B": [q for q in set_b if q in cases],
            "C": [q for q in set_c if q in cases],
            "n_candidates_a": len(cand_a), "n_candidates_c": len(cand_c)}
    with open(os.path.join(args.out, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.GAZE_CASES = " + json.dumps(data, ensure_ascii=False) + ";\n")
    with open(os.path.join(HERE, "gaze_cases_template.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}/index.html  ({len(data['A'])} A + {len(data['B'])} B + {len(data['C'])} C cases, "
          f"{len(os.listdir(img_dir))} frames)")


if __name__ == "__main__":
    main()
