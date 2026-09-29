#!/usr/bin/env python3
"""
EgoServe 검색 결과를 사례별로 눈으로 확인하는 정적 웹페이지를 만든다 (gaze_crop/build_gaze_cases.py 의 EgoServe 판).

  A. gazef 가 full 보다 정답 순위가 크게 좋은 질의 (순위가 절반 이하로 줄어든 것, 좋아진 폭 순)
  B. 반대로 full 이 gazef 보다 크게 좋은 질의 (gaze crop 이 해로운 경우)

heard_speech 그룹(근거가 대화라 화면에 답이 없을 수 있는 것)은 처음부터 뺀다.
순위는 --tol 채점(기본 30s)으로 고르고, 페이지에는 tol 0 / 30 / 60 순위를 모두 보여 준다.

각 사례는 정답 클립의 16프레임 + 정답 시점 프레임(시선 점, gazef crop 박스)과, 질의 시점(지금 장면)의
프레임 한 장을 보여 준다. 프레임 시각·시선 조회·crop 박스는 gaze_crop 의 embed_arms.py 와 같다.

영상과 gaze 가 있는 곳에서 돌린다 (decord + PIL 필요, CPU 만 씀). 결과는 서버 없이 디스크에서 열린다.

    python experiments/egoserve_gaze_crop/build_egoserve_cases.py       # -> egoserve_cases/index.html
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(HERE, "..", "gaze_crop")
sys.path.insert(0, GC)
sys.path.insert(0, os.path.join(HERE, "..", "visual_bottleneck"))
import build_gaze_cases as bgc  # noqa: E402
from gaze_common import (  # noqa: E402
    GAZE_PATH, NFRAMES, VIDEO_ROOT, _abs_sec, clip_span, decord_threads, load_gaze, load_json,
    resolve_video_path,
)

TOLS = ("0", "30", "60")
ARMS = bgc.ARMS
EXCLUDE = {"heard_speech"}


def load_ranks(label):
    out = {}
    for t in TOLS:
        d = load_json(os.path.join(HERE, "results", f"recall_egoserve_{label}_tol{t}.json"))
        out[t] = {q["ID"]: {"rank": q["rank"], "hit": q["hit@3"]} for q in d["per_question"] if q.get("rank")}
    return out


def render_now(q, clips, out_img, size):
    """질의 시점(현재 장면 시작)의 프레임 한 장. 그 시각을 덮는 클립이 없으면 None."""
    from decord import VideoReader, cpu
    from PIL import Image

    qt = str(q["query_time"])
    m = _abs_sec("DAY" + qt[0], qt[1:])
    best = None
    for c in clips:
        a, b = clip_span(c)
        if a <= m <= b:
            best = (c, m - a)
            break
    if best is None:
        return None
    clip, off = best
    path = resolve_video_path(clip["video_path"], VIDEO_ROOT)
    if not os.path.exists(path):
        return None
    vr = VideoReader(path, ctx=cpu(0), num_threads=decord_threads())
    fps = float(vr.get_avg_fps()) or 30.0
    i = int(min(max(round(off * fps), 0), len(vr) - 1))
    im = Image.fromarray(vr[i].asnumpy()).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    name = f"{q['ID']}_now.jpg"
    im.save(os.path.join(out_img, name), quality=86)
    return f"img/{name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="pastobs", help="results/recall_egoserve_<label>_tol*.json")
    ap.add_argument("--tol", choices=TOLS, default="30", help="which scoring picks sets A and B")
    ap.add_argument("--min-ratio", type=float, default=2.0, help="keep cases whose rank changed by at least this factor")
    ap.add_argument("--n-a", type=int, default=40)
    ap.add_argument("--n-b", type=int, default=40)
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--out", default=os.path.join(HERE, "egoserve_cases"))
    args = ap.parse_args()

    pool = load_json(os.path.join(HERE, "pool_egoserve.json"))
    questions = {q["ID"]: q for q in pool["questions"] if q["group"] not in EXCLUDE}
    clips = load_json(os.path.join(GC, "pool_all.json"))["clips"]
    gaze = load_gaze(GAZE_PATH)
    ranks = load_ranks(args.label)

    delta = {}
    for qid in questions:
        r = ranks[args.tol].get(qid)
        if r:
            delta[qid] = math.log(r["rank"]["full"] / r["rank"]["gazef_05"])
    thr = math.log(args.min_ratio)
    cand_a = sorted([q for q, d in delta.items() if d >= thr], key=lambda q: -delta[q])
    cand_b = sorted([q for q, d in delta.items() if -d >= thr], key=lambda q: delta[q])
    print(f"tol {args.tol}: {len(delta)} scored queries (heard_speech excluded); "
          f"A candidates {len(cand_a)}, B candidates {len(cand_b)} (rank changed >= {args.min_ratio}x)")

    located = {}
    for qid in delta:
        tc = bgc.target_clip(questions[qid], clips)
        if tc and tc[0]["key"] in gaze:
            located[qid] = tc
    set_a = [q for q in cand_a if q in located][:args.n_a]
    set_b = [q for q in cand_b if q in located][:args.n_b]
    lost = [q for q in cand_a[:args.n_a] + cand_b[:args.n_b] if q not in located]
    if lost:
        print(f"  no locatable target clip / gaze for {lost}")

    img_dir = os.path.join(args.out, "img")
    os.makedirs(img_dir, exist_ok=True)
    cases = {}
    todo = list(dict.fromkeys(set_a + set_b))
    for n, qid in enumerate(todo, 1):
        q = questions[qid]
        clip, offset, _ = located[qid]
        r = bgc.render_case(q, clip, offset, gaze[clip["key"]], img_dir, args.size)
        if r is None:
            continue
        frac, _ = bgc.off_centre(gaze[clip["key"]])
        r.update({
            "id": qid, "type": q["type"], "group": q["group"], "gap_sec": q["gap_sec"],
            "question": q["question"], "assistant_utterance": q.get("assistant_utterance", ""),
            "current_observation": q.get("current_observation", ""),
            "current_local_context": q.get("current_local_context", ""),
            "current_window": q["current_window"], "past_window": q["past_window"],
            "clip": clip["key"], "target_offset": round(offset, 2),
            "ranks": {t: ranks[t].get(qid) for t in TOLS},
            "off_centre_frac": round(frac, 3),
            "now_src": render_now(q, clips, img_dir, args.size),
        })
        cases[qid] = r
        print(f"  {n}/{len(todo)} #{qid} {q['group']} {clip['key']} @ {offset:.1f}s")

    data = {"tol_a": args.tol, "ratio": bgc.RATIO, "arms": ARMS, "cases": cases,
            "A": [q for q in set_a if q in cases], "B": [q for q in set_b if q in cases],
            "n_candidates_a": len(cand_a)}
    with open(os.path.join(args.out, "data.js"), "w", encoding="utf-8") as f:
        f.write("window.GAZE_CASES = " + json.dumps(data, ensure_ascii=False) + ";\n")
    with open(os.path.join(HERE, "egoserve_cases_template.html"), encoding="utf-8") as f:
        html = f.read()
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {args.out}/index.html  ({len(data['A'])} A + {len(data['B'])} B cases, "
          f"{len(os.listdir(img_dir))} images)")


if __name__ == "__main__":
    main()
