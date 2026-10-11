#!/usr/bin/env python3
"""
Static case browser for the hypothesis "which question types favour gaze, which favour full?"
(same idea as experiments/gaze_crop/build_gaze_cases.py, but on this experiment's QA results):

  A. gaze-only correct -- qa_gaze=1, qa_full=0
  B. full-only correct -- qa_full=1, qa_gaze=0
  C. router missed     -- full/gaze disagree and the router picked the wrong arm (QA)
  S. summary           -- accuracy / discordance / router picks per question type and need_audio

Each case shows, for the full arm and the gaze arm, the Top-3 retrieved clips with three frames each
(25/50/75 %; full frames for full, gaze crops for gaze, same crop as run.py), a hit mark per clip, the model's
reply, plus the frame at the annotated target moment. The router's two votes are shown too.

Needs decord + PIL and the EgoLife videos (CPU only); writes a folder that opens from disk:

    python experiments/visual_routing_qa/build_routing_cases.py          # -> routing_cases/index.html
    scp -r server:.../experiments/visual_routing_qa/routing_cases .  &&  open routing_cases/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "experiments" / "gaze_crop"), str(ROOT / "experiments" / "visual_bottleneck")]

from gaze_common import (GAZE_PATH, QUESTIONS_PATH, VIDEO_ROOT, clip_span, crop_box, gaze_at,  # noqa: E402
                         load_gaze, load_json, resolve_video_path, target_spans)

POOL_ALL = ROOT / "experiments" / "gaze_crop" / "pool_all.json"
FRACS = (0.25, 0.5, 0.75)
RATIO = 0.5
MAX_GAP_SEC = 15.0
ARMS = ("full", "gaze")


def read_results(run: Path, qid: str) -> dict | None:
    out = {}
    for arm in ARMS:
        p = run / "results" / arm / f"{qid}.json"
        if not p.exists():
            return None
        out[arm] = load_json(str(p))
    r = run / "routing" / f"{qid}.json"
    out["route"] = load_json(str(r)) if r.exists() else None
    return out if all(out[a].get("status") == "ok" for a in ARMS) and out["route"] else None


def target_clip(q: dict, clips: list[dict]):
    spans = target_spans(q.get("target_time"), (q.get("target_time") or {}).get("date"))
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
    return best, min(max(m - a, 0.0), b - a)


class Frames:
    """Decode a few frames per clip, caching the VideoReader of the clip being worked on."""

    def __init__(self, clips: dict, gaze: dict, out_img: Path, size: int):
        self.clips, self.gaze, self.out, self.size = clips, gaze, out_img, size

    def _reader(self, key):
        from decord import VideoReader, cpu
        path = resolve_video_path(self.clips[key]["video_path"], VIDEO_ROOT)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        vr = VideoReader(path, ctx=cpu(0))
        return vr, float(vr.get_avg_fps()) or 30.0

    def _save(self, im, name):
        im.thumbnail((self.size, self.size))
        im.save(self.out / name, quality=85)
        return f"img/{name}"

    def clip_frames(self, qid, arm, key, rank):
        from PIL import Image
        vr, fps = self._reader(key)
        shots = []
        for n, frac in enumerate(FRACS):
            idx = int(round(frac * (len(vr) - 1)))
            im = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
            shot = {"t": round(idx / fps, 1)}
            if arm == "gaze":
                xy = gaze_at(self.gaze[key], idx / fps, tol=0.5)
                if xy is not None:
                    w, h = im.size
                    box = crop_box(w, h, xy[0], xy[1], RATIO)
                    shot["full"] = self._save(im.copy(), f"{qid}_{arm}_{rank}_{n}_full.jpg")
                    shot["box"] = [round(box[0] / w, 4), round(box[1] / h, 4), round(box[2] / w, 4), round(box[3] / h, 4)]
                    shot["gx"], shot["gy"] = round(xy[0], 4), round(xy[1], 4)
                    im = im.crop(box)
            shot["src"] = self._save(im, f"{qid}_{arm}_{rank}_{n}.jpg")
            shots.append(shot)
        return shots

    def target(self, qid, key, offset):
        from PIL import Image
        vr, fps = self._reader(key)
        idx = int(min(max(round(offset * fps), 0), len(vr) - 1))
        im = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
        xy = gaze_at(self.gaze[key], idx / fps, tol=0.5) if key in self.gaze else None
        w, h = im.size
        box = crop_box(w, h, xy[0], xy[1], RATIO) if xy is not None else None
        return {"src": self._save(im, f"{qid}_target.jpg"), "t": round(idx / fps, 1),
                "gx": None if xy is None else round(xy[0], 4), "gy": None if xy is None else round(xy[1], 4),
                "box": None if box is None else [round(box[0] / w, 4), round(box[1] / h, 4),
                                                 round(box[2] / w, 4), round(box[3] / h, 4)]}


def summary_rows(records: dict, questions: dict) -> list[dict]:
    """Per (type, need_audio) slice: N, accuracy of full/gaze, discordant counts, router picks."""
    groups = defaultdict(list)
    for qid, r in records.items():
        q = questions[qid]
        groups[(q["type"], bool(q.get("need_audio")))].append(qid)
    rows = []
    for label, pick in [("전체", lambda k: True)] + \
            [(f"{t} · {'오디오' if a else '시각'}", (lambda k, t=t, a=a: k == (t, a)))
             for t, a in sorted(groups)] + \
            [("오디오 필요", lambda k: k[1]), ("시각 근거", lambda k: not k[1])]:
        ids = [qid for k, v in groups.items() if pick(k) for qid in v]
        if not ids:
            continue
        g = lambda arm, qid: int(bool(records[qid][arm]["correct"]))  # noqa: E731
        rows.append({
            "label": label, "n": len(ids),
            "full": sum(g("full", i) for i in ids), "gaze": sum(g("gaze", i) for i in ids),
            "gaze_only": sum(g("gaze", i) and not g("full", i) for i in ids),
            "full_only": sum(g("full", i) and not g("gaze", i) for i in ids),
            "hit60_full": sum(bool(records[i]["full"]["hit_at_3"]["60"]) for i in ids),
            "hit60_gaze": sum(bool(records[i]["gaze"]["hit_at_3"]["60"]) for i in ids),
            "route_gaze": sum(records[i]["route"]["decision"] == "gaze" for i in ids),
            "route_ok": sum(records[i]["route"]["decision"] == ("gaze" if g("gaze", i) > g("full", i) else "full")
                            for i in ids if g("gaze", i) != g("full", i)),
        })
    return rows


def hit_qa_table(records: dict) -> list[dict]:
    """2x2 of retrieval Hit@3 vs QA correctness per arm and tolerance (tol 0 drops questions with no target clip)."""
    out = []
    for tol, label in (("0", "tol=0"), ("60", "tol=60s")):
        ids = [q for q, r in records.items() if r["full"]["target_count"][tol] > 0]
        for arm in ARMS:
            c = defaultdict(int)
            for q in ids:
                c[(bool(records[q][arm]["hit_at_3"][tol]), bool(records[q][arm]["correct"]))] += 1
            out.append({"tol": label, "arm": arm, "n": len(ids), "hit_ok": c[(True, True)],
                        "hit_wrong": c[(True, False)], "miss_ok": c[(False, True)], "miss_wrong": c[(False, False)]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=HERE / "run_data")
    ap.add_argument("--n", type=int, default=30, help="max cases per tab")
    ap.add_argument("--size", type=int, default=360, help="longest side of saved frames, px")
    ap.add_argument("--out", type=Path, default=HERE / "routing_cases")
    args = ap.parse_args()

    questions = {str(e["ID"]): e for e in load_json(QUESTIONS_PATH)["entries"]}
    clips = {c["key"]: c for c in load_json(str(POOL_ALL))["clips"]}
    clip_list = list(clips.values())
    gaze = load_gaze(GAZE_PATH)

    records = {}
    for qid in questions:
        r = read_results(args.run, qid)
        if r:
            records[qid] = r
    print(f"{len(records)} questions with ok full/gaze results and routing")

    ok = lambda r, arm: bool(r[arm]["correct"])  # noqa: E731
    only_gaze = [q for q, r in records.items() if ok(r, "gaze") and not ok(r, "full")]
    only_full = [q for q, r in records.items() if ok(r, "full") and not ok(r, "gaze")]
    missed = [q for q in only_gaze + only_full
              if records[q]["route"]["decision"] != ("gaze" if q in only_gaze else "full")]
    print(f"A gaze-only {len(only_gaze)}, B full-only {len(only_full)}, C router missed {len(missed)}")

    sets = {"A": only_gaze[:args.n], "B": only_full[:args.n], "C": missed[:args.n]}
    todo = list(dict.fromkeys(q for s in sets.values() for q in s))
    img_dir = args.out / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = Frames(clips, gaze, img_dir, args.size)

    cases = {}
    for n, qid in enumerate(todo, 1):
        q, r = questions[qid], records[qid]
        try:
            arms = {}
            for arm in ARMS:
                tolerant = r[arm]["hit_at_3"]
                arms[arm] = {
                    "response": r[arm]["response"], "correct": ok(r, arm), "hit": tolerant,
                    "clips": [{"key": h["key"], "score": round(h["score"], 3), "rank": i + 1,
                               "frames": frames.clip_frames(qid, arm, h["key"], i + 1)}
                              for i, h in enumerate(r[arm]["top3"])]}
            tc = target_clip(q, clip_list)
            target = None
            if tc:
                target = frames.target(qid, tc[0]["key"], tc[1])
                target["clip"] = tc[0]["key"]
                target["in_top3"] = {a: tc[0]["key"] in {h["key"] for h in r[a]["top3"]} for a in ARMS}
        except Exception as exc:  # one bad video must not sink the page
            print(f"  skip Q{qid}: {type(exc).__name__}: {exc}")
            continue
        route = r["route"]
        cases[qid] = {"id": qid, "type": q.get("type"), "need_audio": bool(q.get("need_audio")),
                      "question": q["question"], "choices": q.get("choices") or {}, "answer": q["answer"],
                      "reason": q.get("reason"), "keywords": q.get("keywords"), "arms": arms,
                      "target": target,
                      "route": {"decision": route["decision"], "agree": route["agree"],
                                "votes": [{"set1": v["set1"], "raw": v["raw"], "picked": v["picked"]}
                                          for v in route["votes"]]}}
        print(f"  {n}/{len(todo)} Q{qid}")

    data = {"sets": {k: [q for q in v if q in cases] for k, v in sets.items()},
            "n_candidates": {"A": len(only_gaze), "B": len(only_full), "C": len(missed)},
            "summary": summary_rows(records, questions), "hit_qa": hit_qa_table(records), "n_total": len(records), "cases": cases}
    (args.out / "data.js").write_text("window.ROUTING_CASES = " + json.dumps(data, ensure_ascii=False) + ";\n",
                                      encoding="utf-8")
    html = (HERE / "routing_cases_template.html").read_text(encoding="utf-8")
    (args.out / "index.html").write_text(html, encoding="utf-8")
    print(f"wrote {args.out}/index.html  ({len(cases)} cases, {len(os.listdir(img_dir))} frames)")


if __name__ == "__main__":
    main()
