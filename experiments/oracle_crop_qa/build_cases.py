#!/usr/bin/env python3
"""Static case browser for the oracle crop experiment: the questions whose outcome depends on the condition.

  tab "correct"  questions whose correctness differs across the 7 conditions
  tab "letter"   questions whose chosen letter (A-D) differs across the 7 conditions (superset incl. always-wrong)

Each case shows the question, choices, gold answer and annotated reason, every condition's reply, the text memory,
and up to N_SHOW timestamps of the oracle input: the full frame (gaze dot, center box, gaze box drawn in the page,
not on the image), the center crop and the gaze crop (same crop_box as oracle_crop.py). Timestamps: the one nearest
the annotated target time, the most off-center ones, then evenly spaced.

Needs decord + PIL and the EgoLife videos (CPU only); writes a folder that opens from disk:
    python experiments/oracle_crop_qa/build_cases.py            # -> experiments/oracle_crop_qa/cases/index.html
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle_crop as oc  # noqa: E402  (also sets up the shared sys.path)

from gaze_common import clip_span, crop_box, target_spans  # noqa: E402
from common import VIDEO_ROOT, build_choices, resolve_video_path  # noqa: E402

N_SHOW = 6
FULL_PX, CROP_PX = 384, 256


def letter(response: str) -> str:
    s = (response or "").strip().lstrip("(").upper()
    return s[:1] if s[:1] in "ABCD" else (response or "").strip()[:12] or "∅"


def pick(info: list[dict], spans: list[tuple], starts: dict) -> tuple[list[int], int]:
    absolute = [starts[f["key"]] + f["t"] for f in info]
    points = [p for a, b in spans for p in (a, b)] or absolute[:1]
    target = min(range(len(info)), key=lambda i: min(abs(absolute[i] - p) for p in points))
    chosen = [target]
    for i in sorted((i for i, f in enumerate(info) if f["off"]), key=lambda i: -info[i]["dist"])[:2]:
        if i not in chosen:
            chosen.append(i)
    n = len(info)
    for k in range(N_SHOW):
        if len(chosen) >= min(N_SHOW, n):
            break
        i = round(k * (n - 1) / max(1, N_SHOW - 1))
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen), target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=HERE / "run_data")
    ap.add_argument("--out", type=Path, default=HERE / "cases")
    ap.add_argument("--pool", type=Path, default=oc.base.ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=oc.base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--video-root", type=Path, default=Path(VIDEO_ROOT))
    args = ap.parse_args()
    from decord import VideoReader, cpu
    from PIL import Image

    rows = {str(r["ID"]): r for r in oc.base.read_json(args.qa)}
    clips = {c["key"]: c for c in oc.base.read_json(args.pool)["clips"]}
    manifest = oc.base.read_json(args.run / "crop_manifest.json")
    res = {c: {q: oc.base.read_json(args.run / "results" / c / f"{q}.json") for q in manifest["question_ids"]}
           for c in oc.CONDS}
    qids = manifest["question_ids"]
    corr_var = [q for q in qids if len({bool(res[c][q]["correct"]) for c in oc.CONDS}) > 1]
    letter_var = [q for q in qids if len({letter(res[c][q]["response"]) for c in oc.CONDS}) > 1]
    show = [q for q in qids if q in set(corr_var) | set(letter_var)]
    img_dir = args.out / "img"
    if args.out.exists():
        shutil.rmtree(args.out)
    img_dir.mkdir(parents=True)
    text_ns = oc.SimpleNamespace(out=oc.VRQA / "run_data")
    cases = []
    for n, qid in enumerate(show, 1):
        row, full = rows[qid], res["full"][qid]
        info = oc.base.read_json(args.run / "frames" / f"{qid}.json")
        starts = {k: clip_span(clips[k])[0] for k in full["target_keys"]}
        qt = row["query_time"]
        spans = target_spans(row.get("target_time"), qt.get("date") if isinstance(qt, dict) else None)
        chosen, target = pick(info, spans, starts)
        readers = {}
        frames = []
        for i in chosen:
            f = info[i]
            if f["key"] not in readers:
                path = resolve_video_path(clips[f["key"]]["video_path"], str(args.video_root))
                readers[f["key"]] = VideoReader(path, ctx=cpu(0))
            im = Image.fromarray(readers[f["key"]][f["idx"]].asnumpy()).convert("RGB")
            w, h = im.size
            boxes = {}
            for view in ("center", "gaze"):
                box = crop_box(w, h, *((0.5, 0.5) if view == "center" else f["gaze"]), oc.CROP_RATIO)
                boxes[view] = [box[0] / w, box[1] / h, (box[2] - box[0]) / w, (box[3] - box[1]) / h]
                c = im.crop(box)
                c.thumbnail((CROP_PX, CROP_PX))
                c.save(img_dir / f"{qid}_{i}_{view}.jpg", quality=82)
            small = im.copy()
            small.thumbnail((FULL_PX, FULL_PX))
            small.save(img_dir / f"{qid}_{i}_full.jpg", quality=82)
            frames.append({"i": i, "key": f["key"], "t": f["t"], "gaze": f["gaze"], "dist": f["dist"], "off": f["off"],
                           "target": i == target, "boxes": boxes,
                           "img": {v: f"img/{qid}_{i}_{v}.jpg" for v in ("full", "center", "gaze")}})
        text = oc.base.text_record(text_ns, qid, row)
        c_ok, f_ok = res["center"][qid]["correct"], full["correct"]
        cases.append({
            "id": qid, "type": row.get("type"), "need_audio": bool(row["need_audio"]),
            "question": row["question"], "choices": build_choices(row), "answer": row["answer"],
            "reason": row.get("reason"), "keywords": row.get("keywords"),
            "n_clips": len(full["target_keys"]), "n_ts": len(info), "n_off": sum(f["off"] for f in info),
            "group": "center" if c_ok and not f_ok else "full" if f_ok and not c_ok else "other",
            "tabs": [t for t, s in (("correct", corr_var), ("letter", letter_var)) if qid in s],
            "conds": {c: {"letter": letter(res[c][qid]["response"]), "response": res[c][qid]["response"],
                          "correct": bool(res[c][qid]["correct"]), "n_img": res[c][qid]["num_images"]}
                      for c in oc.CONDS},
            "frames": frames,
            "text": [x["content"] for x in text["retrieved_items"]],
        })
        print(f"{n}/{len(show)} {qid}", flush=True)
    data = {"conds": list(oc.CONDS), "cases": cases, "counts": {"correct": len(corr_var), "letter": len(letter_var)},
            "offcenter_dist": manifest["offcenter_dist"]}
    (args.out / "data.js").write_text("window.DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    shutil.copy(HERE / "cases_template.html", args.out / "index.html")
    print(args.out / "index.html", f"correct={len(corr_var)} letter={len(letter_var)} shown={len(show)}")


if __name__ == "__main__":
    main()
