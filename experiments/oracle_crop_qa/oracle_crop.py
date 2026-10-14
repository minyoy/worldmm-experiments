"""Oracle-frame crop QA: does showing the right frames cropped (gaze / center), or full + crop, change QA accuracy?

No retrieval. Every question gets the frames of its TARGET clip(s) (tol=0), 1 fps, at most MAX_TS timestamps with
proportional per-clip sampling. Same question set, target clips and text memory as visual_routing_qa/oracle.py
(text read from visual_routing_qa/run_data/text_context), same QA model, greedy, no extra instruction. All
conditions use the same timestamps; only the images shown per timestamp differ. See PLAN.md (fixed before any
result was seen).

  full                   full frame
  gaze                   gaze crop
  center                 center crop
  dual_center            full, center crop
  dual_gaze              full, gaze crop
  dual_offcenter         full, + gaze crop only when the gaze is >= OFFCENTER_DIST from the frame center
  dual_offcenter_center  full, + center crop on exactly the same timestamps as dual_offcenter (control)

Usage (from the repo root):
    python experiments/oracle_crop_qa/oracle_crop.py --stage preflight      # no GPU; decodes no frames
    python experiments/oracle_crop_qa/oracle_crop.py --stage example        # no GPU; writes input_example.png
    CUDA_VISIBLE_DEVICES=2,3 python -u experiments/oracle_crop_qa/oracle_crop.py --stage qa   # resumable
    python experiments/oracle_crop_qa/analyze_crop.py
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
VRQA = HERE.parent / "visual_routing_qa"
sys.path.insert(0, str(VRQA))
import run as base  # noqa: E402  (shared helpers; also puts src/ and the experiment dirs on sys.path)
import oracle as orc  # noqa: E402

from gaze_common import crop_box, gaze_at, load_gaze  # noqa: E402
from common import VIDEO_ROOT, build_choices, resolve_video_path  # noqa: E402

CONDS = ("full", "gaze", "center", "dual_center", "dual_gaze", "dual_offcenter", "dual_offcenter_center")
VIEWS = {                      # cond -> (views always shown, view added only on off-center timestamps)
    "full": (("full",), None),
    "gaze": (("gaze",), None),
    "center": (("center",), None),
    "dual_center": (("full", "center"), None),
    "dual_gaze": (("full", "gaze"), None),
    "dual_offcenter": (("full",), "gaze"),
    "dual_offcenter_center": (("full",), "center"),
}
MAX_TS = 32                    # timestamps per question; dual conditions show at most 2 x 32 = 64 images
CROP_RATIO = 0.5               # same as gazef_05 / center_05 in gaze_crop
OFFCENTER_DIST = 0.3 * 0.5 ** 0.5   # PLAN 3.1 definition B: 30% of the center-to-corner distance (~0.212), normalized


def load_questions(args):
    """The oracle.py question set and target clips, checked against its manifest."""
    ns = SimpleNamespace(pool=args.pool, qa=args.qa, out=args.text_root, limit=None)
    rows, order, target_keys, clips = orc.load_questions(ns)
    ref = base.read_json(args.text_root / "oracle_manifest.json")
    if order != ref["question_ids"] or any(target_keys[q] != ref["target_keys"][q] for q in order):
        raise ValueError("Question set or target clips differ from visual_routing_qa/run_data/oracle_manifest.json")
    return rows, order, target_keys, clips


def select(order: list[str], args) -> list[str]:
    if args.qids:
        missing = set(args.qids) - set(order)
        if missing:
            raise ValueError(f"Not in the question set: {sorted(missing)}")
        order = [q for q in order if q in set(args.qids)]
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        order = order[i::n]
    return order[:args.limit] if args.limit is not None else order


def timestamps(keys: list[str], clips: dict, video_root: Path, decode: bool) -> list[tuple]:
    """(key, VideoReader, frame_idx, fps) for the target clips: 1 fps, <= MAX_TS, proportional per clip."""
    from decord import VideoReader, cpu

    readers, per_clip = [], []
    for key in keys:
        path = resolve_video_path(clips[key]["video_path"], str(video_root))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        vr = VideoReader(path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        if fps <= 0 or len(vr) < 2:
            raise ValueError(f"Bad video: {path}")
        readers.append((key, vr if decode else None, fps))
        per_clip.append(list(range(0, len(vr), max(1, int(fps)))))
    capped = orc.proportional_cap(per_clip, MAX_TS)
    return [(key, vr, idx, fps) for (key, vr, fps), idxs in zip(readers, capped) for idx in idxs]


def frame_info(slots: list[tuple], gaze: dict) -> list[dict]:
    out = []
    for key, _, idx, fps in slots:
        if key not in gaze or not gaze[key].get("samples"):
            raise ValueError(f"No real gaze samples for {key}")
        xy = gaze_at(gaze[key], idx / fps, tol=0.5)
        if xy is None:
            raise ValueError(f"Missing gaze at {key}:{idx / fps:.2f}s")
        dist = math.hypot(xy[0] - 0.5, xy[1] - 0.5)
        out.append({"key": key, "idx": idx, "t": round(idx / fps, 3), "gaze": [round(xy[0], 4), round(xy[1], 4)],
                    "dist": round(dist, 4), "off": dist >= OFFCENTER_DIST})
    return out


def view_image(image, view: str, xy: list[float]):
    if view == "full":
        return image
    cx, cy = (0.5, 0.5) if view == "center" else xy
    return image.crop(crop_box(*image.size, cx, cy, CROP_RATIO))


def build_input(cond: str, frames: list, info: list[dict]) -> tuple[list, list[str], str]:
    """Images in prompt order ([t1 views..., t2 views...]), their ids and a hash over ids and pixels."""
    always, extra = VIEWS[cond]
    images, ids, digest = [], [], hashlib.sha256()
    for image, f in zip(frames, info):
        for view in always + ((extra,) if extra and f["off"] else ()):
            img = view_image(image, view, f["gaze"])
            tag = f"{view}:{f['key']}:{f['idx']}"
            digest.update(f"{tag}|{img.size}".encode())
            digest.update(img.tobytes())
            ids.append(tag)
            images.append(img)
    return images, ids, digest.hexdigest()


def decode(slots: list[tuple]) -> list:
    from PIL import Image

    return [Image.fromarray(vr[idx].asnumpy()).convert("RGB") for _, vr, idx, _ in slots]


def make_manifest(args, order: list[str], target_keys: dict) -> None:
    code = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest = {"design": "oracle target clips (tol=0), 1 fps, <=MAX_TS timestamps proportional per clip; "
                          "views per timestamp differ by condition; no extra instruction",
                "conditions": list(CONDS), "views": {c: [list(a), b] for c, (a, b) in VIEWS.items()},
                "question_ids": order, "target_keys": {q: target_keys[q] for q in order},
                "max_timestamps": MAX_TS, "crop_ratio": CROP_RATIO, "offcenter_dist": OFFCENTER_DIST,
                "qa_model": args.qa_model, "text_root": str(args.text_root),
                "pool_sha256": base.sha256(args.pool), "qa_sha256": base.sha256(args.qa),
                "gaze_sha256": base.sha256(args.gaze)}
    path = args.out / "crop_manifest.json"
    if path.exists():
        old = base.read_json(path)
        if {k: v for k, v in old.items() if k != "code_sha256"} != manifest:
            raise ValueError("Manifest changed. Use a fresh --out directory")
        if old.get("code_sha256") != code:
            print("warning: oracle_crop.py changed since this --out was started", flush=True)
        manifest["code_sha256"] = old.get("code_sha256")
    else:
        manifest["code_sha256"] = code
    base.write_json(path, manifest)


def all_info(args, order, target_keys, clips, gaze) -> dict:
    """{qid: frame info} without decoding pixels; cached in out/frames/."""
    infos = {}
    for qid in order:
        path = args.out / "frames" / f"{qid}.json"
        if path.exists():
            infos[qid] = base.read_json(path)
            continue
        infos[qid] = frame_info(timestamps(target_keys[qid], clips, args.video_root, decode=False), gaze)
        base.write_json(path, infos[qid])
    return infos


def preflight(args, rows, order, target_keys, clips) -> None:
    gaze = load_gaze(str(args.gaze))
    no_gaze = sorted({k for q in order for k in target_keys[q] if k not in gaze or not gaze[k].get("samples")})
    no_video = sorted({k for q in order for k in target_keys[q]
                       if not os.path.isfile(resolve_video_path(clips[k]["video_path"], str(args.video_root)))})
    no_text = [q for q in order if not base.text_path(SimpleNamespace(out=args.text_root), q).exists()]
    print(f"questions: {len(order)} (need_audio=False: {sum(not rows[q]['need_audio'] for q in order)})")
    print(f"missing video: {len(no_video)}, missing gaze: {len(no_gaze)}, missing text context: {len(no_text)}")
    if no_video or no_gaze or no_text:
        raise ValueError("incomplete inputs; see counts above")
    infos = all_info(args, order, target_keys, clips, gaze)
    n_ts = [len(infos[q]) for q in order]
    n_off = [sum(f["off"] for f in infos[q]) for q in order]
    frames = [f for q in order for f in infos[q]]
    print(f"timestamps per question: min {min(n_ts)}, median {int(np.median(n_ts))}, max {max(n_ts)}; "
          f"capped at {MAX_TS}: {sum(n == MAX_TS for n in n_ts)}")
    print(f"off-center timestamps (dist >= {OFFCENTER_DIST}): {sum(f['off'] for f in frames)}/{len(frames)} "
          f"= {100 * sum(f['off'] for f in frames) / len(frames):.1f}%; questions with >=1: {sum(n > 0 for n in n_off)}, "
          f">=5: {sum(n >= 5 for n in n_off)}, mean per question {np.mean(n_off):.1f}")
    for cond in CONDS:
        always, extra = VIEWS[cond]
        imgs = [len(always) * t + (o if extra else 0) for t, o in zip(n_ts, n_off)]
        print(f"  {cond:22s} images per question: median {int(np.median(imgs))}, mean {np.mean(imgs):.1f}, max {max(imgs)}")
    items = {q: base.text_record(SimpleNamespace(out=args.text_root), q, rows[q])["retrieved_items"] for q in order}
    sizes = {q: len(items[q]) for q in order}
    chars = {q: sum(len(x["content"]) for x in items[q]) for q in order}
    longest = max(order, key=lambda q: chars[q])
    multi = max(order, key=lambda q: (len(target_keys[q]), chars[q]))
    print(f"smoke candidates: longest text {longest} ({chars[longest]} chars, {sizes[longest]} items, "
          f"{len(infos[longest])} ts); most target clips {multi} ({len(target_keys[multi])} clips, {chars[multi]} chars)")


def example(args, rows, order, target_keys, clips) -> None:
    """One single-clip question with the most off-center timestamps: every condition's images for one off-center and
    one central timestamp, as a grid (rows = conditions)."""
    from PIL import Image, ImageDraw

    gaze = load_gaze(str(args.gaze))
    infos = all_info(args, order, target_keys, clips, gaze)
    qid = args.qids[0] if args.qids else max((q for q in order if len(target_keys[q]) == 1),
                                             key=lambda q: max(f["dist"] for f in infos[q]))
    slots = timestamps(target_keys[qid], clips, args.video_root, decode=True)
    info = frame_info(slots, gaze)
    i_off = max(range(len(info)), key=lambda i: info[i]["dist"])
    i_on = min(range(len(info)), key=lambda i: info[i]["dist"])
    picks = sorted({i_off, i_on})
    frames = decode([slots[i] for i in picks])
    sub = [info[i] for i in picks]
    cell, label_w = 220, 190
    rows_img = []
    for cond in CONDS:
        images, ids, _ = build_input(cond, frames, sub)
        row = Image.new("RGB", (label_w + cell * 4, cell + 20), "white")
        d = ImageDraw.Draw(row)
        d.text((6, 8), cond, fill="black")
        for j, (img, tag) in enumerate(zip(images, ids)):
            thumb = img.copy()
            thumb.thumbnail((cell - 8, cell - 8))
            if tag.startswith("full:"):        # mark the gaze on full frames so the crop position can be checked
                f = sub[[f"{x['key']}:{x['idx']}" for x in sub].index(tag.split(":", 1)[1])]
                gx, gy = f["gaze"][0] * thumb.width, f["gaze"][1] * thumb.height
                ImageDraw.Draw(thumb).ellipse([gx - 4, gy - 4, gx + 4, gy + 4], outline=(255, 0, 0), width=2)
            row.paste(thumb, (label_w + j * cell, 4))
            d.text((label_w + j * cell, cell - 2), tag.split(":")[0] + f" idx={tag.rsplit(':', 1)[1]}", fill="black")
        rows_img.append(row)
    head = Image.new("RGB", (label_w + cell * 4, 40), "white")
    ImageDraw.Draw(head).text((6, 6), f"Q{qid}  timestamps: " + ", ".join(
        f"idx {f['idx']} gaze ({f['gaze'][0]:.2f}, {f['gaze'][1]:.2f}) dist {f['dist']:.2f} off={f['off']}" for f in sub),
        fill="black")
    out = Image.new("RGB", (head.width, head.height + sum(r.height for r in rows_img)), "white")
    out.paste(head, (0, 0))
    y = head.height
    for r in rows_img:
        out.paste(r, (0, y))
        y += r.height
    path = HERE / "input_example.png"
    out.save(path)
    print(path)


def run_qa(args, rows, order, target_keys, clips) -> None:
    import torch

    from worldmm.llm import LLMModel, PromptTemplateManager
    from eval_egolife import evaluate_prediction

    gaze = load_gaze(str(args.gaze))
    llm = LLMModel(model_name=args.qa_model, fps=1)
    # qa_prompt only needs the QA template; WorldMemory (and its embedding models) is not loaded.
    wm = SimpleNamespace(prompt_template_manager=PromptTemplateManager(), qa_template_name="qa_egolife")
    text_ns = SimpleNamespace(out=args.text_root)
    for qid in order:
        row = rows[qid]
        todo = [c for c in args.conds if not (args.out / "results" / c / f"{qid}.json").exists()]
        if not todo:
            continue
        context = base.text_record(text_ns, qid, row)
        try:
            slots = timestamps(target_keys[qid], clips, args.video_root, decode=True)
            info = frame_info(slots, gaze)
            frames = decode(slots)
            base.write_json(args.out / "frames" / f"{qid}.json", info)
            slot_error = None
        except Exception as exc:
            frames, info, slot_error = None, None, f"{type(exc).__name__}: {exc}"
        for cond in todo:
            result = {"ID": qid, "condition": cond, "question": row["question"], "answer": row["answer"],
                      "need_audio": bool(row["need_audio"]), "type": row.get("type"),
                      "target_keys": target_keys[qid], "text_hash": context["items_hash"]}
            try:
                if frames is None:
                    raise RuntimeError(slot_error)
                images, ids, image_hash = build_input(cond, frames, info)
                result.update({"frame_ids": ids, "num_images": len(images), "image_hash": image_hash,
                               "n_timestamps": len(info), "n_offcenter": sum(f["off"] for f in info)})
                prompt = base.qa_prompt(wm, row["question"], build_choices(row), context["retrieved_items"], images)
                for d in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(d)
                response = llm.generate(prompt, do_sample=False, max_new_tokens=32)
                result["peak_mem_gb"] = [round(torch.cuda.max_memory_allocated(d) / 2 ** 30, 2)
                                         for d in range(torch.cuda.device_count())]
                result["response"] = response
                result["correct"] = evaluate_prediction(response, row["answer"], build_choices(row))
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
            base.write_json(args.out / "results" / cond / f"{qid}.json", result)
            print(f"crop {qid}/{cond}: {result['status']} {result.get('correct')} "
                  f"imgs={result.get('num_images')} mem={result.get('peak_mem_gb')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("preflight", "example", "qa"), required=True)
    ap.add_argument("--pool", type=Path, default=base.ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--gaze", type=Path, default=base.ROOT / "experiments/gaze_crop/gaze_points.json")
    ap.add_argument("--video-root", type=Path, default=Path(VIDEO_ROOT))
    ap.add_argument("--text-root", type=Path, default=VRQA / "run_data",
                    help="holds text_context/ and oracle_manifest.json of the gaze-prompting oracle run (read only)")
    ap.add_argument("--out", type=Path, default=HERE / "run_data", help="frames/, results/<cond>/, crop_manifest.json")
    ap.add_argument("--qa-model", default="qwen3vl-8b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--qids", nargs="+", help="run only these question ids (smoke)")
    ap.add_argument("--shard", help="i/n: every n-th question starting at i")
    ap.add_argument("--conds", nargs="+", choices=CONDS, default=list(CONDS))
    args = ap.parse_args()
    rows, order, target_keys, clips = load_questions(args)
    make_manifest(args, order, target_keys)
    if args.stage == "preflight":
        preflight(args, rows, order, target_keys, clips)
    elif args.stage == "example":
        example(args, rows, order, target_keys, clips)
    else:
        run_qa(args, rows, select(order, args), target_keys, clips)


if __name__ == "__main__":
    main()
