"""Oracle-frame gaze prompting: how much does conveying the gaze help when the right frames are present?

No retrieval. Every question gets the frames of its TARGET clip(s) (tol=0, the clips whose span overlaps the annotated
target_time), 1 fps, at most 64 frames with proportional per-clip sampling (the rule of
visual_bottleneck/extract_oracle_frames.py). Same text memory as run.py (run_data/text_context), same QA model, greedy.
Only the way the gaze is conveyed differs between conditions.

Pre-registered design (fixed before any oracle result was seen)
  questions : all questions with at least one target clip at tol=0 (487 of 500)
  conditions: oracle_full (no gaze information), oracle_text, oracle_mark, oracle_saliency
  analyses  : (1) all questions with a target clip, (2) need_audio=False (visual-evidence) questions only
  primary   : oracle_{text,mark,saliency} - oracle_full, paired, on analysis (1)

Usage (from the repo root, in the worldmm env):
    python experiments/visual_routing_qa/oracle.py --stage preflight        # no GPU: counts questions, clips, frames
    python experiments/visual_routing_qa/oracle.py --stage qa               # GPU; resumable
    python experiments/visual_routing_qa/analyze_oracle.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402  (shared helpers; also puts src/ and the experiment dirs on sys.path)

from gaze_common import clip_span, gaze_at, load_gaze  # noqa: E402
from common import VIDEO_ROOT, build_choices, resolve_video_path  # noqa: E402

CONDS = ("oracle_full", "oracle_text", "oracle_mark", "oracle_saliency")
MAX_FRAMES = 64
TOL = 0                     # target clips = clips overlapping the annotated target_time (13 questions have none)

# EgoGazeVQA-Plus prompting, ported from test_tool/qwenvl_test/{test_gaze,test_mark,test_saliencemap}.py and
# gaze_trajectory.py. Pixel constants assume ~1408 px frames (Aria RGB, same as EgoLife); they are NOT rescaled.
MARK_RADIUS = 20                                    # filled red dot, as in the paper figure
SAL_SIGMA, SAL_RADIUS, SAL_WEIGHT = 20, 60, 20      # plot_gaze_trajectory defaults

PROMPTS = {
    "oracle_text": (
        "I provide you with a video and the normalized gaze coordinates for each corresponding frame. "
        "The frames are given below in order (Frame 1 is the first image).\n"
        "You need to follow these steps to answer the question:\n"
        "1. Observe the position of the annotated gaze points in each frame. The coordinate is from left to right "
        "for the x-axis and from top to bottom for the y-axis.\n"
        "2. Analyze the video while considering the gaze point information and then answer the question."),
    "oracle_mark": (
        "I provide you with a video that contains gaze information. For each frame, the gaze point is marked on "
        "the image with a red dot.\n"
        "You must follow these steps to answer the question:\n"
        "1. Focus on the objects marked by each red dot.\n"
        "2. Observe the video in chronological order according to the gaze sequence in step 1."),
    "oracle_saliency": (
        "For each video clip I provide a picture (the saliency map) followed by that clip's frames.\n"
        "You must follow these steps to answer the question:\n"
        "1. The picture before each clip's frames is the saliency grayscale map of the gaze trajectory from the "
        "first-person perspective, with gaze sequence from low brightness to high brightness.\n"
        "2. Remember the location and time sequence of gaze areas in that map.\n"
        "3. Observe the clip's frames according to the gaze sequence in step 2."),
}


def draw_mark(image, xn: float, yn: float):
    from PIL import ImageDraw

    w, h = image.size
    cx, cy = int(xn * w), int(yn * h)
    ImageDraw.Draw(image).ellipse([cx - MARK_RADIUS, cy - MARK_RADIUS, cx + MARK_RADIUS, cy + MARK_RADIUS],
                                  fill=(255, 0, 0))
    return image


def saliency_map(size: tuple[int, int], points: list[tuple[float, float]]):
    """Port of gaze_trajectory.plot_gaze_trajectory: one grayscale map per frame group, later gaze = brighter."""
    import cv2
    from PIL import Image

    width, height = size
    sal = np.zeros((height, width), dtype=np.float32)
    steps = np.linspace(10, 100, len(points))
    for i, (xn, yn) in enumerate(points):
        gx, gy = int(xn * width), int(yn * height)
        for dx in range(-SAL_RADIUS, SAL_RADIUS + 1, 10):
            for dy in range(-SAL_RADIUS, SAL_RADIUS + 1, 10):
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist <= SAL_RADIUS:
                    tx = max(0, min(gx + dx, width - 1))
                    ty = max(0, min(gy + dy, height - 1))
                    sal[ty, tx] += SAL_WEIGHT * np.exp(-dist ** 2 / (2 * SAL_SIGMA ** 2))
        gx, gy = max(0, min(gx, width - 1)), max(0, min(gy, height - 1))
        sal[gy, gx] += SAL_WEIGHT * steps[i]
    ksize = int(6 * SAL_SIGMA + 1)
    ksize += ksize % 2 == 0
    sal = cv2.GaussianBlur(sal, (ksize, ksize), 0)
    sal = cv2.normalize(sal, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Image.fromarray(sal).convert("RGB")


def proportional_cap(per_clip: list[list[int]], max_frames: int) -> list[list[int]]:
    """Mirror VisualMemory._frames_to_context_dict: scale each clip's list by max/total, keep >= 1."""
    total = sum(len(x) for x in per_clip)
    if total <= max_frames:
        return per_clip
    ratio = max_frames / total
    out = []
    for idx in per_clip:
        keep = max(1, int(len(idx) * ratio))
        pos = np.linspace(0, len(idx) - 1, keep, dtype=int).tolist()
        out.append([idx[p] for p in pos])
    return out


def load_questions(args):
    """(rows, ordered question ids that have >=1 target clip at TOL, {qid: sorted target keys}, clips, need_audio)."""
    pool = base.read_json(args.pool)
    rows = {str(r["ID"]): r for r in base.read_json(args.qa)}
    questions = {str(q["ID"]): q for q in pool["questions"]}
    mpath = args.out / "manifest.json"
    if not mpath.exists():                        # e.g. a fresh smoke --out; the main run's manifest has the same pool
        mpath = HERE / "run_data" / "manifest.json"
    excluded = set(base.read_json(mpath).get("excluded_embedding_keys", []))
    clips = {c["key"]: c for c in pool["clips"] if c["key"] not in excluded}   # same candidate set as run.py
    order, target_keys = [], {}
    for qid in (str(q["ID"]) for q in pool["questions"]):
        if qid not in rows:
            continue
        keys = sorted(base.targets(questions[qid], clips, TOL), key=lambda k: clip_span(clips[k])[0])
        if keys:
            order.append(qid)
            target_keys[qid] = keys
    if args.limit is not None:
        order = order[:args.limit]
    return rows, order, target_keys, clips


def oracle_slots(keys: list[str], clips: dict, video_root: Path) -> list[tuple]:
    """(key, VideoReader, frame_idx, fps) for the target clips: 1 fps, <= MAX_FRAMES, proportional per clip."""
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
        readers.append((key, vr, fps))
        per_clip.append(list(range(0, len(vr), max(1, int(fps)))))
    capped = proportional_cap(per_clip, MAX_FRAMES)
    return [(key, vr, idx, fps) for (key, vr, fps), idxs in zip(readers, capped) for idx in idxs]


def build_input(cond: str, slots: list[tuple], gaze: dict) -> tuple[list, list[str], str, str | None]:
    """Images, frame ids, image hash and the instruction `note` for one condition."""
    from PIL import Image

    frames = []                                   # (key, idx, PIL image, gaze xy normalized to the frame)
    for key, vr, idx, fps in slots:
        xy = gaze_at(gaze[key], idx / fps, tol=0.5)
        if xy is None:
            raise ValueError(f"Missing gaze at {key}:{idx / fps:.2f}s")
        frames.append((key, idx, Image.fromarray(vr[idx].asnumpy()).convert("RGB"), xy))
    images, ids, digest = [], [], hashlib.sha256()

    def add(tag: str, image) -> None:
        digest.update(f"{tag}|{cond}|{image.size}".encode())
        digest.update(image.tobytes())
        ids.append(tag)
        images.append(image)

    note = PROMPTS.get(cond)
    if cond == "oracle_full":
        for key, idx, image, _ in frames:
            add(f"{key}:{idx}", image)
    elif cond == "oracle_text":
        lines = [f"Frame {i + 1}: Gaze({round(xy[0], 3)}, {round(xy[1], 3)})" for i, (_, _, _, xy) in enumerate(frames)]
        note = "Gaze information for the relevant frames:\n" + "\n".join(lines) + "\n" + note
        for key, idx, image, _ in frames:
            add(f"{key}:{idx}", image)
    elif cond == "oracle_mark":
        for key, idx, image, xy in frames:
            add(f"{key}:{idx}", draw_mark(image, *xy))
    elif cond == "oracle_saliency":
        for key in dict.fromkeys(k for k, *_ in frames):
            group = [f for f in frames if f[0] == key]
            add(f"map:{key}", saliency_map(group[0][2].size, [f[3] for f in group]))
            for _, idx, image, _ in group:
                add(f"{key}:{idx}", image)
    else:
        raise ValueError(cond)
    return images, ids, digest.hexdigest(), note


def make_manifest(args, order: list[str], target_keys: dict) -> None:
    manifest = {"design": "oracle frames, tol=0 targets, 1 fps, <=64 frames, proportional per clip",
                "conditions": list(CONDS), "question_ids": order, "target_keys": {q: target_keys[q] for q in order},
                "qa_model": args.qa_model, "max_frames": MAX_FRAMES, "mark_radius": MARK_RADIUS,
                "saliency": [SAL_SIGMA, SAL_RADIUS, SAL_WEIGHT],
                "prompts": hashlib.sha256(repr(sorted(PROMPTS.items())).encode()).hexdigest(),
                "pool_sha256": base.sha256(args.pool), "qa_sha256": base.sha256(args.qa),
                "gaze_sha256": base.sha256(args.gaze)}
    path = args.out / "oracle_manifest.json"
    if path.exists() and base.read_json(path) != manifest:
        raise ValueError("Oracle manifest changed. Use a fresh --out directory")
    base.write_json(path, manifest)


def preflight(args, rows, order, target_keys, clips) -> None:
    gaze = load_gaze(str(args.gaze))
    n_clips = [len(target_keys[q]) for q in order]
    no_gaze = sorted({k for q in order for k in target_keys[q] if k not in gaze or not gaze[k].get("samples")})
    no_video = sorted({k for q in order for k in target_keys[q]
                       if not os.path.isfile(resolve_video_path(clips[k]["video_path"], str(args.video_root)))})
    need = {q: bool(rows[q]["need_audio"]) for q in order}
    no_text = [q for q in order if not base.text_path(args, q).exists()]
    print(f"questions with a target clip (tol={TOL}): {len(order)}  "
          f"(need_audio=False: {sum(not v for v in need.values())}, True: {sum(need.values())})")
    print(f"target clips per question: min {min(n_clips)}, median {int(np.median(n_clips))}, max {max(n_clips)}; "
          f"questions with >1 clip: {sum(c > 1 for c in n_clips)}")
    print(f"missing video: {len(no_video)}, missing gaze: {len(no_gaze)}, missing text context: {len(no_text)}")
    if no_video or no_gaze or no_text:
        raise ValueError("incomplete inputs; see counts above")


def run_qa(args, rows, order, target_keys, clips) -> None:
    from types import SimpleNamespace

    from worldmm.llm import LLMModel, PromptTemplateManager
    from eval_egolife import evaluate_prediction

    gaze = load_gaze(str(args.gaze))
    llm = LLMModel(model_name=args.qa_model, fps=1)
    # qa_prompt only needs the QA template; WorldMemory (and its ~12 GB of embedding models) is not loaded.
    wm = SimpleNamespace(prompt_template_manager=PromptTemplateManager(), qa_template_name="qa_egolife")
    for qid in order:
        row = rows[qid]
        context = base.text_record(args, qid, row)
        todo = [c for c in args.conds if not (args.out / "results" / c / f"{qid}.json").exists()]
        if not todo:
            continue
        try:
            slots = oracle_slots(target_keys[qid], clips, args.video_root)
        except Exception as exc:
            slots, slot_error = None, f"{type(exc).__name__}: {exc}"
        for cond in todo:
            result = {"ID": qid, "condition": cond, "question": row["question"], "answer": row["answer"],
                      "need_audio": bool(row["need_audio"]), "type": row.get("type"),
                      "target_keys": target_keys[qid], "text_hash": context["items_hash"]}
            try:
                if slots is None:
                    raise RuntimeError(slot_error)
                images, ids, image_hash, note = build_input(cond, slots, gaze)
                result.update({"frame_ids": ids, "num_images": len(images), "image_hash": image_hash})
                prompt = base.qa_prompt(wm, row["question"], build_choices(row), context["retrieved_items"],
                                        images, note=note)
                response = llm.generate(prompt, do_sample=False, max_new_tokens=32)
                result["response"] = response
                result["correct"] = evaluate_prediction(response, row["answer"], build_choices(row))
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
            base.write_json(args.out / "results" / cond / f"{qid}.json", result)
            print(f"oracle {qid}/{cond}: {result['status']} {result.get('correct')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("preflight", "qa"), required=True)
    ap.add_argument("--pool", type=Path, default=base.ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--gaze", type=Path, default=base.ROOT / "experiments/gaze_crop/gaze_points.json")
    ap.add_argument("--video-root", type=Path, default=Path(VIDEO_ROOT))
    ap.add_argument("--out", type=Path, default=HERE / "run_data",
                    help="holds text_context/ (read) and results/oracle_*/ (written)")
    ap.add_argument("--qa-model", default="qwen3vl-8b")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--conds", nargs="+", choices=CONDS, default=list(CONDS))
    args = ap.parse_args()
    rows, order, target_keys, clips = load_questions(args)
    make_manifest(args, order, target_keys)
    if args.stage == "preflight":
        preflight(args, rows, order, target_keys, clips)
    else:
        run_qa(args, rows, order, target_keys, clips)


if __name__ == "__main__":
    main()
