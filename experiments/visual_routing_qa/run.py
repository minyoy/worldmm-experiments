"""Paired full/gaze QA experiment. Run from any directory; see README.md."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments" / "gaze_crop"),
                str(ROOT / "experiments" / "visual_bottleneck")]

from gaze_common import clip_span, crop_box, gaze_at, load_gaze, target_spans  # noqa: E402
from common import VIDEO_ROOT, build_choices, resolve_video_path  # noqa: E402

ARMS = {"full": "full", "gaze": "gazef_05"}


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def items_hash(items: list[dict]) -> str:
    data = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def load_inputs(args):
    pool = read_json(args.pool)
    rows = {str(r["ID"]): r for r in read_json(args.qa)}
    questions = {str(q["ID"]): q for q in pool["questions"]}
    order = [str(q["ID"]) for q in pool["questions"] if str(q["ID"]) in rows]
    if args.limit is not None:
        order = order[:args.limit]
    if not order:
        raise ValueError("No matching QA questions in pool")
    clips = {c["key"]: c for c in pool["clips"]}
    embeds = {}
    expected = set(clips)
    for arm, stem in ARMS.items():
        with np.load(args.emb_dir / f"{stem}.npz", allow_pickle=False) as z:
            keys = list(map(str, z["keys"].tolist()))
            matrix = np.asarray(z["emb"], dtype=np.float32)
        if len(keys) != len(set(keys)) or set(keys) - expected:
            raise ValueError(f"{arm} embedding keys contain duplicates or keys outside the pool")
        if matrix.ndim != 2 or len(matrix) != len(keys) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid {arm} embedding matrix")
        norms = np.linalg.norm(matrix, axis=1)
        if (norms <= 0).any():
            raise ValueError(f"Zero vectors in {arm} embeddings")
        embeds[arm] = {k: v / n for k, v, n in zip(keys, matrix, norms)}
    if next(iter(embeds["full"].values())).shape != next(iter(embeds["gaze"].values())).shape:
        raise ValueError("Full/gaze embedding dimensions differ")
    common = set(embeds["full"]) & set(embeds["gaze"])
    if len(common) < 3:
        raise ValueError("Fewer than three clips have both full and gaze embeddings")
    omitted = sorted(expected - common)
    if omitted:
        print(f"Using {len(common)}/{len(expected)} clips in both indexes; "
              f"excluded missing-embedding clips: {omitted}", flush=True)
    clips = {key: clip for key, clip in clips.items() if key in common}
    embeds = {arm: {key: vec for key, vec in values.items() if key in common}
              for arm, values in embeds.items()}
    return pool, rows, questions, order, clips, embeds


def targets(q: dict, clips: dict, tolerance: float) -> set[str]:
    qt = q["query_time"]
    fallback_date = qt.get("date") if isinstance(qt, dict) else f"DAY{str(qt)[0]}"
    spans = target_spans(q.get("target_time"), fallback_date)
    if not spans:
        return set(q.get("target_keys") or [])
    return {key for key, clip in clips.items()
            if any(a - tolerance <= clip_span(clip)[1] and
                   clip_span(clip)[0] <= b + tolerance for a, b in spans)}


def rank_clips(query_vec: np.ndarray, embeddings: dict, clips: dict, query_time: int) -> list[dict]:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    n = np.linalg.norm(q)
    if n <= 0 or not np.isfinite(q).all():
        raise ValueError("Invalid query embedding")
    q /= n
    scores = []
    for key, clip in clips.items():
        if int(clip["ts_end"]) <= query_time:
            scores.append((key, float(np.dot(q, embeddings[key]))))
    scores.sort(key=lambda x: (-x[1], x[0]))
    return [{"key": key, "score": score} for key, score in scores[:3]]


def frame_payload(top3: list[dict], clips: dict, gaze: dict, arm: str,
                  video_root: Path) -> tuple[list, list[str], str]:
    """WorldMM 1 fps frames, globally capped at 64; crop after frame selection."""
    from decord import VideoReader, cpu
    from PIL import Image

    slots = []
    for hit in top3:
        key = hit["key"]
        path = resolve_video_path(clips[key]["video_path"], str(video_root))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        vr = VideoReader(path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        if fps <= 0 or len(vr) < 2:
            raise ValueError(f"Bad video duration: {path}")
        interval = max(1, int(fps))
        indices = list(range(0, len(vr), interval))
        if not indices:
            raise ValueError(f"No frames: {path}")
        if arm == "gaze" and (key not in gaze or not gaze[key].get("samples")):
            raise ValueError(f"No real gaze samples for {key}")
        for idx in indices:
            slots.append((key, vr, idx, fps))
    if len(slots) > 64:
        slots = [slots[i] for i in np.linspace(0, len(slots) - 1, 64, dtype=int)]
    images, metadata = [], []
    digest = hashlib.sha256()
    for key, vr, idx, fps in slots:
        image = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
        if arm == "gaze":
            xy = gaze_at(gaze[key], idx / fps, tol=0.5)
            if xy is None:
                raise ValueError(f"Missing gaze at {key}:{idx / fps:.2f}s")
            image = image.crop(crop_box(*image.size, xy[0], xy[1], 0.5))
        digest.update(key.encode())
        digest.update(str(idx).encode())
        digest.update(str(image.size).encode())
        digest.update(image.tobytes())
        metadata.append(f"{key}:{idx}")
        images.append(image)
    if not images or len({s.split(":")[0] for s in metadata}) != 3:
        raise ValueError("Top-3 must supply image frames from all three clips")
    return images, metadata, digest.hexdigest()


def text_path(args, qid: str) -> Path:
    return args.out / "text_context" / f"{qid}.json"


def text_record(args, qid: str, row: dict) -> dict:
    path = text_path(args, qid)
    if not path.exists():
        raise FileNotFoundError(f"Missing text context {path}; run --stage text")
    data = read_json(path)
    if data.get("question") != row["question"] or data.get("choices") != build_choices(row):
        raise ValueError(f"Text context mismatch for {qid}")
    if data.get("items_hash") != items_hash(data["retrieved_items"]):
        raise ValueError(f"Text context hash mismatch for {qid}")
    return data


def prepare_text(args, rows: dict, order: list[str]) -> None:
    """Generate text-only WorldMemory contexts in query-time order."""
    from worldmm.embedding import EmbeddingModel
    from worldmm.llm import LLMModel, PromptTemplateManager
    from worldmm.memory import WorldMemory
    from eval_egolife import disable_visual_memory, MEMORY_REASONING_TEXT_ONLY

    # Only this experiment's graph workdirs are reset; existing global caches are untouched.
    for gran in ("30sec", "3min", "10min", "1h"):
        source = args.episodic_source / gran
        target = args.out / "episodic_cache" / gran
        target.mkdir(parents=True, exist_ok=True)
        name = "openie_results_ner_gpt-5-mini.json"
        if not (source / name).exists():
            raise FileNotFoundError(f"Missing precomputed OpenIE cache: {source / name}")
        if not (target / name).exists():
            shutil.copy2(source / name, target / name)
        for child in target.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    llm = LLMModel(model_name=args.qa_model, fps=1)
    wm = WorldMemory(embedding_model=EmbeddingModel(), retriever_llm_model=llm,
                     respond_llm_model=llm, prompt_template_manager=PromptTemplateManager(),
                     episodic_granularities=["30sec", "3min", "10min", "1h"],
                     episodic_cache_root=str(args.out / "episodic_cache"))
    wm.set_retrieval_top_k(episodic=3, semantic=10, visual=3)
    disable_visual_memory(wm)
    # Keep template visible in manifest, independent of the imported helper implementation.
    (args.out / "text_reasoning_prompt.txt").write_text(MEMORY_REASONING_TEXT_ONLY, encoding="utf-8")
    caps = args.data / "EgoLifeCap" / args.subject
    wm.load_episodic_captions(caption_files={g: str(caps / f"{args.subject}_{g}.json")
                                            for g in ("30sec", "3min", "10min", "1h")})
    semantic = args.metadata / "semantic_memory" / args.subject / "semantic_consolidation_results_gpt-5-mini.json"
    wm.load_semantic_triples(data=read_json(semantic))
    for qid in sorted(order, key=lambda i: int(rows[i]["query_time"]["date"][-1] +
                                                  str(rows[i]["query_time"]["time"]).zfill(8))):
        row = rows[qid]
        path = text_path(args, qid)
        if path.exists():
            text_record(args, qid, row)
            continue
        qt = int(row["query_time"]["date"][-1] + str(row["query_time"]["time"]).zfill(8))
        answer = wm.answer(query=row["question"], choices=build_choices(row), until_time=qt)
        items = [{"memory_type": x.memory_type, "content": x.content,
                  "query": x.query, "round_num": x.round_num}
                 for x in answer.retrieved_items if x.memory_type in ("episodic", "semantic")]
        write_json(path, {"ID": qid, "question": row["question"], "choices": build_choices(row),
                          "retrieved_items": items, "items_hash": items_hash(items),
                          "round_history": answer.round_history, "num_rounds": answer.num_rounds,
                          "model": args.qa_model})
        print(f"text {qid}: {len(items)} items", flush=True)


DUAL = ("dual_gaze", "dual_retrieve")   # D: gaze top-3 shown full+gaze;  E: full top-3 (full) + gaze top-3 (gaze)
DUAL_SLOTS = 32                          # frame slots per view, so every condition shows at most 64 images


def _clip_slots(top3: list[dict], clips: dict, gaze: dict, need_gaze: bool, video_root: Path, cap: int) -> list:
    """1 fps frame slots of the Top-3 clips, evenly thinned to `cap` (same walk as frame_payload)."""
    from decord import VideoReader, cpu

    slots = []
    for hit in top3:
        key = hit["key"]
        path = resolve_video_path(clips[key]["video_path"], str(video_root))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        vr = VideoReader(path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        if fps <= 0 or len(vr) < 2:
            raise ValueError(f"Bad video duration: {path}")
        if need_gaze and (key not in gaze or not gaze[key].get("samples")):
            raise ValueError(f"No real gaze samples for {key}")
        slots.extend((key, vr, idx, fps) for idx in range(0, len(vr), max(1, int(fps))))
    if len(slots) > cap:
        slots = [slots[i] for i in np.linspace(0, len(slots) - 1, cap, dtype=int)]
    if len({k for k, *_ in slots}) != 3:
        raise ValueError("Top-3 must supply image frames from all three clips")
    return slots


def _slot_image(slot: tuple, view: str, gaze: dict):
    from PIL import Image

    key, vr, idx, fps = slot
    image = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
    if view == "gaze":
        xy = gaze_at(gaze[key], idx / fps, tol=0.5)
        if xy is None:
            raise ValueError(f"Missing gaze at {key}:{idx / fps:.2f}s")
        image = image.crop(crop_box(*image.size, xy[0], xy[1], 0.5))
    return image


def dual_payload(cond: str, top3: dict, clips: dict, gaze: dict, video_root: Path) -> tuple[list, list[str], str]:
    """Images for the two dual-view conditions; `top3` = {"full": [...], "gaze": [...]} from A/B.

    dual_gaze:     the GAZE Top-3 clips; each of up to DUAL_SLOTS timestamps appears as full frame, then gaze crop.
    dual_retrieve: the FULL Top-3 clips as full frames (up to DUAL_SLOTS), then the GAZE Top-3 clips as gaze
                   crops (up to DUAL_SLOTS). The two Top-3 sets may overlap; each view keeps its own frames.
    """
    plan = []                                    # (slot, view) in prompt order
    if cond == "dual_gaze":
        for slot in _clip_slots(top3["gaze"], clips, gaze, True, video_root, DUAL_SLOTS):
            plan += [(slot, "full"), (slot, "gaze")]
    elif cond == "dual_retrieve":
        plan += [(s, "full") for s in _clip_slots(top3["full"], clips, gaze, False, video_root, DUAL_SLOTS)]
        plan += [(s, "gaze") for s in _clip_slots(top3["gaze"], clips, gaze, True, video_root, DUAL_SLOTS)]
    else:
        raise ValueError(cond)
    images, metadata, digest = [], [], hashlib.sha256()
    for slot, view in plan:
        image = _slot_image(slot, view, gaze)
        digest.update(f"{slot[0]}|{slot[2]}|{view}|{image.size}".encode())
        digest.update(image.tobytes())
        metadata.append(f"{view}:{slot[0]}:{slot[2]}")
        images.append(image)
    return images, metadata, digest.hexdigest()


def qa_prompt(wm, question: str, choices: dict, items: list[dict], images: list,
              note: str | None = None):
    import copy
    full_query = f"Query: {question}"
    if choices:
        full_query += "\nChoices: " + " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
    content = [{"type": "text", "text": full_query + "\n\nContext:\n"}]
    content.extend({"type": "text", "text": x["content"]} for x in items)
    if note:
        content.append({"type": "text", "text": "\n" + note + "\n"})
    content.extend({"type": "image", "image": img} for img in images)
    if choices:
        content.append({"type": "text", "text":
                        "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."})
    messages = copy.deepcopy(wm.prompt_template_manager.render(wm.qa_template_name))
    messages.append({"role": "user", "content": content})
    return messages


def load_result(path: Path) -> dict:
    return read_json(path) if path.exists() else {}


def make_manifest(args, pool: dict, order: list[str], clips: dict) -> None:
    manifest = {"subject": args.subject, "question_ids": order, "qa_model": args.qa_model,
                "router_model": args.router_model, "top_k": 3, "fps": 1, "max_frames": 64,
                "gaze_ratio": 0.5, "pool_sha256": sha256(args.pool),
                "qa_sha256": sha256(args.qa), "gaze_sha256": sha256(args.gaze),
                "embeddings": {arm: sha256(args.emb_dir / f"{stem}.npz") for arm, stem in ARMS.items()},
                "n_pool_clips": len(pool["clips"]), "n_candidate_clips": len(clips),
                "excluded_embedding_keys": sorted({c["key"] for c in pool["clips"]} - set(clips))}
    path = args.out / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Run manifest changed. Use a fresh --out directory")
    write_json(path, manifest)


def run_qa(args, rows: dict, questions: dict, order: list[str], clips: dict, embeds: dict) -> None:
    from worldmm.embedding import EmbeddingModel
    from worldmm.llm import LLMModel, PromptTemplateManager
    from worldmm.memory import WorldMemory
    from eval_egolife import evaluate_prediction

    for qid in order:
        text_record(args, qid, rows[qid])
    gaze = load_gaze(str(args.gaze))
    if not gaze:
        raise ValueError(f"No gaze data in {args.gaze}")
    embedder = EmbeddingModel()
    llm = LLMModel(model_name=args.qa_model, fps=1)
    wm = WorldMemory(embedding_model=embedder, retriever_llm_model=llm,
                     respond_llm_model=llm, prompt_template_manager=PromptTemplateManager())
    for qid in order:
        row, q = rows[qid], questions[qid]
        context = text_record(args, qid, row)
        qt = int(q["query_time"])
        query_vec = None
        if any(not (args.out / "results" / arm / f"{qid}.json").exists() for arm in ARMS):
            query_vec = np.asarray(embedder.encode_vis_query(row["question"]))
        for arm in ARMS:
            path = args.out / "results" / arm / f"{qid}.json"
            if path.exists():
                continue
            top3 = rank_clips(query_vec, embeds[arm], clips, qt)
            if len(top3) != 3:
                raise ValueError(f"{qid}/{arm}: fewer than three visible clips")
            result = {"ID": qid, "condition": arm, "question": row["question"],
                      "answer": row["answer"], "embedding": ARMS[arm], "top3": top3,
                      "text_hash": context["items_hash"],
                      "hit_at_3": {str(tol): bool(set(x["key"] for x in top3) &
                                                   targets(q, clips, tol)) for tol in (0, 60)},
                      "target_count": {str(tol): len(targets(q, clips, tol)) for tol in (0, 60)}}
            try:
                images, frame_ids, image_hash = frame_payload(top3, clips, gaze, arm, args.video_root)
                result.update({"frame_ids": frame_ids, "num_frames": len(images),
                               "image_hash": image_hash, "image_type": arm})
                prompt = qa_prompt(wm, row["question"], build_choices(row),
                                   context["retrieved_items"], images)
                response = llm.generate(prompt, do_sample=False, max_new_tokens=32)
                result["response"] = response
                result["correct"] = evaluate_prediction(response, row["answer"], build_choices(row))
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
            write_json(path, result)
            print(f"qa {qid}/{arm}: {result['status']} {result.get('correct')}", flush=True)

        # D/E reuse the retrieval already stored for A/B (text memory identical, visual forced, same QA call)
        ab = {arm: load_result(args.out / "results" / arm / f"{qid}.json") for arm in ARMS}
        for cond in DUAL:
            path = args.out / "results" / cond / f"{qid}.json"
            if path.exists():
                continue
            if any(ab[arm].get("status") != "ok" for arm in ARMS):
                print(f"qa {qid}/{cond}: skipped, A/B not ok", flush=True)
                continue
            top3 = {arm: ab[arm]["top3"] for arm in ARMS}
            union = {tol: bool(ab["full"]["hit_at_3"][tol] or ab["gaze"]["hit_at_3"][tol]) for tol in ("0", "60")}
            result = {"ID": qid, "condition": cond, "question": row["question"], "answer": row["answer"],
                      "embedding": "full+gazef_05", "top3": top3["gaze"] if cond == "dual_gaze" else top3["full"] + top3["gaze"],
                      "top3_full": top3["full"], "top3_gaze": top3["gaze"],
                      "text_hash": context["items_hash"],
                      # D shows the gaze Top-3; E shows the union of both Top-3 sets (6 clips, so its hit is A OR B)
                      "hit_at_3": ab["gaze"]["hit_at_3"] if cond == "dual_gaze" else union,
                      "target_count": ab["full"]["target_count"]}
            try:
                images, frame_ids, image_hash = dual_payload(cond, top3, clips, gaze, args.video_root)
                result.update({"frame_ids": frame_ids, "num_frames": len(images),
                               "image_hash": image_hash, "image_type": cond})
                prompt = qa_prompt(wm, row["question"], build_choices(row),
                                   context["retrieved_items"], images)
                response = llm.generate(prompt, do_sample=False, max_new_tokens=32)
                result["response"] = response
                result["correct"] = evaluate_prediction(response, row["answer"], build_choices(row))
                result["status"] = "ok"
            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
            write_json(path, result)
            print(f"qa {qid}/{cond}: {result['status']} {result.get('correct')}", flush=True)


ROUTE_METHOD = "evidence_v1"
ROUTE_FRACS = (0.25, 0.5, 0.75)     # same relative timestamps in every retrieved clip
ROUTE_SIZE = 448                    # longest side of each frame shown to the router, px
ROUTE_SYSTEM = (
    "You are shown a question and two candidate sets of video frames, Set 1 and Set 2. Each set holds frames "
    "from three retrieved video clips of the same person's first-person recording. Decide which set contains "
    "better visual evidence for answering the question: the frames that show what the question asks about "
    "(the relevant people, objects, or action). The sets are labelled only by number; judge by what the frames "
    "show. Reply with exactly 1 or 2.")


def evidence_images(top3: list[dict], clips: dict, gaze: dict, arm: str, video_root: Path) -> list[list]:
    """ROUTE_FRACS frames from each retrieved clip, cropped on the gaze for the gaze arm."""
    from decord import VideoReader, cpu
    from PIL import Image

    per_clip = []
    for hit in top3:
        key = hit["key"]
        path = resolve_video_path(clips[key]["video_path"], str(video_root))
        vr = VideoReader(path, ctx=cpu(0))
        fps = float(vr.get_avg_fps())
        if fps <= 0 or len(vr) < 2:
            raise ValueError(f"Bad video duration: {path}")
        if arm == "gaze" and (key not in gaze or not gaze[key].get("samples")):
            raise ValueError(f"No real gaze samples for {key}")
        frames = []
        for frac in ROUTE_FRACS:
            idx = int(round(frac * (len(vr) - 1)))
            image = Image.fromarray(vr[idx].asnumpy()).convert("RGB")
            if arm == "gaze":
                xy = gaze_at(gaze[key], idx / fps, tol=0.5)
                if xy is None:
                    raise ValueError(f"Missing gaze at {key}:{idx / fps:.2f}s")
                image = image.crop(crop_box(*image.size, xy[0], xy[1], 0.5))
            image.thumbnail((ROUTE_SIZE, ROUTE_SIZE))
            frames.append(image)
        per_clip.append(frames)
    return per_clip


def route_prompt(question: str, first: list[list], second: list[list]) -> list[dict]:
    content = [{"type": "text", "text": f"Question: {question}\n"}]
    for number, per_clip in ((1, first), (2, second)):
        content.append({"type": "text", "text": f"\nSet {number}:\n"})
        for i, frames in enumerate(per_clip, 1):
            content.append({"type": "text", "text": f"Clip {i}: "})
            content.extend({"type": "image", "image": img} for img in frames)
            content.append({"type": "text", "text": "\n"})
    content.append({"type": "text", "text": "\nWhich set contains better visual evidence? Reply 1 or 2."})
    return [{"role": "system", "content": ROUTE_SYSTEM}, {"role": "user", "content": content}]


def parse_choice(raw: str) -> int | None:
    token = str(raw).strip().strip(".").strip()
    return int(token) if token in ("1", "2") else None


def route(args, rows: dict, order: list[str], clips: dict) -> None:
    """Evidence-aware routing: show the router the frames each arm retrieved, in both orders.

    A decision counts only when both orders pick the same arm; otherwise (disagreement, or a reply that is
    not 1/2) the router abstains and the question falls back to FULL. Labels are numbers, so the arm
    names carry no prior; which arm is Set 1 is swapped between the two queries.
    """
    from worldmm.llm import LLMModel

    missing = [(qid, arm) for qid in order for arm in ARMS
               if load_result(args.out / "results" / arm / f"{qid}.json").get("status") != "ok"]
    if missing:
        raise ValueError(f"A/B QA must finish first; missing/error examples: {missing[:5]}")
    stale = [p.name for p in (args.out / "routing").glob("*.json")
             if read_json(p).get("method") != ROUTE_METHOD] if (args.out / "routing").exists() else []
    if stale:
        raise ValueError(f"{len(stale)} routing files in {args.out / 'routing'} come from a different routing "
                         f"method; move routing/ and results/routing/ aside first (e.g. to routing_v0/)")
    gaze = load_gaze(str(args.gaze))
    llm = LLMModel(model_name=args.router_model, fps=1)
    for qid in order:
        row = rows[qid]
        route_path = args.out / "routing" / f"{qid}.json"
        if route_path.exists():
            cached = read_json(route_path)
            if cached.get("question") != row["question"]:
                raise ValueError(f"Routing question changed: {qid}")
            decision, agree = cached["decision"], cached["agree"]
        else:
            found = {arm: read_json(args.out / "results" / arm / f"{qid}.json") for arm in ARMS}
            imgs = {arm: evidence_images(found[arm]["top3"], clips, gaze, arm, args.video_root) for arm in ARMS}
            votes = []
            for first, second in (("full", "gaze"), ("gaze", "full")):
                raw = str(llm.generate(route_prompt(row["question"], imgs[first], imgs[second]),
                                       do_sample=False, max_new_tokens=8)).strip()
                n = parse_choice(raw)
                votes.append({"set1": first, "set2": second, "raw": raw,
                              "picked": None if n is None else (first, second)[n - 1]})
            picks = [v["picked"] for v in votes]
            agree = picks[0] is not None and picks[0] == picks[1]
            decision = picks[0] if agree else "full"
            write_json(route_path, {"ID": qid, "question": row["question"], "method": ROUTE_METHOD,
                                    "votes": votes, "agree": agree, "decision": decision,
                                    "fracs": list(ROUTE_FRACS)})
        if decision not in ARMS:
            raise ValueError(f"Bad cached router decision for {qid}: {decision}")
        selected = read_json(args.out / "results" / decision / f"{qid}.json")
        result = dict(selected, condition="routing", selected_arm=decision, router_agree=agree,
                      source_path=f"results/{decision}/{qid}.json")
        assert result["text_hash"] == selected["text_hash"]
        write_json(args.out / "results" / "routing" / f"{qid}.json", result)
        print(f"routing {qid}: {decision} agree={agree}", flush=True)


def preflight(args, pool: dict, rows: dict, questions: dict, order: list[str], clips: dict) -> None:
    gaze = load_gaze(str(args.gaze))
    no_video = [k for k, c in clips.items()
                if not os.path.isfile(resolve_video_path(c["video_path"], str(args.video_root)))]
    no_gaze = [k for k in clips if k not in gaze or not gaze[k].get("samples")]
    no_target = {str(t): [qid for qid in order if not targets(questions[qid], clips, t)]
                 for t in (0, 60)}
    report = {"questions": len(order), "pool_clips": len(pool["clips"]),
              "candidate_clips": len(clips),
              "excluded_embedding_keys": sorted({c["key"] for c in pool["clips"]} - set(clips)),
              "missing_video": no_video,
              "missing_gaze": no_gaze, "no_target": no_target,
              "missing_text": [qid for qid in order if not text_path(args, qid).exists()]}
    write_json(args.out / "preflight.json", report)
    summary = {k: len(v) if isinstance(v, list) else v for k, v in report.items()
               if k != "no_target"}
    summary["no_target"] = {tol: len(ids) for tol, ids in no_target.items()}
    print(json.dumps(summary, indent=2))
    if no_video or no_gaze:
        raise ValueError("Pool video/gaze coverage incomplete; see preflight.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("preflight", "text", "qa", "route", "all"), required=True)
    ap.add_argument("--subject", default="A1_JAKE")
    ap.add_argument("--pool", type=Path, default=ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--emb-dir", type=Path, default=ROOT / "experiments/gaze_crop/emb")
    ap.add_argument("--gaze", type=Path, default=ROOT / "experiments/gaze_crop/gaze_points.json")
    ap.add_argument("--data", type=Path, default=ROOT / "data/EgoLife")
    ap.add_argument("--metadata", type=Path, default=ROOT / "output/metadata")
    ap.add_argument("--episodic-source", type=Path, default=ROOT / ".cache/episodic_memory")
    ap.add_argument("--video-root", type=Path, default=Path(VIDEO_ROOT))
    ap.add_argument("--out", type=Path, default=HERE / "run_data")
    ap.add_argument("--qa-model", default="qwen3vl-8b")
    ap.add_argument("--router-model", default="qwen3vl-8b")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    pool, rows, questions, order, clips, embeds = load_inputs(args)
    make_manifest(args, pool, order, clips)
    if args.stage in ("preflight", "all"):
        preflight(args, pool, rows, questions, order, clips)
    if args.stage in ("text", "all"):
        prepare_text(args, rows, order)
    if args.stage in ("qa", "all"):
        run_qa(args, rows, questions, order, clips, embeds)
    if args.stage in ("route", "all"):
        route(args, rows, order, clips)


if __name__ == "__main__":
    main()
