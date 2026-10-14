"""Oracle-frame QA with GazeQwen (setting A: retrieved text memory + oracle frames as ONE video).

Same 487 questions, same text_context/, same oracle frames (1 fps, <= 64, proportional per clip) as oracle.py, but
the QA model is Qwen2.5-VL-7B (GazeQwen's backbone) and the answer is the arg-max A/B/C/D logit at the first answer token.
The frames go in as a single video because GazeQwen's hooks inject into the video-token block (grid_thw[0]).

Conditions, in the order they are run (all share frames and V-JEPA features):
    gq_q25_full     plain Qwen2.5-VL, no LoRA, no hooks                          (phase 1, before LoRA is applied)
    gq_lora_only    + LoRA, hooks installed but never activated
    gq_inject_only  + injection with one fixation that is never active           (V-JEPA residual, no gaze)
    gq_gaze         + injection with the clip's real fixations (>= --min-dur s)
    gq_placebo      same fixation times/durations, centres random and >= 0.25 away from the real ones

Fixations: StreamGaze `extract_fixation_segments` (EgoLife 10 Hz adaptation in fixation_frame_selection) + step 1.5's
duration filter. Step 1.5's scene-histogram / black-screen filters are not applied. Several oracle clips are joined into
one video, so fixation times are mapped onto the video's own timeline (frame k of the video = k seconds).

    # one question, no model: check the prompt, token count and fixation mapping
    python oracle_gazeqwen.py --qids 1 --dry
    # one question on GPU 3 (measures time and peak memory), then everything
    cd ~/gazeqwen && PYTHONPATH=".:third_party/vjepa2" CUDA_VISIBLE_DEVICES=3 python .../oracle_gazeqwen.py --qids 1
    cd ~/gazeqwen && PYTHONPATH=".:third_party/vjepa2" CUDA_VISIBLE_DEVICES=3 nohup python .../oracle_gazeqwen.py > log 2>&1 &
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import oracle  # noqa: E402  (question selection, oracle frame slots; pulls in run.py as `base`)
from gq_smoke import GQ, fixations, random_placebo  # noqa: E402

base = oracle.base
build_choices = oracle.build_choices
CONDS = ("gq_q25_full", "gq_lora_only", "gq_inject_only", "gq_gaze", "gq_placebo")
INACTIVE = [[0.5, 0.5, -1000.0, 1.0]]            # one fixation that is never active at any frame time
FINAL = "\nPlease provide only the final answer from the choices given (e.g., A, B, C, or D)."


def qa_system_prompt() -> str:
    spec = importlib.util.spec_from_file_location("qa_egolife", base.ROOT / "src/worldmm/llm/templates/qa_egolife.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.prompt_template[0]["content"]


def load_video(slots, max_side):
    """Decode the oracle frames -> (N, 3, H, W) uint8 (N even), and (clip key, real time) of every frame."""
    from PIL import Image

    pil, meta = [], []
    for key, vr, i, fps in slots:                       # one slot per frame; vr.get_batch segfaults on these files
        pil.append(Image.fromarray(vr[i].asnumpy()).convert("RGB"))
        meta.append((key, i / fps))
    if len(pil) % 2:                                    # temporal patch size 2
        pil, meta = pil[:-1], meta[:-1]
    w, h = pil[0].size
    scale = max_side / max(w, h)
    size = (max(28, round(w * scale / 28) * 28), max(28, round(h * scale / 28) * 28))
    frames = torch.stack([torch.from_numpy(np.array(im.resize(size))).permute(2, 0, 1) for im in pil])
    return frames, meta


def video_scanpath(meta, gaze, min_dur):
    """Fixations of every clip in the video, on the video's timeline (frame k at k s) -> (N, 4) [x, y, mid, dur]."""
    rows = []
    for key in dict.fromkeys(k for k, _ in meta):
        pos = np.array([p for p, (k, _) in enumerate(meta) if k == key], dtype=np.float64)
        real = np.array([t for k, t in meta if k == key], dtype=np.float64)
        for x, y, mid, dur in fixations(gaze[key]["samples"], min_dur).tolist():
            s, e = np.interp([mid - dur / 2, mid + dur / 2], real, pos)   # clamps outside the kept frames
            if e - s > 1e-6:
                rows.append([x, y, (s + e) / 2, e - s])
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 4)


def build_inputs(proc, system, row, context, frames, dev):
    choices = build_choices(row)
    query = f"Query: {row['question']}\nChoices: " + " ".join(f"({k}) {v}" for k, v in sorted(choices.items()))
    head = query + "\n\nContext:\n" + "".join(x["content"] for x in context["retrieved_items"]) + "\n"
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": head}, {"type": "video"},
                                             {"type": "text", "text": FINAL}]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], videos=[frames], fps=1.0, return_tensors="pt")
    return inputs.to(dev) if dev is not None else inputs, sorted(choices)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=base.ROOT / "experiments/gaze_crop/pool_all.json")
    ap.add_argument("--qa", type=Path, default=base.ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json")
    ap.add_argument("--gaze", type=Path, default=base.ROOT / "experiments/gaze_crop/gaze_points.json")
    ap.add_argument("--video-root", type=Path, default=Path(oracle.VIDEO_ROOT))
    ap.add_argument("--out", type=Path, default=HERE / "run_data")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-side", type=int, default=448, help="video frame long side (multiple of 28 after rounding)")
    ap.add_argument("--min-dur", type=float, default=2.5, help="min fixation seconds (StreamGaze step 1.5: 2.5)")
    ap.add_argument("--qids", nargs="+", help="only these question ids")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--conds", nargs="+", choices=CONDS, default=list(CONDS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="eager", choices=("eager", "sdpa"),
                    help="eager is what the GazeQwen repo specifies; sdpa avoids the O(L^2) attention matrix (OOM on ~10k tokens)")
    ap.add_argument("--tag", default="", help="suffix for the results folders, e.g. _sdpa")
    ap.add_argument("--dry", action="store_true", help="build the inputs for the first question and exit (no model)")
    args = ap.parse_args()

    rows, order, target_keys, clips = oracle.load_questions(args)
    if args.qids:
        order = [q for q in order if q in set(args.qids)]
    gaze = oracle.load_gaze(str(args.gaze))
    system = qa_system_prompt()
    res_path = lambda c, q: args.out / "results" / f"oracle_{c}{args.tag}" / f"{q}.json"          # noqa: E731
    todo = {c: [q for q in order if not res_path(c, q).exists()] for c in args.conds}
    print({c: len(v) for c, v in todo.items()}, flush=True)

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)

    def prepare(qid):
        slots = oracle.oracle_slots(target_keys[qid], clips, args.video_root)
        frames, meta = load_video(slots, args.max_side)
        context = base.text_record(args, qid, rows[qid])
        return frames, meta, context

    if args.dry:
        qid = order[0]
        frames, meta, context = prepare(qid)
        inputs, letters = build_inputs(proc, system, rows[qid], context, frames, None)
        scan = video_scanpath(meta, gaze, args.min_dur)
        print(f"qid {qid}: frames {tuple(frames.shape)}, clips {len(target_keys[qid])}, input tokens {inputs['input_ids'].shape[1]}, "
              f"video_grid_thw {inputs['video_grid_thw'].tolist()}, second_per_grid_ts {inputs.get('second_per_grid_ts')}")
        print(f"fixations (>= {args.min_dur}s) on the video timeline:\n{scan.numpy().round(2)}")
        print(proc.decode(inputs["input_ids"][0][:60]), "...")
        return

    from transformers import Qwen2_5_VLForConditionalGeneration

    from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks
    from gazeqwen.lora import apply_lora, load_lora_state_dict
    from gazeqwen.model import GazeLens
    from gazeqwen.train import compute_frame_times, get_abcd_token_ids
    from gazeqwen.vjepa_features import VJEPAFeatureExtractor

    dev = torch.device("cuda")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation=args.attn).eval()
    abcd = get_abcd_token_ids(proc.tokenizer)
    # The bf16 lm_head quantises logits to ~0.06-0.125, which hides small gaze effects. Capture the last-position
    # hidden state going into lm_head and take the four answer logits in fp32 instead.
    captured = {}
    model.lm_head.register_forward_hook(lambda m, i, o: captured.__setitem__("h", i[0][0, -1].detach()))

    def run_cond(cond, qid, frames, meta, context, forward):
        row = rows[qid]
        result = {"ID": qid, "condition": cond, "question": row["question"], "answer": row["answer"],
                  "need_audio": bool(row["need_audio"]), "type": row.get("type"), "target_keys": target_keys[qid],
                  "text_hash": context["items_hash"], "n_frames": int(frames.shape[0]), "min_dur": args.min_dur, "attn": args.attn}
        try:
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            inputs, letters = build_inputs(proc, system, row, context, frames, dev)
            result["n_tokens"] = int(inputs["input_ids"].shape[1])
            forward(inputs)
            w = model.lm_head.weight[abcd_idx(letters)].float()
            logits = w @ captured["h"].float()
            probs = torch.softmax(logits.float(), 0).cpu().tolist()
            pred = letters[int(np.argmax(probs))]
            result.update({"probs": dict(zip(letters, probs)), "response": pred, "correct": pred == row["answer"],
                           "seconds": round(time.time() - t0, 2), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
                           "status": "ok"})
        except Exception as exc:                                      # one bad question must not stop the run
            result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            torch.cuda.empty_cache()
        base.write_json(res_path(cond, qid), result)
        print(f"{cond} {qid}: {result['status']} {result.get('correct')} {result.get('seconds')}s "
              f"{result.get('peak_gb')}GB tokens={result.get('n_tokens')}", flush=True)

    def abcd_idx(letters):
        return torch.tensor([abcd["ABCD".index(c)] for c in letters], device=dev)

    def plain_forward(inputs):
        with torch.no_grad():
            model(**inputs, logits_to_keep=1)

    # ---- phase 1: plain model (LoRA cannot be removed once applied, so this runs first) ----
    for qid in todo.get("gq_q25_full", []):
        try:
            frames, meta, context = prepare(qid)
        except Exception as exc:
            print(f"{qid}: prepare failed {type(exc).__name__}: {exc}", flush=True)
            continue
        run_cond("gq_q25_full", qid, frames, meta, context, plain_forward)

    rest = [c for c in args.conds if c != "gq_q25_full"]
    if not any(todo[c] for c in rest):
        return

    # ---- phase 2: GazeQwen ----
    ck = torch.load(GQ / "checkpoints/gazeqwen/best_model.pt", map_location="cpu", weights_only=False)
    f_theta = GazeLens(**ck["config"]).to(dev).eval()
    f_theta.load_state_dict(ck["state_dict"])
    apply_lora(model, rank=ck["lora_config"]["rank"], alpha=ck["lora_config"]["alpha"])
    print(f"LoRA tensors loaded: {load_lora_state_dict(model, ck['lora_state_dict'])}", flush=True)
    ctx = GazeLensContext(f_theta)
    register_gazelens_hooks(model, ctx)
    vjepa = VJEPAFeatureExtractor(device=dev, checkpoint_path=str(GQ / "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"))

    pending = sorted({q for c in rest for q in todo[c]}, key=order.index)
    for qid in pending:
        try:
            frames, meta, context = prepare(qid)
            scan = video_scanpath(meta, gaze, args.min_dur)
            placebo = random_placebo(scan, args.seed + int(qid))
        except Exception as exc:
            print(f"{qid}: prepare failed {type(exc).__name__}: {exc}", flush=True)
            continue
        feats_cache = {}

        def active_forward(scanpath):
            def forward(inputs):
                grid = inputs["video_grid_thw"][0]
                T, H, W = int(grid[0]), int(grid[1]) // 2, int(grid[2]) // 2
                if "f" not in feats_cache:
                    feats_cache["f"] = vjepa.extract_from_raw_frames(frames, T, H, W)
                with torch.no_grad(), ctx.active(scanpath.to(dev), compute_frame_times(0.0, T, 1.0),
                                                 backbone_features=feats_cache["f"]):
                    model(**inputs, logits_to_keep=1)
            return forward

        plan = {"gq_lora_only": plain_forward,                       # hooks exist but ctx is inactive -> no injection
                "gq_inject_only": active_forward(torch.tensor(INACTIVE)),
                "gq_gaze": active_forward(scan),
                "gq_placebo": active_forward(placebo)}
        for cond, forward in plan.items():
            if cond in args.conds and qid in todo[cond]:
                run_cond(cond, qid, frames, meta, context, forward)
                if cond == "gq_gaze":
                    r = res_path(cond, qid)
                    d = base.read_json(r)
                    d["n_fixations"] = int(len(scan))
                    base.write_json(r, d)


if __name__ == "__main__":
    main()
