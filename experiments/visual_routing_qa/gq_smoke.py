"""GazeQwen smoke test on one EgoLife oracle clip (no WorldMM text memory, no QA answer scoring).

Checks that (1) the hooks fire on the installed transformers, (2) V-JEPA features extract, and
(3) which part of the GazeQwen delta comes from LoRA, from the V-JEPA injection, and from the gaze itself.

Conditions (one forward each, A/B/C/D probabilities):
    q25_full   plain Qwen2.5-VL, no LoRA, no hooks
    lora_only  LoRA applied, hooks installed but never activated (no injection)
    nogaze     LoRA + injection with an empty scanpath (GazeLens returns a zero residual: must equal lora_only)
    inject_only LoRA + injection with one fixation that is never active at any frame time (V-JEPA residual, no gaze PE)
    gaze       LoRA + injection with the clip's real fixations
    placebo    same fixation times/durations, centres drawn at random far from the real ones

Fixations come from the StreamGaze detector (`extract_fixation_segments`, as adapted to 10 Hz EgoLife gaze in
experiments/fixation_frame_selection) followed by StreamGaze step 1.5's minimum-duration filter. The scene-histogram
and black-screen filters of step 1.5 are NOT applied. gaze/placebo are repeated for each --min-durs value.

    cd ~/gazeqwen && PYTHONPATH=".:third_party/vjepa2" CUDA_VISIBLE_DEVICES=3 \
        python ~/WorldMM/experiments/visual_routing_qa/gq_smoke.py --qid 1
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("WORLDMM_ROOT", Path.home() / "WorldMM"))
HERE = ROOT / "experiments/visual_routing_qa"
VIDEO_ROOT = os.environ.get("WORLDMM_VIDEO_ROOT", "/datasets/EgoLife")
GQ = Path.home() / "gazeqwen"
sys.path.insert(0, str(ROOT / "experiments/fixation_frame_selection/streamgaze"))

# fixation_frame_selection/README.md 2.2: radius 0.05, min_dur 0.3 s, gap 0.2 s, dropout 0.4 s
DETECTOR = dict(radius_thresh=0.05, duration_thresh=0.3, gap_thresh=0.2, dropout_thresh=0.4)


def fixations(samples, min_dur, detector=None):
    """StreamGaze I-DT on [t, x, y] samples, keep >= min_dur -> (N, 4) [x, y, t_mid, dur]."""
    import pandas as pd
    from preprocess.gaze_processing import extract_fixation_segments

    rows = sorted((float(s[0]), float(s[1]), float(s[2])) for s in samples if float(s[0]) >= 0)
    if len(rows) < 2:
        return torch.zeros(0, 4)
    df = pd.DataFrame(rows, columns=["time_seconds", "px", "py"])
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):      # the detector prints per call
        fx = extract_fixation_segments(df, **(detector or DETECTOR))
    out = [[f["center_x"], f["center_y"], (f["start_time"] + f["end_time"]) / 2, f["duration"]]
           for f in fx if f["duration"] >= min_dur]
    return torch.tensor(out, dtype=torch.float32).reshape(-1, 4)


def random_placebo(scan, seed, min_dist=0.25, lo=0.1, hi=0.9):
    """Keep each fixation's time and duration, move its centre to a random point >= min_dist from the real one."""
    rng = np.random.default_rng(seed)
    out = scan.clone()
    for i in range(len(out)):
        while True:
            x, y = rng.uniform(lo, hi, 2)
            if np.hypot(x - float(scan[i, 0]), y - float(scan[i, 1])) >= min_dist:
                break
        out[i, 0], out[i, 1] = float(x), float(y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="1")
    ap.add_argument("--n-frames", type=int, default=30, help="1 fps frames; 30 covers a whole clip so every fixation can be active")
    ap.add_argument("--size", type=int, nargs=2, default=(392, 224), help="W H, multiples of 28")
    ap.add_argument("--min-durs", default="0.3,2.0,2.5",
                    help="fixation minimum durations to try (s). GazeQwen's demo CSV: min 2.04, median 2.54; "
                         "StreamGaze step 1.5 filters at 2.5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = ap.parse_args()

    from decord import VideoReader, cpu
    from PIL import Image

    man = json.load(open(HERE / "run_data/oracle_manifest.json"))
    pool = json.load(open(ROOT / "experiments/gaze_crop/pool_all.json"))
    gaze = json.load(open(ROOT / "experiments/gaze_crop/gaze_points.json"))
    gaze = gaze.get("clips", gaze)
    qa = {str(r["ID"]): r for r in json.load(open(ROOT / "data/EgoLife/EgoLifeQA/EgoLifeQA_A1_JAKE.json"))}
    clips = {c["key"]: c for c in pool["clips"]}
    row, key = qa[args.qid], man["target_keys"][args.qid][0]
    path = clips[key]["video_path"].replace("data/EgoLife/", VIDEO_ROOT + "/")
    vr = VideoReader(path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())
    idx = list(range(0, len(vr), max(1, int(fps))))[:args.n_frames]
    if len(idx) % 2:
        idx = idx[:-1]                      # temporal patch size is 2
    frames = torch.stack([torch.from_numpy(np.array(Image.fromarray(vr[i].asnumpy()).convert("RGB").resize(tuple(args.size)))).permute(2, 0, 1)
                          for i in idx])   # (N, 3, H, W) uint8
    min_durs = [float(v) for v in args.min_durs.split(",")]
    scans = {d: fixations(gaze[key]["samples"], d) for d in min_durs}
    print(f"clip {key}: {len(idx)} frames @1fps, {len(gaze[key]['samples'])} gaze samples")
    for d, s in scans.items():
        print(f"  min_dur>={d}s: {len(s)} fixations", s.numpy().round(2).tolist() if len(s) else "")

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from gazeqwen.hooks import GazeLensContext, register_gazelens_hooks
    from gazeqwen.lora import apply_lora, load_lora_state_dict
    from gazeqwen.model import GazeLens
    from gazeqwen.train import build_mcqa_prompt, compute_frame_times, get_abcd_token_ids
    from gazeqwen.vjepa_features import VJEPAFeatureExtractor

    dev = torch.device("cuda")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="eager").eval()
    proc = AutoProcessor.from_pretrained(args.model)

    opts = [f"{c}. {row['choice_' + c.lower()]}" for c in "ABCD"]
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": build_mcqa_prompt(row["question"], opts)}]}],
        tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], videos=[frames], return_tensors="pt").to(dev)
    grid = inputs["video_grid_thw"][0]
    T, H, W = int(grid[0]), int(grid[1]) // 2, int(grid[2]) // 2
    times = compute_frame_times(0.0, T, 1.0)
    abcd = torch.tensor(get_abcd_token_ids(proc.tokenizer), device=dev)
    print(f"T={T} llm_grid={H}x{W} visual tokens={T * H * W}")
    results = {}

    def probs(label, ctx=None, scanpath=None, feats=None):
        calls = {"n": 0}
        if ctx is not None and ctx.f_theta is not None:
            orig = ctx.f_theta.forward

            def counting(*a, **k):
                calls["n"] += 1
                return orig(*a, **k)

            ctx.f_theta.forward = counting
        try:
            with torch.no_grad():
                if scanpath is None:
                    logits = model(**inputs).logits[0, -1, abcd].float()
                else:
                    with ctx.active(scanpath.to(dev), times, backbone_features=feats):
                        logits = model(**inputs).logits[0, -1, abcd].float()
        finally:
            if ctx is not None and ctx.f_theta is not None:
                ctx.f_theta.forward = orig
        p = torch.softmax(logits, 0).cpu().numpy()
        results[label] = logits
        print(f"{label:22s} f_theta calls={calls['n']:3d} probs={p.round(4)}", flush=True)

    # --- baselines that must run before LoRA is applied ---
    probs("q25_full")

    ck = torch.load(GQ / "checkpoints/gazeqwen/best_model.pt", map_location="cpu", weights_only=False)
    f_theta = GazeLens(**ck["config"]).to(dev).eval()
    f_theta.load_state_dict(ck["state_dict"])
    apply_lora(model, rank=ck["lora_config"]["rank"], alpha=ck["lora_config"]["alpha"])
    load_lora_state_dict(model, ck["lora_state_dict"])
    ctx = GazeLensContext(f_theta)
    register_gazelens_hooks(model, ctx)
    vjepa = VJEPAFeatureExtractor(device=dev, checkpoint_path=str(GQ / "checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"))
    feats = vjepa.extract_from_raw_frames(frames, T, H, W)

    probs("lora_only", ctx)                                         # hooks installed, never activated
    probs("nogaze", ctx, torch.zeros(0, 4), feats)                  # empty scanpath -> zero residual (sanity: == lora_only)
    probs("inject_only", ctx, torch.tensor([[0.5, 0.5, -1000.0, 1.0]]), feats)   # inactive fixation: V-JEPA residual, no gaze
    for d, scan in scans.items():
        probs(f"gaze  (>= {d}s, n={len(scan)})", ctx, scan, feats)
        probs(f"placebo (>= {d}s, n={len(scan)})", ctx, random_placebo(scan, args.seed + int(args.qid)), feats)

    def gap(a, b):
        return float((results[a] - results[b]).abs().max())

    print("\nmax |logit diff|")
    print(f"  lora_only vs q25_full : {gap('lora_only', 'q25_full'):.3f}   (LoRA)")
    print(f"  nogaze    vs lora_only: {gap('nogaze', 'lora_only'):.3f}   (sanity: empty scanpath should add nothing)")
    print(f"  inject_only vs lora_only: {gap('inject_only', 'lora_only'):.3f}   (V-JEPA injection, no gaze)")
    for d, scan in scans.items():
        g, p = f"gaze  (>= {d}s, n={len(scan)})", f"placebo (>= {d}s, n={len(scan)})"
        note = "  [no fixations: gaze == nogaze by construction]" if len(scan) == 0 else ""
        print(f"  >= {d}s: gaze vs inject_only {gap(g, 'inject_only'):.3f} | gaze vs placebo {gap(g, p):.3f}{note}")
    print("answer:", row["answer"], "| need_audio:", row["need_audio"])


if __name__ == "__main__":
    main()
