"""Draw what GazeQwen actually receives for one eval sample: real gaze vs placebo.

Row 1: the 8 frames Qwen sees (4 temporal steps), with each frame's real time and the gaze lookup
       time of its step; real (green) / placebo (red) fixation drawn on the step where it is active.
Row 2: GazeLens latent attention over the visual positions of the active step (layer-13 module,
       first perceiver block, averaged over latents) with real gaze, placebo, and no gaze, plus
       the difference map, using the sample's real V-JEPA features.

    PYTHONPATH=".:third_party/vjepa2" python scripts/plot_gaze_vs_placebo.py --index 634
"""
import argparse
import glob
import json
import logging
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gazeqwen.data import GazeLensDataset
from gazeqwen.eval import placebo_scanpath
from gazeqwen.model import GazeLens, _detr_cosine_pe
from gazeqwen.train import compute_frame_times, create_video_clip, pick_fps

MODEL = ("/home/minyoy/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/"
         "cc594898137f460bfe9f0759e9844b3ce807cfb5")


def latent_attention(m, mi, X, scanpath, t):
    """Mean attention of the module's latents over the visual positions (first perceiver block)."""
    HW = X.shape[0]
    pe = _detr_cosine_pe(scanpath, t, m.d_latent) if scanpath.shape[0] else None
    G = torch.zeros(HW, m.d_latent) if pe is None else pe.unsqueeze(0).expand(HW, -1)
    L = m.latents[mi]
    pad = torch.zeros(m.n_latents, m.d_latent)
    Q = m.q_projs[mi][0](L)
    K = m.k_projs[mi][0](torch.cat([X, L])) + m.g_projs[mi][0](torch.cat([G, pad]))
    a = torch.softmax(Q @ K.t() / math.sqrt(m.d_latent), -1)
    return a[:, :HW].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=634, help="sample index in the test38 results order")
    ap.add_argument("--out", default="results/figs")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    ds = GazeLensDataset(qa_files=sorted(glob.glob("dataset/qa/*.json")), video_root="dataset/videos",
                         fixation_root="dataset/fixations", split="test",
                         split_file="splits/test_egtea_egoexo.json")
    s = ds[args.index]
    res = {c: json.load(open(f"results/{c}_test38.json"))["results"][args.index]
           for c in ("gazeqwen", "placebo", "lora_only")}
    assert res["gazeqwen"]["question_text"] == s["question_text"]

    # --- exactly what eval.py feeds Qwen ---
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    import decord

    start, end = s["start_time"], s["end_time"]
    fps = pick_fps(end - start)
    clip = create_video_clip(s["video_path"], start, end, target_fps=fps)
    msg = [{"role": "user", "content": [{"type": "video", "video": f"file://{clip}", "fps": fps, "max_frames": 8},
                                         {"type": "text", "text": "x"}]}]
    _, vids = process_vision_info(msg)
    frames = vids[0]                                   # (8, 3, H, W) float [0, 255]
    proc = AutoProcessor.from_pretrained(MODEL)
    grid = proc(text=[proc.apply_chat_template(msg, tokenize=False)], videos=vids, return_tensors="pt")["video_grid_thw"][0]
    T, llm_H, llm_W = int(grid[0]), int(grid[1]) // 2, int(grid[2]) // 2
    vr = decord.VideoReader(clip)
    idx = torch.linspace(0, len(vr) - 1, frames.shape[0]).round().long().tolist()
    real_t = [start + i / vr.get_avg_fps() for i in idx]
    lookup = compute_frame_times(start, T, fps)

    sp = s["scanpath"]
    pl = placebo_scanpath(sp, (s["video_path"], s["task_type"], s["question_text"]), 0)
    st, en = sp[:, 2] - sp[:, 3] / 2, sp[:, 2] + sp[:, 3] / 2
    active = [[j for j in range(len(sp)) if st[j] <= t <= en[j]] for t in lookup]
    step = next((k for k, a in enumerate(active) if a), None)

    # --- GazeLens attention on this sample's V-JEPA features ---
    ck = torch.load("checkpoints/gazeqwen/best_model.pt", map_location="cpu", weights_only=False)
    m = GazeLens(**{k: v for k, v in ck["config"].items() if k != "version"})
    m.load_state_dict(ck["state_dict"])
    m.eval()
    maps = {}
    if step is not None:
        from gazeqwen.vjepa_features import VJEPAFeatureExtractor
        vj = VJEPAFeatureExtractor(torch.device("cpu"), checkpoint_path="checkpoints/vjepa2_1_vitb_dist_vitG_384.pt")
        with torch.no_grad():
            feats = vj.extract_from_raw_frames(frames, T, llm_H, llm_W)
            mi = 1                                     # module for LLM layer 13
            X = m.proj_ins[mi](feats[step].float())
            t = lookup[step]
            for name, path in (("real gaze", sp), ("placebo", pl), ("no gaze", sp[:0])):
                maps[name] = latent_attention(m, mi, X, path, t).reshape(llm_H, llm_W).numpy()
            r_real = m(feats[step], sp, t, 13, llm_H, llm_W)
            r_plac = m(feats[step], pl, t, 13, llm_H, llm_W)
            rdiff = ((r_real - r_plac).norm() / r_real.norm()).item()

    # --- figure ---
    n = frames.shape[0]
    fig = plt.figure(figsize=(2.6 * n, 9.2))
    gs = fig.add_gridspec(3, n, height_ratios=[0.55, 1, 1.15], hspace=0.35)
    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    opts = "   ".join(s["options"])
    preds = (f"gold {s['answer']}   |   gaze → {res['gazeqwen']['pred']}   placebo → {res['placebo']['pred']}   "
             f"lora_only → {res['lora_only']['pred']}")
    fix_txt = "; ".join(f"#{j}: {st[j]:.1f}–{en[j]:.1f}s  real ({sp[j,0]:.2f},{sp[j,1]:.2f})  placebo ({pl[j,0]:.2f},{pl[j,1]:.2f})"
                        for j in range(len(sp)))
    ax.text(0, 1, f"sample {args.index} · {s['task_type']} · {os.path.basename(s['video_path'])} · clip [{start:.0f}s, {end:.0f}s], fps={fps}\n"
                  f"Q: {s['question_text']}\n{opts}\n{preds}\nfixations in clip: {fix_txt}",
            va="top", ha="left", fontsize=10, family="monospace", wrap=True)

    H, W = frames.shape[2], frames.shape[3]
    for f in range(n):
        k = f // 2
        a = fig.add_subplot(gs[1, f])
        a.imshow(frames[f].permute(1, 2, 0).numpy().astype(np.uint8))
        a.set_xticks([]); a.set_yticks([])
        col = "#d62728" if active[k] else "black"
        a.set_title(f"step {k} · frame {f}\nreal {real_t[f]:.1f}s | gaze lookup {lookup[k]:.0f}s",
                    fontsize=9, color=col)
        for j in active[k]:
            a.scatter(sp[j, 0] * W, sp[j, 1] * H, s=160, facecolors="none", edgecolors="#2ca02c", linewidths=2.5)
            a.scatter(pl[j, 0] * W, pl[j, 1] * H, s=160, facecolors="none", edgecolors="#d62728", linewidths=2.5)

    if maps:
        base = frames[2 * step + 1].permute(1, 2, 0).numpy().astype(np.uint8)
        vmax = max(v.max() for v in maps.values())
        panels = list(maps.items()) + [("real − placebo", maps["real gaze"] - maps["placebo"])]
        span = n // len(panels)
        for p, (name, mp) in enumerate(panels):
            a = fig.add_subplot(gs[2, p * span:(p + 1) * span])
            a.imshow(base, alpha=0.55)
            if name.startswith("real −"):
                lim = max(abs(mp).max(), 1e-12)
                im = a.imshow(mp, cmap="coolwarm", vmin=-lim, vmax=lim, alpha=0.75, extent=(0, W, H, 0))
                a.set_title(f"{name}\nmax |Δ| = {lim:.1e} (map max {vmax:.1e})", fontsize=10)
            else:
                im = a.imshow(mp, cmap="magma", vmin=0, vmax=vmax, alpha=0.6, extent=(0, W, H, 0))
                a.set_title(f"latent attention, step {step} · {name}", fontsize=10)
            for j in active[step]:
                if name in ("real gaze", "real − placebo"):
                    a.scatter(sp[j, 0] * W, sp[j, 1] * H, s=160, facecolors="none", edgecolors="#2ca02c", linewidths=2.5)
                if name in ("placebo", "real − placebo"):
                    a.scatter(pl[j, 0] * W, pl[j, 1] * H, s=160, facecolors="none", edgecolors="#d62728", linewidths=2.5)
            a.set_xticks([]); a.set_yticks([])
            plt.colorbar(im, ax=a, fraction=0.035)
        fig.text(0.5, 0.015, f"green = real fixation, red = placebo (same time span, centre moved ≥ 0.25). "
                             f"GazeLens residual (layer 13, step {step}): ‖real − placebo‖ / ‖real‖ = {rdiff:.1e}",
                 ha="center", fontsize=10)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"sample{args.index}_gaze_vs_placebo.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out, "| active per step", active, "| real times", [round(x, 1) for x in real_t], "| lookup", lookup)


if __name__ == "__main__":
    main()
