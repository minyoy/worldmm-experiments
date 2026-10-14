"""Split GazeLens' attention score over the patch grid into its content term and its gaze term.

    score_i = q . W_k x_i      (content: V-JEPA feature of patch i -> varies per patch)
            + q . W_g G        (gaze: one vector copied to every patch -> same for all i)

Panels on the frame of the active step (layer-13 module, first block, latent 0):
  content term | gaze term (real) | attention with real gaze | attention with placebo

    PYTHONPATH=".:third_party/vjepa2" python scripts/plot_score_decomposition.py --index 634
"""
import argparse
import glob
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=634)
    ap.add_argument("--latent", type=int, default=0)
    ap.add_argument("--out", default="results/figs")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    ds = GazeLensDataset(qa_files=sorted(glob.glob("dataset/qa/*.json")), video_root="dataset/videos",
                         fixation_root="dataset/fixations", split="test",
                         split_file="splits/test_egtea_egoexo.json")
    s = ds[args.index]
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor

    start, end = s["start_time"], s["end_time"]
    fps = pick_fps(end - start)
    clip = create_video_clip(s["video_path"], start, end, target_fps=fps)
    msg = [{"role": "user", "content": [{"type": "video", "video": f"file://{clip}", "fps": fps, "max_frames": 8},
                                         {"type": "text", "text": "x"}]}]
    _, vids = process_vision_info(msg)
    frames = vids[0]
    proc = AutoProcessor.from_pretrained(MODEL)
    grid = proc(text=[proc.apply_chat_template(msg, tokenize=False)], videos=vids, return_tensors="pt")["video_grid_thw"][0]
    T, gh, gw = int(grid[0]), int(grid[1]) // 2, int(grid[2]) // 2
    lookup = compute_frame_times(start, T, fps)

    sp = s["scanpath"]
    pl = placebo_scanpath(sp, (s["video_path"], s["task_type"], s["question_text"]), 0)
    st, en = sp[:, 2] - sp[:, 3] / 2, sp[:, 2] + sp[:, 3] / 2
    step = next(k for k, t in enumerate(lookup) if ((st <= t) & (t <= en)).any())
    t = lookup[step]
    j = int(((st <= t) & (t <= en)).nonzero()[0])

    ck = torch.load("checkpoints/gazeqwen/best_model.pt", map_location="cpu", weights_only=False)
    m = GazeLens(**{k: v for k, v in ck["config"].items() if k != "version"})
    m.load_state_dict(ck["state_dict"])
    m.eval()
    from gazeqwen.vjepa_features import VJEPAFeatureExtractor
    vj = VJEPAFeatureExtractor(torch.device("cpu"), checkpoint_path="checkpoints/vjepa2_1_vitb_dist_vitG_384.pt")

    mi, HW, li = 1, gh * gw, args.latent
    with torch.no_grad():
        feats = vj.extract_from_raw_frames(frames, T, gh, gw)
        X = m.proj_ins[mi](feats[step].float())
        L = m.latents[mi]
        pad = torch.zeros(m.n_latents, m.d_latent)
        Q = m.q_projs[mi][0](L)
        KX = m.k_projs[mi][0](torch.cat([X, L]))            # content part of every key

        def parts(path):
            pe = _detr_cosine_pe(path, t, m.d_latent)
            G = (torch.zeros(HW, m.d_latent) if pe is None else pe.unsqueeze(0).expand(HW, -1))
            KG = m.g_projs[mi][0](torch.cat([G, pad]))      # gaze part of every key
            att = torch.softmax(Q @ (KX + KG).t() / math.sqrt(m.d_latent), -1)
            return (Q[li] @ KG[:HW].t()) / math.sqrt(m.d_latent), att[li, :HW]

        content = (Q[li] @ KX[:HW].t()) / math.sqrt(m.d_latent)
        gaze_real, att_real = parts(sp)
        _, att_plac = parts(pl)

    rs = lambda v: v.reshape(gh, gw).numpy()                # noqa: E731
    img = frames[2 * step + 1].permute(1, 2, 0).numpy().astype(np.uint8)
    H, W = img.shape[:2]
    gx, gy, px, py = float(sp[j, 0]), float(sp[j, 1]), float(pl[j, 0]), float(pl[j, 1])

    c, g = rs(content), rs(gaze_real)
    lo, hi = min(c.min(), g.min()), max(c.max(), g.max())
    amax = float(max(att_real.max(), att_plac.max()))
    panels = [
        (c, f"① content term  q·W_k x_i\n(V-JEPA feature of each patch)\nrange {c.min():.2f} … {c.max():.2f}", "viridis", lo, hi),
        (g, f"② gaze term  q·W_g G  (real gaze)\nsame for every patch = {g.mean():.3f}\n(same colour scale as ①)", "viridis", lo, hi),
        (rs(att_real), "③ attention = softmax(① + ②)\nwith REAL gaze", "magma", 0, amax),
        (rs(att_plac), "④ attention = softmax(① + ②')\nwith PLACEBO gaze", "magma", 0, amax),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(21, 6.6))
    for ax, (val, title, cmap, vmin, vmax) in zip(axes, panels):
        ax.imshow(img)
        im = ax.imshow(val, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.65, extent=(0, W, H, 0), interpolation="nearest")
        for k in range(gw + 1):
            ax.axvline(k * W / gw, color="white", lw=0.3, alpha=0.6)
        for k in range(gh + 1):
            ax.axhline(k * H / gh, color="white", lw=0.3, alpha=0.6)
        ax.scatter(gx * W, gy * H, s=200, facecolors="none", edgecolors="#2ca02c", linewidths=2.5)
        if "PLACEBO" in title:
            ax.scatter(px * W, py * H, s=200, facecolors="none", edgecolors="#d62728", linewidths=2.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.045)
    fig.suptitle(f"sample {args.index} · step {step} (lookup {t:.0f} s) · {gh}x{gw} patches · layer-13 module, block 1, latent {li}   "
                 f"| green = real fixation ({gx:.2f}, {gy:.2f}), red = placebo ({px:.2f}, {py:.2f})", fontsize=12)
    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"sample{args.index}_score_decomposition.png")
    fig.savefig(out, dpi=105, bbox_inches="tight")
    print(out, "content std", float(c.std()), "gaze std", float(g.std()), "att max", amax)


if __name__ == "__main__":
    main()
