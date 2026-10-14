"""Show, on the patch grid of one frame, the gaze term GazeLens adds to every visual key.

Two panels on the frame of the active step, with the LLM visual-token grid overlaid (one cell =
one visual token = 28x28 px of the resized frame): the gaze contribution to latent 0's attention
score, q0 . W_g G_i, for the real gaze and for the placebo. Every cell gets the same value.

    PYTHONPATH=".:third_party/vjepa2" python scripts/plot_gaze_bias_grid.py --index 634
"""
import argparse
import glob
import logging
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

    ck = torch.load("checkpoints/gazeqwen/best_model.pt", map_location="cpu", weights_only=False)
    m = GazeLens(**{k: v for k, v in ck["config"].items() if k != "version"})
    m.load_state_dict(ck["state_dict"])
    m.eval()
    mi, HW = 1, gh * gw                                # module for LLM layer 13, block 0
    with torch.no_grad():
        q0 = m.q_projs[mi][0](m.latents[mi])[0]        # query of latent 0

        def gaze_term(path):
            pe = _detr_cosine_pe(path, t, m.d_latent)
            G = pe.unsqueeze(0).expand(HW, -1)         # exactly what model.py does
            gk = m.g_projs[mi][0](G)                   # (HW, 256) added to the visual keys
            return gk, (gk @ q0).reshape(gh, gw).numpy()

        _, sc_real = gaze_term(sp)
        _, sc_plac = gaze_term(pl)
        j = int(((st <= t) & (t <= en)).nonzero()[0])

    img = frames[2 * step + 1].permute(1, 2, 0).numpy().astype(np.uint8)
    H, W = img.shape[:2]
    gx, gy = float(sp[j, 0]), float(sp[j, 1])
    px, py = float(pl[j, 0]), float(pl[j, 1])
    # narrow shared range so the real and placebo constants land on clearly different colours
    lo, hi = min(sc_real.min(), sc_plac.min()), max(sc_real.max(), sc_plac.max())
    pad = max(0.5 * (hi - lo), 1e-3)
    lo, hi = lo - pad, hi + pad

    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(1, 2, wspace=0.12)

    def draw(ax, val, title, cmap, vmin, vmax, alpha=0.55):
        ax.imshow(img)
        im = ax.imshow(val, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, extent=(0, W, H, 0), interpolation="nearest")
        for c in range(gw + 1):
            ax.axvline(c * W / gw, color="white", lw=0.4, alpha=0.7)
        for r in range(gh + 1):
            ax.axhline(r * H / gh, color="white", lw=0.4, alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.04)
        return ax

    a = draw(fig.add_subplot(gs[0, 0]), sc_real,
             f"GazeQwen, real gaze ({gx:.2f}, {gy:.2f})\ngaze term in latent-0 score, every patch = {sc_real.mean():.3f}\n"
             f"(std over {HW} patches = {sc_real.std():.1e})\ncolour range {lo:.3f}–{hi:.3f}", "viridis", lo, hi)
    a.scatter(gx * W, gy * H, s=220, facecolors="none", edgecolors="#2ca02c", linewidths=3)
    a = draw(fig.add_subplot(gs[0, 1]), sc_plac,
             f"GazeQwen, placebo ({px:.2f}, {py:.2f})\ngaze term in latent-0 score, every patch = {sc_plac.mean():.3f}\n"
             f"(std over {HW} patches = {sc_plac.std():.1e})\ncolour range {lo:.3f}–{hi:.3f}", "viridis", lo, hi)
    a.scatter(px * W, py * H, s=220, facecolors="none", edgecolors="#d62728", linewidths=3)
    fig.suptitle(f"sample {args.index} · step {step} (lookup {t:.0f} s) · grid {gh}x{gw} = {HW} visual tokens · "
                 "layer-13 module, block 1", fontsize=11)

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, f"sample{args.index}_gaze_bias_grid.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(out, f"grid {gh}x{gw}", "real const", sc_real.mean(), sc_real.std(), "placebo const", sc_plac.mean(), sc_plac.std())


if __name__ == "__main__":
    main()
