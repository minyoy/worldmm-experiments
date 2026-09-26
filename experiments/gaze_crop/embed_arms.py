#!/usr/bin/env python3
"""
Embed every pool clip once per arm, changing nothing but the pixels.

Each clip is decoded once (16 uniform frames, decord), then the same frames are cropped per arm
and handed to EmbeddingModel.encode_video as a frame list. VLM2Vec's qwen_vl_utils treats a list
under the "video" key as a video, so the model, nframes (16) and max_pixels (360*420) stay exactly
what the authors used to build visual_embeddings.pkl -- see preprocess/visual_memory/
extract_visual_features.py, which relies on encode_video's defaults.

Arms
    full          the whole frame. The within-this-script baseline.
    center@R      square crop, side = R x the shorter edge, centred on the frame.
    gaze@R        same box, centred on the clip's gaze point (gaze_points.json).
    pkl           no GPU: the authors' own embedding for that clip, copied out of
                  visual_embeddings.pkl. Reference point only -- it goes through the mp4 path of
                  fetch_video rather than the frame-list path, so tiny differences from `full` are
                  expected and are not part of the comparison.

center@R is not optional. An egocentric frame is centre-biased, so cropping alone enlarges the
subject and drops background; without center@R a gaze win cannot be separated from a crop win.

    python experiments/gaze_crop/embed_arms.py --arms full center@0.5 gaze@0.5
    python experiments/gaze_crop/embed_arms.py --arms pkl                 # laptop-friendly
    python experiments/gaze_crop/embed_arms.py --arms gaze@0.5 --gaze gaze_random.json \
        --arm-suffix _rand                                                # crop-anywhere control

Writes emb/<arm>.npz  {keys: [clip_key], emb: [n, d]}  plus emb/<arm>.meta.json.
Re-running is incremental: existing rows are kept unless --overwrite.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    EMB_DIR, GAZE_PATH, MAX_PIXELS, NFRAMES, POOL_PATH, VIDEO_ROOT, VISUAL_EMB_PKL, arm_filename,
    crop_box, decord_threads, gaze_at, load_gaze, load_json, parse_arm, resolve_video_path,
    save_json,
)


def frame_indices_uniform(total: int, n: int) -> List[int]:
    """Same rule as visual_bottleneck/extract_oracle_frames.frame_indices_uniform."""
    if total <= n:
        return list(range(total))
    return np.linspace(0, total - 1, n, dtype=int).tolist()


def decode(path: str, nframes: int):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0), num_threads=decord_threads())
    idx = frame_indices_uniform(len(vr), nframes)
    arr = vr.get_batch(idx).asnumpy()
    fps = float(vr.get_avg_fps()) or 30.0
    return arr, idx, fps


def crop_frames(arr, idx, fps, mode: str, ratio: Optional[float], gaze_entry: Optional[Dict[str, Any]],
                per_frame: bool, tol: float):
    """PIL frames for one arm. Returns None when the arm cannot be built for this clip."""
    from PIL import Image
    h, w = arr.shape[1:3]
    out = []
    if mode == "full":
        return [Image.fromarray(a) for a in arr]
    if mode == "center":
        box = crop_box(w, h, 0.5, 0.5, ratio)
        return [Image.fromarray(a).crop(box) for a in arr]
    # gaze (mode "gaze" = one box per clip; "gazef" = follow the gaze frame by frame)
    if not gaze_entry:
        return None
    per_frame = per_frame or mode == "gazef"
    fixed = None
    if not per_frame:
        p = gaze_at(gaze_entry, None)
        if p is None:
            return None
        fixed = crop_box(w, h, p[0], p[1], ratio)
    for a, fi in zip(arr, idx):
        if fixed is not None:
            box = fixed
        else:
            p = gaze_at(gaze_entry, fi / fps, tol=tol)
            if p is None:
                return None
            box = crop_box(w, h, p[0], p[1], ratio)
        out.append(Image.fromarray(a).crop(box))
    return out


class Store:
    """An arm's npz, loaded so a run can resume where the last one stopped."""

    def __init__(self, path: str, overwrite: bool):
        self.path = path
        self.rows: Dict[str, np.ndarray] = {}
        self.prev_meta: Dict[str, Any] = {}
        meta_path = path.replace(".npz", ".meta.json")
        if os.path.exists(path) and not overwrite:
            z = np.load(path, allow_pickle=False)
            self.rows = {k: v for k, v in zip(z["keys"].tolist(), z["emb"])}
            if os.path.exists(meta_path):
                self.prev_meta = load_json(meta_path)

    def has(self, key: str) -> bool:
        return key in self.rows

    def reset(self) -> None:
        """Throw away what was loaded, so the arm is rebuilt from scratch."""
        self.rows = {}
        self.prev_meta = {}

    def put(self, key: str, vec: np.ndarray) -> None:
        self.rows[key] = np.asarray(vec, dtype=np.float32)

    def save(self, meta: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        keys = sorted(self.rows)
        np.savez(self.path, keys=np.array(keys), emb=np.stack([self.rows[k] for k in keys]))
        meta = dict(meta, n=len(keys), saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        save_json(meta, self.path.replace(".npz", ".meta.json"))


def fill_from_pkl(store: Store, pool: Dict[str, Any], pkl_path: str) -> Tuple[int, int]:
    import pickle
    with open(pkl_path, "rb") as f:
        lookup = pickle.load(f)
    hit = miss = 0
    for r in pool["clips"]:
        # the pkl is keyed by the ORIGINAL repo-relative video path
        vec = lookup.get(r["video_path"])
        if vec is None:
            miss += 1
            continue
        store.put(r["key"], np.asarray(vec, dtype=np.float32))
        hit += 1
    return hit, miss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL_PATH)
    ap.add_argument("--gaze", default=GAZE_PATH)
    ap.add_argument("--arms", nargs="+", default=["full", "center@0.5", "gaze@0.5", "gazef@0.5"])
    ap.add_argument("--arm-suffix", default="", help="appended to each output filename, e.g. _rand")
    ap.add_argument("--out-dir", default=EMB_DIR)
    ap.add_argument("--video-root", default=VIDEO_ROOT)
    ap.add_argument("--nframes", type=int, default=NFRAMES)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    ap.add_argument("--per-frame-gaze", action="store_true",
                    help="make every gaze@R arm follow the gaze frame by frame. Prefer the gazef@R "
                         "arm name, which does the same for one arm only, so gaze@R (clip median) "
                         "and gazef@R (per frame) can be compared in a single run.")
    ap.add_argument("--gaze-tol", type=float, default=0.5, help="seconds; nearest-sample tolerance")
    ap.add_argument("--pkl", default=VISUAL_EMB_PKL)
    ap.add_argument("--limit", type=int, default=None, help="first N pool clips (smoke test)")
    ap.add_argument("--targets-only", action="store_true", help="skip distractors (debugging)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true", help="recompute instead of resuming")
    ap.add_argument("--dry-run", action="store_true",
                    help="decode and crop, but do not load the model: checks videos, gaze coverage "
                         "and crop geometry without a GPU")
    args = ap.parse_args()

    pool = load_json(args.pool)
    gaze = load_gaze(args.gaze)
    clips = [r for r in pool["clips"] if not (args.targets_only and not r["is_target"])]
    if args.limit:
        clips = clips[:args.limit]

    specs = [(name, *parse_arm(name)) for name in args.arms]
    stores = {name: Store(os.path.join(args.out_dir, arm_filename(name + args.arm_suffix)), args.overwrite)
              for name, _, _ in specs}
    # the transform the gaze file was built with. Rows made under a different one are not
    # comparable, and mixing them inside one npz is invisible afterwards -- so it is refused.
    gaze_transform = None
    if os.path.exists(args.gaze):
        gaze_transform = load_json(args.gaze).get("transform")
    base_meta = {"pool": args.pool, "nframes": args.nframes, "max_pixels": args.max_pixels,
                 "video_root": args.video_root, "gaze_file": args.gaze,
                 "gaze_transform": gaze_transform,
                 "per_frame_gaze": args.per_frame_gaze, "n_clips_requested": len(clips)}

    # A gaze arm whose stored rows were built under a different transform is rebuilt, not resumed:
    # resuming would mix two coordinate frames inside one npz, which is invisible afterwards and
    # makes every number from that arm meaningless.
    for name, mode, _ in specs:
        if mode not in ("gaze", "gazef"):
            continue
        prev = stores[name].prev_meta
        was = prev.get("gaze_transform") if prev else None
        if prev and was not in (None, gaze_transform):
            print(f"WARNING: {name} holds {len(stores[name].rows)} embeddings built with transform "
                  f"'{was}', but {args.gaze} now says '{gaze_transform}'. Discarding them and "
                  f"rebuilding this arm from scratch.")
            stores[name].reset()

    # ---- pkl arm: no video, no model -----------------------------------------
    for name, mode, _ in specs:
        if mode == "pkl":
            hit, miss = fill_from_pkl(stores[name], pool, args.pkl)
            stores[name].save(dict(base_meta, arm=name, source=args.pkl))
            print(f"[{name}] copied {hit} embeddings from the pkl ({miss} pool clips missing)")

    gpu_specs = [(n, m, r) for n, m, r in specs if m != "pkl"]
    if not gpu_specs:
        return

    todo = [r for r in clips if any(not stores[n].has(r["key"]) for n, _, _ in gpu_specs)]
    print(f"arms: {[n for n, _, _ in gpu_specs]}")
    print(f"clips to do: {len(todo)}/{len(clips)} (the rest are already in emb/)")
    need_gaze = any(m in ("gaze", "gazef") for _, m, _ in gpu_specs)
    if need_gaze:
        have = sum(1 for r in clips if r["key"] in gaze)
        print(f"gaze coverage: {have}/{len(clips)} clips from {args.gaze}")
        fake = next((v.get("source", "") for v in gaze.values()
                     if str(v.get("source", "")).startswith("pseudo:")), None)
        if fake:
            print(f"  !! {args.gaze} holds FABRICATED gaze ({fake}), not eye tracking. "
                  f"'pseudo:center' is the pipeline test (it must equal center@R); "
                  f"'pseudo:random' is the crop-anywhere control. Do not report either as gaze.")
        if have == 0:
            print("  no gaze for any pool clip. Either point --gaze at a prepared file, or drop the "
                  "gaze arm and run `--arms full center@0.5`.")
            return

    model = None
    if not args.dry_run:
        from worldmm.embedding import EmbeddingModel  # late import: --dry-run needs no GPU
        model = EmbeddingModel()

    missing_video, missing_gaze, sizes = [], [], set()
    t_start = time.time()
    for i, r in enumerate(todo, 1):
        path = resolve_video_path(r["video_path"], args.video_root)
        if not os.path.exists(path):
            missing_video.append(path)
            continue
        try:
            arr, idx, fps = decode(path, args.nframes)
        except Exception as e:
            print(f"  [{r['key']}] decode failed: {e}")
            missing_video.append(path)
            continue
        sizes.add((int(arr.shape[2]), int(arr.shape[1])))

        for name, mode, ratio in gpu_specs:
            if stores[name].has(r["key"]):
                continue
            frames = crop_frames(arr, idx, fps, mode, ratio, gaze.get(r["key"]),
                                 args.per_frame_gaze, args.gaze_tol)
            if frames is None:
                missing_gaze.append((r["key"], name))
                continue
            if args.dry_run:
                continue
            vec = model.encode_video([{"video": frames, "nframes": len(frames),
                                       "max_pixels": args.max_pixels}])
            stores[name].put(r["key"], np.asarray(vec)[0])

        if args.save_every and i % args.save_every == 0:
            rate = i / max(time.time() - t_start, 1e-9)
            eta_min = (len(todo) - i) / rate / 60
            if not args.dry_run:
                for name, _, _ in gpu_specs:
                    stores[name].save(dict(base_meta, arm=name))
            print(f"  ... {i}/{len(todo)} clips, {rate:.2f} clip/s "
                  f"({rate * len(gpu_specs):.2f} embeddings/s), ETA {eta_min:.0f} min"
                  f"{'' if args.dry_run else ', checkpointed'}")

    if not args.dry_run:
        for name, _, _ in gpu_specs:
            stores[name].save(dict(base_meta, arm=name))
            print(f"[{name}] {len(stores[name].rows)} embeddings -> {stores[name].path}")
    else:
        print(f"[dry run] decoded {len(todo) - len(missing_video)} clips, frame sizes {sorted(sizes)}, "
              f"no model loaded")
    dt = time.time() - t_start
    if todo:
        print(f"{len(todo)} clips x {len(gpu_specs)} arm(s) in {dt / 60:.1f} min "
              f"= {dt / len(todo):.2f} s/clip. Scale that to your real pool: "
              f"a 6,223-clip pool would take {6223 * dt / len(todo) / 3600:.1f} h at this rate.")
    if missing_video:
        print(f"WARNING: {len(missing_video)} clip videos missing/unreadable, e.g. {missing_video[:2]}")
    if missing_gaze:
        print(f"WARNING: {len(missing_gaze)} (clip, arm) pairs had no gaze point, e.g. {missing_gaze[:3]}. "
              f"They are absent from that arm; recall_eval.py reports the shortfall rather than "
              f"quietly substituting a centre crop.")


if __name__ == "__main__":
    main()
