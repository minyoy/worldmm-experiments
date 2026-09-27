#!/usr/bin/env python3
"""
Turn an arm's embeddings into a drop-in replacement for visual_embeddings.pkl, so the existing
evaluator can answer questions with a gaze-cropped visual index and nothing else changed.

VisualMemory looks embeddings up by the caption's repo-relative video_path
("data/EgoLife/A1_JAKE/DAY1/....mp4"), so that is what the pkl is keyed by. Clips with no row are
left without an embedding, which VisualMemory skips at index time -- i.e. the pool this arm was
embedded over IS the retrieval index. Embed with `build_pool.py --all-clips` before running any
accuracy comparison against the shipped pkl, or the arm is searching a much smaller haystack.

    python experiments/gaze_crop/export_pkl.py --arm gazef@0.5
    # -> emb/gazef_05.pkl, then:
    python experiments/visual_bottleneck/eval_egolife.py --condition E_prime \
        --visual-path experiments/gaze_crop/emb/gazef_05.pkl \
        --results-dir experiments/gaze_crop/results_qa/gazef_05

Condition E_prime = B's cached text context + visual top-k for the question text. The frames handed
to the LLM are the retrieved clip's own full frames at 1 fps, exactly as before, so the only thing
the crop changes is WHICH clips get retrieved. That isolates the retrieval effect from a
presentation effect (showing the LLM cropped frames), which is a separate experiment.
"""

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import EMB_DIR, POOL_PATH, arm_filename, load_json  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="arm name, e.g. full | center@0.5 | gazef@0.5")
    ap.add_argument("--arm-suffix", default="", help="same suffix embed_arms.py was given, e.g. _randf")
    ap.add_argument("--emb-dir", default=EMB_DIR)
    ap.add_argument("--pool", default=POOL_PATH)
    ap.add_argument("--out", default=None, help="default: <emb-dir>/<arm>.pkl")
    args = ap.parse_args()

    npz_path = os.path.join(args.emb_dir, arm_filename(args.arm + args.arm_suffix))
    if not os.path.exists(npz_path):
        raise SystemExit(f"{npz_path} not found; run embed_arms.py --arms {args.arm} first")
    out = args.out or npz_path.replace(".npz", ".pkl")

    pool = load_json(args.pool)
    key_to_path = {r["key"]: r["video_path"] for r in pool["clips"]}
    z = np.load(npz_path, allow_pickle=False)
    keys, emb = z["keys"].tolist(), z["emb"].astype(np.float32)

    lookup, unknown = {}, []
    for k, vec in zip(keys, emb):
        vp = key_to_path.get(k)
        if vp is None:
            unknown.append(k)
            continue
        lookup[vp] = vec
    with open(out, "wb") as f:
        pickle.dump(lookup, f)

    print(f"{args.arm}: {len(lookup)} embeddings -> {out}")
    if unknown:
        print(f"WARNING: {len(unknown)} embedded clips are not in {args.pool}, e.g. {unknown[:2]}")
    n_pool = len(pool["clips"])
    print(f"this pkl will index {len(lookup)} clips (pool has {n_pool}, scope "
          f"'{pool['distractor_scope']}')")
    if pool["distractor_scope"] != "all-clips":
        print("NOTE: a sampled pool makes retrieval easier than the real 6,223-clip index. Accuracy "
              "from it is only comparable to another arm exported from the SAME pool, never to "
              "visual_bottleneck/results/E_prime.json.")


if __name__ == "__main__":
    main()
