"""
Shared helpers for the gaze-crop retrieval experiment.

Pure Python except for numpy; nothing here imports torch or worldmm, so build_pool /
prepare_gaze / recall_eval's bookkeeping run on a laptop. Only embed_arms.py needs the GPU.

Paths, timestamps and the 120-question subset come from the visual-bottleneck experiment
(../visual_bottleneck/common.py) so both experiments talk about the same clips.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
VB_DIR = os.path.abspath(os.path.join(EXP_DIR, "..", "visual_bottleneck"))

# Loaded by path, not by name: this module also lives in a directory the scripts put on
# sys.path, and both files would otherwise answer to "common".
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("vb_common", os.path.join(VB_DIR, "common.py"))
vb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(vb)
sys.modules.setdefault("vb_common", vb)

# re-exported so the scripts import everything from one place
CAPTION_30SEC = vb.CAPTION_30SEC
DATA_DIR = vb.DATA_DIR
METADATA_DIR = vb.METADATA_DIR
QA_PATH = vb.QA_PATH
QA_TYPES = vb.QA_TYPES
REPO_ROOT = vb.REPO_ROOT
SUBJECT = vb.SUBJECT
SUBSET_PATH = vb.SUBSET_PATH
VIDEO_ROOT = vb.VIDEO_ROOT
clip_display_key = vb.clip_display_key
load_json = vb.load_json
load_subset = vb.load_subset
resolve_video_path = vb.resolve_video_path
save_json = vb.save_json
subset_by_id = vb.subset_by_id
ts_display = vb.ts_display
ts_int = vb.ts_int

POOL_PATH = os.path.join(EXP_DIR, "pool.json")
GAZE_PATH = os.path.join(EXP_DIR, "gaze_points.json")
EMB_DIR = os.path.join(EXP_DIR, "emb")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
ANALYSIS_DIR = os.path.join(EXP_DIR, "analysis")

# The authors' own embedding settings (preprocess/visual_memory/extract_visual_features.py ->
# EmbeddingModel.encode_video defaults). Every arm must use these, or the comparison also
# measures a resolution change.
NFRAMES = 16
MAX_PIXELS = 360 * 420

VISUAL_EMB_PKL = os.path.join(METADATA_DIR, "visual_memory", SUBJECT, "visual_embeddings.pkl")


# The eye-gaze release ships separately from the videos; this is where it is expected to sit.
# Overridable with WORLDMM_GAZE_ROOT, or --gaze-root on each script.
#   huggingface-cli download Wangtwohappy/EgoLife_EyeTracking_EyeGaze --repo-type dataset \
#       --include "EyeGaze/A1_JAKE/*" --local-dir data/EgoLife
GAZE_ROOT = os.environ.get("WORLDMM_GAZE_ROOT", os.path.join(DATA_DIR, "EyeGaze", SUBJECT))


# ---------------------------------------------------------------------------
# clips
# ---------------------------------------------------------------------------

def clip_key(video_path: str) -> str:
    """Filesystem-safe, unique id for a 30-sec clip: 'DAY1_A1_JAKE_11343000'."""
    return os.path.splitext(os.path.basename(video_path))[0]


def clip_start_sec(video_path: str) -> Optional[float]:
    """Seconds since midnight of a clip's start, read from its filename (..._HHMMSSFF.mp4)."""
    m = re.search(r"_(\d{8})$", clip_key(video_path))
    if not m:
        return None
    t = m.group(1)
    return float(int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])) + int(t[6:8]) / 100.0


# ---------------------------------------------------------------------------
# arms.  "full" | "center@<ratio>" | "gaze@<ratio>" | "pkl"
# ---------------------------------------------------------------------------

# gaze@R  fixes one box per clip at the clip's median gaze point (steady, and the naive reading).
# gazef@R follows the gaze frame by frame. On this data the per-clip median sits within a few
# percent of the frame centre (30 s of saccades average out), so gaze@R is close to center@R by
# construction and gazef@R is where the location information actually survives.
ARM_RE = re.compile(r"^(full|center|gaze|gazef|pkl)(?:@([0-9.]+))?$")


def parse_arm(name: str) -> Tuple[str, Optional[float]]:
    m = ARM_RE.match(name.strip())
    if not m:
        raise ValueError(f"bad arm '{name}'. Use full | center@0.5 | gaze@0.5 | gazef@0.5 | pkl")
    mode, ratio = m.group(1), m.group(2)
    if mode in ("center", "gaze", "gazef"):
        if ratio is None:
            raise ValueError(f"arm '{name}' needs a crop ratio, e.g. {mode}@0.5")
        r = float(ratio)
        if not 0 < r <= 1:
            raise ValueError(f"crop ratio must be in (0, 1], got {r}")
        return mode, r
    if ratio is not None:
        raise ValueError(f"arm '{name}' takes no ratio")
    return mode, None


def arm_filename(name: str) -> str:
    return name.replace("@", "_").replace(".", "") + ".npz"


# ---------------------------------------------------------------------------
# cropping.  A square whose side is `ratio` x the frame's shorter edge, centred on
# (cx, cy) in NORMALISED coordinates, shifted (not shrunk) to stay inside the frame so
# every arm sees the same amount of pixels regardless of where the gaze landed.
# ---------------------------------------------------------------------------

def crop_box(w: int, h: int, cx_norm: float, cy_norm: float, ratio: float) -> Tuple[int, int, int, int]:
    side = max(1, int(round(min(w, h) * ratio)))
    cx, cy = cx_norm * w, cy_norm * h
    x0 = int(round(min(max(cx - side / 2, 0), w - side)))
    y0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return x0, y0, x0 + side, y0 + side


# ---------------------------------------------------------------------------
# gaze_points.json
#   {clip_key: {"median": [xn, yn], "samples": [[t_in_clip_sec, xn, yn], ...], "source": str}}
# Normalised to [0, 1] with the origin at the frame's top-left, as PIL sees it.
# ---------------------------------------------------------------------------

def load_gaze(path: str = GAZE_PATH) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    data = load_json(path)
    return data.get("clips", data)


def gaze_at(entry: Dict[str, Any], t_sec: Optional[float], tol: float = 0.5) -> Optional[Tuple[float, float]]:
    """Gaze point for a moment inside the clip: nearest sample within `tol`, else the clip median.

    t_sec=None asks for the median outright, which is the naive (and steadier) setting: one box
    per clip instead of a box that jumps with every saccade.
    """
    if t_sec is not None and entry.get("samples"):
        best, best_d = None, None
        for s in entry["samples"]:
            d = abs(float(s[0]) - t_sec)
            if best_d is None or d < best_d:
                best, best_d = s, d
        if best is not None and best_d is not None and best_d <= tol:
            return float(best[1]), float(best[2])
    med = entry.get("median")
    if med:
        return float(med[0]), float(med[1])
    return None


def unit(x):
    import numpy as np
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def pct(n: int, d: int) -> str:
    return f"{n}/{d} = {100 * n / d:.1f}%" if d else f"{n}/0"
