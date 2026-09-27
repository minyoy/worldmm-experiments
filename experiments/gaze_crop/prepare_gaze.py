#!/usr/bin/env python3
"""
EgoLife eye gaze -> one normalised file the embedder can read.

Source (per clip, one CSV per mp4, 10 Hz):
    https://huggingface.co/datasets/Wangtwohappy/EgoLife_EyeTracking_EyeGaze
    EyeGaze/A1_JAKE/DAY1/DAY1_A1_JAKE_11100000.csv   <- same stem as the clip's mp4

    huggingface-cli download Wangtwohappy/EgoLife_EyeTracking_EyeGaze --repo-type dataset \
        --include "EyeGaze/A1_JAKE/*" --local-dir data/EgoLife

That CSV is Aria MPS eye gaze: angles in the Central Pupil Frame, not pixels.
    tracking_timestamp_us, left_yaw_rads_cpf, right_yaw_rads_cpf, pitch_rads_cpf, depth_m, ...
So it needs (a) per-clip time rebasing -- the timestamps are a device clock, ~21042 s here, and the
first sample of each file is that clip's t=0 -- and (b) projection into the video frame.

Projection is approximate and the script says so in its output. Doing it exactly needs the device
calibration from the VRS file (RGB fisheye624 + the CPF->camera extrinsic), which the gaze release
does not ship. What runs instead: the cyclopean direction (left/right yaw averaged) projected with
one focal length, equidistant by default because Aria's RGB lens is a ~110 degree fisheye and a
pinhole model pushes edge gaze too far out. The CPF->camera offset (a few cm) is ignored, which
matters most for close targets -- median depth in this data is ~0.6 m, so expect a degree or two of
error, not a different half of the frame.

Which is why --dump-variants exists: EgoLife's mp4s are processed (Aria stores RGB rotated 90
degrees) and the sign conventions cannot be settled from a spec sheet. It renders one frame under
every rotation/flip so you can pick --transform by eye before spending GPU time.

Output (gaze_points.json), normalised to [0,1], origin top-left as PIL sees the decoded frame:
    {"clips": {"<clip_key>": {"median": [xn, yn], "samples": [[t_in_clip_sec, xn, yn], ...],
                              "n": N, "clipped": M, "source": "<file>"}}}

    python experiments/gaze_crop/prepare_gaze.py --dump-variants 3
    python experiments/gaze_crop/prepare_gaze.py --transform rot90cw --dump-overlay 6
    python experiments/gaze_crop/prepare_gaze.py --pseudo center   # pipeline test: gaze@R == center@R
    python experiments/gaze_crop/prepare_gaze.py --pseudo random-frames \
        --out gaze_random_frames.json                  # moving-box control
"""

import argparse
import csv
import math
import os
import random
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gaze_common import (  # noqa: E402
    GAZE_PATH, GAZE_ROOT, POOL_PATH, VIDEO_ROOT, clip_span, clip_start_sec, decord_threads,
    load_json, resolve_video_path, save_json,
)

# Aria RGB at its native 1408x1408. Scaled by the decoded frame's width.
ARIA_FOCAL_PX = 611.0
ARIA_NATIVE_W = 1408
# Eye origins in CPF, from the Aria docs: left [0.0315, 0, 0], right [-0.0315, 0, 0].
EYE_HALF_BASELINE_M = 0.0315


def combined_gaze(left_yaw: float, right_yaw: float, pitch: float) -> Tuple[float, float, Optional[float]]:
    """The docs' `compute_depth_and_combined_gaze_direction`, without the toolkit.

    Both eyes sit on the CPF x axis, so each ray is x = +-b + z*tan(yaw) and y = z*tan(pitch).
    Solving the two x equations gives the vergence depth; averaging them gives x/z, i.e. the
    combined direction is atan of the MEAN OF THE TANGENTS (not of the angles -- that differs by
    up to ~1.6 deg on this data, about 20 px). Pitch is common to both rays, so it passes through.

    Verified against the CSV's own depth_m column on A1_JAKE DAY1: median |z - depth_m| = 0.011 m,
    which is also what confirms the left/right sign convention.
    """
    tl, tr = math.tan(left_yaw), math.tan(right_yaw)
    den = tr - tl
    depth = (2 * EYE_HALF_BASELINE_M / den) if abs(den) > 1e-9 else None
    if depth is not None and depth <= 0:
        depth = None
    return math.atan((tl + tr) / 2), pitch, depth

X_COLS = ("gaze_x", "projected_point_2d_x", "gaze_2d_x", "x_pixel", "screen_x", "x", "u", "px")
Y_COLS = ("gaze_y", "projected_point_2d_y", "gaze_2d_y", "y_pixel", "screen_y", "y", "v", "py")
YAW_COLS = ("yaw_rads_cpf", "yaw_rad", "yaw")
LEFT_YAW_COLS = ("left_yaw_rads_cpf", "left_yaw")
RIGHT_YAW_COLS = ("right_yaw_rads_cpf", "right_yaw")
PITCH_COLS = ("pitch_rads_cpf", "pitch_rad", "pitch")
T_COLS = ("tracking_timestamp_us", "t_in_clip_sec", "time_s", "time_sec", "timestamp_s",
          "timestamp_us", "timestamp_ns", "timestamp", "time", "sec", "frame_index", "frame")

OPS = ("none", "rot90cw", "rot90ccw", "flipx", "flipy")
# combinations worth eyeballing: a rotation, then an axis flip
VARIANTS = ("none", "flipx", "flipy", "flipx+flipy", "rot90cw", "rot90cw+flipx", "rot90cw+flipy",
            "rot90ccw", "rot90ccw+flipx")

CLIP_SECONDS = 30.0  # EgoLife's 30-sec segments; used to sanity-check a rebased timeline


def pick(cols: List[str], wanted: Tuple[str, ...]) -> Optional[str]:
    low = {c.strip().lower(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def apply_transform(xn: float, yn: float, how: str) -> Tuple[float, float]:
    """`how` is one or more ops joined by '+', applied left to right."""
    for op in str(how).split("+"):
        op = op.strip()
        if op == "rot90cw":
            xn, yn = 1.0 - yn, xn
        elif op == "rot90ccw":
            xn, yn = yn, 1.0 - xn
        elif op == "flipx":
            xn = 1.0 - xn
        elif op == "flipy":
            yn = 1.0 - yn
        elif op not in ("", "none"):
            raise ValueError(f"unknown transform op '{op}'; pick from {OPS} joined by '+'")
    return xn, yn


def make_projector(w: int, h: int, projection: str, focal_px: Optional[float],
                   hfov: Optional[float]) -> Tuple[Any, Dict[str, Any]]:
    """(yaw, pitch) in radians -> normalised image point, plus the parameters used."""
    if focal_px is None:
        focal_px = (ARIA_FOCAL_PX * w / ARIA_NATIVE_W) if hfov is None else (w / 2) / math.tan(
            math.radians(hfov) / 2)
    f = float(focal_px)
    info = {"projection": projection, "focal_px": f, "frame_size": [w, h],
            "implied_hfov_deg": round(2 * math.degrees(math.atan((w / 2) / f)), 1)}

    def project(yaw: float, pitch: float) -> Tuple[float, float]:
        # direction in CPF, +x right / +y down / +z forward
        vx, vy, vz = math.tan(yaw), math.tan(pitch), 1.0
        if projection == "pinhole":
            return 0.5 + f * vx / w, 0.5 + f * vy / h
        n = math.sqrt(vx * vx + vy * vy + vz * vz)
        theta = math.acos(max(-1.0, min(1.0, vz / n)))       # angle off the optical axis
        r = f * theta                                         # equidistant: r = f * theta
        rho = math.hypot(vx, vy) or 1e-9
        return 0.5 + r * (vx / rho) / w, 0.5 + r * (vy / rho) / h

    return project, info


def rebase(samples: List[List[float]]) -> Tuple[List[List[float]], bool]:
    """Per-clip files carry a device clock. Shift so the first sample is t=0 when needed."""
    ts = [s[0] for s in samples if s[0] >= 0]
    if not ts:
        return samples, False
    t0 = min(ts)
    if 0.0 <= t0 <= CLIP_SECONDS + 5 and max(ts) <= CLIP_SECONDS + 10:
        return samples, False                                  # already clip-relative
    return [[s[0] - t0 if s[0] >= 0 else -1.0, s[1], s[2]] for s in samples], True


def to_seconds(col: str, raw: str) -> Optional[float]:
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    c = col.strip().lower()
    if c.endswith("_us"):
        return v / 1e6
    if c.endswith("_ns"):
        return v / 1e9
    if c in ("frame", "frame_index"):
        return None
    if v > 1e6:                                                # bare HHMMSSFF, e.g. 11343019
        s = str(int(v)).zfill(8)
        return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6]) + int(s[6:8]) / 100.0
    return v


def parse_csv(path: str, fmt: str, project, transform: str, w: int, h: int,
              cpf_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
              default_depth: float = 1.0
              ) -> Tuple[List[List[float]], str, int, Optional[float], Optional[float]]:
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], "empty file", 0, None, None
    cols = list(rows[0].keys())
    xc, yc = pick(cols, X_COLS), pick(cols, Y_COLS)
    yawc = pick(cols, YAW_COLS)
    lyc, ryc = pick(cols, LEFT_YAW_COLS), pick(cols, RIGHT_YAW_COLS)
    pitchc = pick(cols, PITCH_COLS)
    tc = pick(cols, T_COLS)
    depthc = pick(cols, ("depth_m", "depth"))

    has_angles = pitchc and (yawc or (lyc and ryc))
    if fmt == "aria_yawpitch" and not has_angles:
        return [], f"no yaw/pitch columns in {cols[:6]}", 0, None, None
    mode = "xy" if (xc and yc and fmt != "aria_yawpitch") else ("angles" if has_angles else None)
    if mode is None:
        return [], f"no usable columns in {cols[:8]}", 0, None, None

    samples: List[List[float]] = []
    clipped = 0
    depths: List[float] = []
    depth_err: List[float] = []      # |vergence depth - depth_m column|: the sign-convention check
    use_offset = any(abs(v) > 1e-9 for v in cpf_offset)
    for r in rows:
        t = to_seconds(tc, r[tc]) if tc else None
        if mode == "xy":
            try:
                x, y = float(r[xc]), float(r[yc])
            except (TypeError, ValueError):
                continue
            xn, yn = (x, y) if max(abs(x), abs(y)) <= 1.5 else (x / w, y / h)
        else:
            try:
                pitch = float(r[pitchc])
                z_verg = None
                if yawc:
                    yaw = float(r[yawc])
                else:
                    yaw, pitch, z_verg = combined_gaze(float(r[lyc]), float(r[ryc]), pitch)
            except (TypeError, ValueError):
                continue
            d_col = None
            if depthc:
                try:
                    d_col = float(r[depthc])
                except (TypeError, ValueError):
                    d_col = None
            if z_verg is not None and d_col is not None:
                depth_err.append(abs(z_verg - d_col))
            if use_offset:
                # the docs' route: take the 3D gaze point at its depth, move it into the camera
                # frame, then project. Only meaningful once a CPF->camera translation is known.
                z = d_col or z_verg or default_depth
                px, py, pz = z * math.tan(yaw), z * math.tan(pitch), z
                px, py, pz = px - cpf_offset[0], py - cpf_offset[1], pz - cpf_offset[2]
                if pz <= 1e-6:
                    continue
                xn, yn = project(math.atan(px / pz), math.atan(py / pz))
            else:
                xn, yn = project(yaw, pitch)
        xn, yn = apply_transform(xn, yn, transform)
        if depthc:
            try:
                depths.append(float(r[depthc]))
            except (TypeError, ValueError):
                pass
        # Gaze outside the frame is real (the wearer looked past the camera's edge); clamp rather
        # than drop, so the crop box still tracks the edge the gaze went to.
        cx, cy = min(max(xn, 0.0), 1.0), min(max(yn, 0.0), 1.0)
        clipped += (cx != xn) or (cy != yn)
        samples.append([t if t is not None else -1.0, cx, cy])

    how = f"columns {xc}/{yc}" if mode == "xy" else (
        f"combined yaw atan(mean tan of {lyc},{ryc})" if not yawc else f"yaw {yawc}") + f" + {pitchc}"
    if mode == "angles" and use_offset:
        how += f" + CPF offset {cpf_offset}"
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    return samples, how, clipped, med(depths), med(depth_err)


def aria_official_projector(vrs_file: str, stream_label: str = "camera-rgb"):
    """The docs' exact route, when a VRS with the device calibration is available.

    mps.read_eyegaze + get_gaze_vector_reprojection, i.e. the real fisheye624 model and the real
    CPF->camera extrinsic, instead of this script's single-focal approximation. Returns a function
    (csv_path, depth_default) -> [[t_sec, xn, yn], ...] in the CALIBRATED sensor's normalised
    coordinates. EgoLife's gaze release ships no VRS, so this path is opt-in and unexercised here;
    it fails loudly rather than silently falling back.
    """
    import projectaria_tools.core.mps as mps
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection

    provider = data_provider.create_vrs_data_provider(vrs_file)
    device_calib = provider.get_device_calibration()
    cam_calib = device_calib.get_camera_calib(stream_label)
    cw, ch = cam_calib.get_image_size()

    def run(csv_path: str, depth_default: float) -> Tuple[List[List[float]], str]:
        gaze = mps.read_eyegaze(csv_path)
        out: List[List[float]] = []
        t0 = None
        for g in gaze:
            t = g.tracking_timestamp.total_seconds()
            t0 = t if t0 is None else t0
            depth = getattr(g, "depth", 0.0) or depth_default
            xy = get_gaze_vector_reprojection(g, stream_label, device_calib, cam_calib, depth)
            if xy is None:
                continue
            out.append([t - t0, float(xy[0]) / cw, float(xy[1]) / ch])
        return out, f"projectaria get_gaze_vector_reprojection ({stream_label}, {cw}x{ch})"

    return run


def parse_json(path: str, w: int, h: int, transform: str
               ) -> Tuple[List[List[float]], str, int, None, None]:
    data = load_json(path)
    if isinstance(data, dict):
        for k in ("gaze", "samples", "points", "data"):
            if k in data:
                data = data[k]
                break
    if not isinstance(data, list):
        return [], "json is not a list of points", 0, None, None
    samples, clipped = [], 0
    for it in data:
        t = None
        if isinstance(it, dict):
            x, y = it.get("x", it.get("gaze_x")), it.get("y", it.get("gaze_y"))
            t = it.get("t", it.get("time", it.get("t_sec")))
        elif isinstance(it, (list, tuple)) and len(it) >= 3:
            t, x, y = it[0], it[1], it[2]
        elif isinstance(it, (list, tuple)) and len(it) == 2:
            x, y = it
        else:
            continue
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        xn, yn = (x, y) if max(abs(x), abs(y)) <= 1.5 else (x / w, y / h)
        xn, yn = apply_transform(xn, yn, transform)
        cx, cy = min(max(xn, 0.0), 1.0), min(max(yn, 0.0), 1.0)
        clipped += (cx != xn) or (cy != yn)
        samples.append([float(t) if t is not None else -1.0, cx, cy])
    return samples, "json", clipped, None, None


def median_xy(samples: List[List[float]]) -> List[float]:
    xs = sorted(s[1] for s in samples)
    ys = sorted(s[2] for s in samples)
    m = len(xs) // 2
    return [xs[m], ys[m]]


def frame_size(pool: Dict[str, Any], video_root: str) -> Optional[Tuple[int, int]]:
    try:
        from decord import VideoReader, cpu
    except ImportError:
        return None
    for r in pool["clips"]:
        p = resolve_video_path(r["video_path"], video_root)
        if os.path.exists(p):
            try:
                vr = VideoReader(p, ctx=cpu(0), num_threads=decord_threads())
                h, w = vr[0].shape[:2]
                return int(w), int(h)
            except Exception:
                return None
    return None


def pseudo(pool: Dict[str, Any], kind: str, seed: int, rate_hz: float = 10.0
           ) -> Dict[str, Dict[str, Any]]:
    """대조군용 가짜 gaze 파일.

    세 종류이고, 무엇을 반증하려는지가 서로 다르다.

      center        클립마다 (0.5, 0.5) 고정. 파이프라인 점검용 -- center@R 과 같은 픽셀이 나와야 한다.
      random-frames 프레임마다 독립적으로 무작위. **`gazef@R` 의 박스 움직임까지 흉내낸다.**
                    "gazef 가 이긴 것은 시선 위치가 아니라 박스가 움직여 16프레임이 서로 다른 영역을
                    덮은 탓"이라는 설명을 끊어 내는 유일한 대조군이다.

    고정 무작위 박스(클립당 점 하나)는 뺐다. `samples` 가 비면 gaze_at() 이 median 으로 폴백해 박스가
    클립 내내 고정되므로, center@R 과 "중앙이냐 아니냐"만 다른 arm 이 되어 얻는 것이 없었다.

    출력 samples 는 [클립 내 초, x, y] 이고 rate_hz 간격으로 깔린다(기본 10 Hz = 실제 Aria CSV 와 같음).
    """
    rng = random.Random(seed)
    out = {}
    for r in pool["clips"]:
        if kind == "center":
            xy, samples = [0.5, 0.5], []
        elif kind == "random-frames":
            cs, ce = clip_span(r)
            n = max(1, int(round(max(ce - cs, 1.0) * rate_hz)))
            samples = [[i / rate_hz, rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)]
                       for i in range(n)]
            xy = [statistics.median(s[1] for s in samples),
                  statistics.median(s[2] for s in samples)]
        else:
            raise SystemExit(f"unknown --pseudo {kind}")
        out[r["key"]] = {"median": xy, "samples": samples, "n": len(samples),
                         "source": f"pseudo:{kind}"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL_PATH)
    ap.add_argument("--out", default=GAZE_PATH)
    ap.add_argument("--gaze-root", default=GAZE_ROOT,
                    help=f"directory holding the per-clip gaze CSVs (default: {GAZE_ROOT})")
    ap.add_argument("--format", choices=("auto", "xy_csv", "aria_yawpitch", "per_clip_json"),
                    default="auto")
    ap.add_argument("--pseudo", choices=("center", "random-frames"), default=None,
                    help="no gaze: fabricate a point per clip. 'center' makes gaze@R identical to "
                         "center@R (pipeline test); 'random-frames' re-draws a random box every "
                         "frame, the control for 'the box merely moved' rather than 'it moved to "
                         "the gaze'.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pseudo-rate", type=float, default=10.0,
                    help="--pseudo random-frames 의 샘플 간격(Hz). 기본 10 = 실제 gaze CSV 와 같음")
    ap.add_argument("--video-root", default=VIDEO_ROOT)
    ap.add_argument("--frame-width", type=int, default=None, help="default: read from a pool video")
    ap.add_argument("--frame-height", type=int, default=None)
    ap.add_argument("--projection", choices=("equidistant", "pinhole"), default="equidistant",
                    help="Aria RGB is a fisheye, so equidistant (r = f*theta) is the better cheap "
                         "model; pinhole (r = f*tan theta) overshoots away from the centre.")
    ap.add_argument("--focal-px", type=float, default=None,
                    help=f"default: Aria RGB {ARIA_FOCAL_PX:.0f}px at {ARIA_NATIVE_W}px wide, scaled "
                         f"to the decoded frame")
    ap.add_argument("--hfov", type=float, default=None,
                    help="alternative to --focal-px: horizontal FOV in degrees")
    ap.add_argument("--cpf-offset", default="0,0,0", metavar="X,Y,Z",
                    help="CPF->RGB camera translation in metres. Default 0,0,0 ignores it, which is "
                         "self-consistent (with no translation the depth cannot change the "
                         "projected direction). Ignoring a ~3cm offset costs ~2.9 deg = 31px at this "
                         "data's median depth of 0.59m. Set it only from a real calibration.")
    ap.add_argument("--default-depth", type=float, default=1.0,
                    help="depth for samples with no usable depth, only used with --cpf-offset. The "
                         "Aria docs use 1.0 m: arm's reach, where a held object would be.")
    ap.add_argument("--aria-vrs", default=None,
                    help="path to a VRS with the device calibration. When given, projection goes "
                         "through projectaria_tools' get_gaze_vector_reprojection -- the exact "
                         "fisheye624 model and the real CPF->camera extrinsic -- instead of this "
                         "script's single-focal approximation. EgoLife's gaze release has no VRS.")
    ap.add_argument("--transform", default="none",
                    help=f"fix a coordinate-frame mismatch: {OPS} joined by '+', e.g. rot90cw+flipy. "
                         f"Pick it with --dump-variants.")
    ap.add_argument("--dump-variants", type=int, default=0, metavar="N",
                    help="render N clips' middle frame under every rotation/flip, as a labelled "
                         "grid, so --transform can be chosen by eye")
    ap.add_argument("--dump-overlay", type=int, default=0, metavar="N",
                    help="render N frames with the chosen transform's gaze point and crop box drawn")
    ap.add_argument("--overlay-ratio", type=float, default=0.5)
    args = ap.parse_args()

    pool = load_json(args.pool)
    keys = {r["key"] for r in pool["clips"]}

    if args.pseudo:
        clips = pseudo(pool, args.pseudo, args.seed, args.pseudo_rate)
        save_json({"format": f"pseudo:{args.pseudo}", "transform": "none", "clips": clips}, args.out)
        print(f"pseudo gaze ({args.pseudo}) for {len(clips)} clips -> {args.out}")
        if args.dump_overlay:
            overlay(pool, clips, args)
        return

    if not os.path.isdir(args.gaze_root):
        ap.error(f"gaze root not found: {args.gaze_root}\n"
                 f"Pass --gaze-root, set WORLDMM_GAZE_ROOT, or use --pseudo center / --pseudo random-frames.")
    print(f"gaze root: {args.gaze_root}")

    w, h = args.frame_width, args.frame_height
    if w is None or h is None:
        probed = frame_size(pool, args.video_root)
        if probed is None:
            ap.error("could not read a video's resolution; pass --frame-width/--frame-height")
        w, h = probed
        print(f"frame size probed from video: {w}x{h}")
    try:
        cpf_offset = tuple(float(x) for x in args.cpf_offset.split(","))
        assert len(cpf_offset) == 3
    except (ValueError, AssertionError):
        ap.error("--cpf-offset takes three comma-separated metres, e.g. 0.0,-0.01,0.01")

    official = None
    if args.aria_vrs:
        try:
            official = aria_official_projector(args.aria_vrs)
        except ImportError as e:
            ap.error(f"--aria-vrs needs projectaria_tools: {e}")
        except Exception as e:
            ap.error(f"could not open {args.aria_vrs}: {e}")
        print(f"projection: projectaria_tools via {args.aria_vrs} (calibrated), "
              f"transform '{args.transform}'")
    project, proj_info = make_projector(w, h, args.projection, args.focal_px, args.hfov)
    if official is None:
        print(f"projection: {proj_info['projection']}, focal {proj_info['focal_px']:.1f}px "
              f"(implied HFOV {proj_info['implied_hfov_deg']}deg), transform '{args.transform}', "
              f"CPF offset {cpf_offset}")
    proj_info["cpf_offset"] = list(cpf_offset)
    proj_info["calibrated"] = official is not None

    by_clip: Dict[str, Dict[str, Any]] = {}
    day_samples: Dict[str, List[List[float]]] = {}
    day_how: Dict[str, str] = {}
    notes: List[str] = []
    depth_errs: List[float] = []
    n_files = n_rebased = 0
    for dirpath, _, filenames in os.walk(args.gaze_root):
        for fn in sorted(filenames):
            path = os.path.join(dirpath, fn)
            stem, ext = os.path.splitext(fn)
            ext = ext.lower()
            if ext == ".json" and args.format in ("auto", "per_clip_json"):
                samples, how, clipped, depth, derr = parse_json(path, w, h, args.transform)
            elif ext in (".csv", ".txt") and official is not None:
                samples, how = official(path, args.default_depth)
                samples = [[t, *apply_transform(x, y, args.transform)] for t, x, y in samples]
                clipped = sum(1 for _, x, y in samples if not (0 <= x <= 1 and 0 <= y <= 1))
                samples = [[t, min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)] for t, x, y in samples]
                depth = derr = None
            elif ext in (".csv", ".txt"):
                samples, how, clipped, depth, derr = parse_csv(
                    path, args.format, project, args.transform, w, h, cpf_offset, args.default_depth)
            else:
                continue
            if derr is not None:
                depth_errs.append(derr)
            if not samples:
                notes.append(f"{fn}: {how}")
                continue
            n_files += 1
            if stem in keys:
                samples, was_rebased = rebase(samples)
                n_rebased += was_rebased
                by_clip[stem] = {"median": median_xy(samples), "samples": samples, "n": len(samples),
                                 "clipped": clipped, "median_depth_m": depth, "source": path,
                                 "how": how, "rebased": was_rebased}
            else:
                day = next((d for d in ("DAY1", "DAY2", "DAY3", "DAY4", "DAY5", "DAY6", "DAY7")
                            if d.lower() in path.lower()), None)
                if day is None:
                    notes.append(f"{fn}: matches no clip key and no DAYn, skipped")
                    continue
                day_samples.setdefault(day, []).extend(samples)
                day_how[day] = how

    from_day = {d: 0 for d in day_samples}
    for r in pool["clips"]:                      # day-level streams, sliced per clip
        if r["key"] in by_clip:
            continue
        stream = day_samples.get(r["date"])
        c0 = clip_start_sec(r["video_path"])
        if not stream or c0 is None:
            continue
        inside = [[s[0] - c0, s[1], s[2]] for s in stream if s[0] >= 0 and 0.0 <= s[0] - c0 <= 31.0]
        if not inside:
            continue
        by_clip[r["key"]] = {"median": median_xy(inside), "samples": inside, "n": len(inside),
                             "clipped": 0, "median_depth_m": None,
                             "source": f"{r['date']} day-level stream",
                             "how": day_how.get(r["date"], "?"), "rebased": False}
        from_day[r["date"]] = from_day.get(r["date"], 0) + 1

    # A day-level file's times must be absolute (seconds since midnight / HHMMSSFF); Aria's
    # tracking_timestamp_us is a device clock. A stream that lines up with no clip is almost always
    # that, or a stem that failed to match -- say so instead of silently contributing nothing.
    for day, n in from_day.items():
        if n == 0:
            ts = [x[0] for x in day_samples[day] if x[0] >= 0]
            span = f"{min(ts):.0f}..{max(ts):.0f}s" if ts else "no times"
            notes.append(f"{day}: a day-level file ({span}) matched no clip. Aria timestamps are a "
                         f"device clock, not time of day -- name each CSV after its clip "
                         f"(DAY1_A1_JAKE_11100000.csv) so it is read as that clip.")

    covered = sum(1 for r in pool["clips"] if r["key"] in by_clip)
    n_tgt = sum(1 for r in pool["clips"] if r["is_target"])
    tgt_covered = sum(1 for r in pool["clips"] if r["is_target"] and r["key"] in by_clip)
    save_json({"format": args.format, "transform": args.transform, "clips": by_clip,
               "vergence_depth_check_median_m": (sorted(depth_errs)[len(depth_errs) // 2]
                                                 if depth_errs else None), **proj_info}, args.out)

    print(f"gaze files parsed: {n_files} ({n_rebased} had a device clock rebased to clip time)")
    print(f"clips with gaze: {covered}/{len(pool['clips'])} (targets: {tgt_covered}/{n_tgt})")
    if by_clip:
        v = list(by_clip.values())
        ns = sorted(x["n"] for x in v)
        print(f"samples per clip: min {ns[0]}, median {ns[len(ns)//2]}, max {ns[-1]}")
        xs = sorted(x["median"][0] for x in v)
        ys = sorted(x["median"][1] for x in v)
        mid = len(xs) // 2
        print(f"per-clip median gaze: x {xs[mid]:.3f}, y {ys[mid]:.3f} "
              f"(x/y spread p10-p90: {xs[len(xs)//10]:.2f}-{xs[-len(xs)//10]:.2f} / "
              f"{ys[len(ys)//10]:.2f}-{ys[-len(ys)//10]:.2f})")
        print("  if that centre sits at 0.5/0.5 with almost no spread, gaze adds nothing a centre "
              "crop does not already have -- check the transform before concluding it")
        clipped = sum(x["clipped"] for x in v)
        total = sum(x["n"] for x in v)
        print(f"samples clamped to the frame edge: {clipped}/{total} = {100*clipped/max(total,1):.1f}% "
              f"(a big number usually means the wrong projection or transform)")
        print(f"example: {v[0]['how']}")
        if depth_errs:
            m = sorted(depth_errs)[len(depth_errs) // 2]
            print(f"vergence check: median |reconstructed depth - depth_m| = {m:.3f} m over "
                  f"{len(depth_errs)} file(s). Small means the left/right columns and their signs "
                  f"are being read as the Aria docs define them; large means they are not.")
    for n in notes[:10]:
        print(f"  note: {n}")
    print(f"wrote {args.out}")

    # Rendering is a convenience, not the product: gaze_points.json is already written, so a
    # failure here (a missing video, a codec, PIL) must not fail the run that comes after it.
    for label, fn in (("variants", lambda: variants(pool, args, w, h)),
                      ("overlay", lambda: overlay(pool, by_clip, args))):
        if getattr(args, f"dump_{label}"):
            try:
                fn()
            except Exception as e:
                print(f"WARNING: --dump-{label} failed ({type(e).__name__}: {e}). "
                      f"{args.out} is written and usable; rerun the dump separately if you want it.")


def _mid_frame(video_path: str, video_root: str):
    from decord import VideoReader, cpu
    from PIL import Image
    p = resolve_video_path(video_path, video_root)
    if not os.path.exists(p):
        return None
    vr = VideoReader(p, ctx=cpu(0), num_threads=decord_threads())
    return Image.fromarray(vr[len(vr) // 2].asnumpy())


def _draw(img, xn: float, yn: float, ratio: float, label: str):
    from PIL import ImageDraw
    from gaze_common import crop_box
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle(crop_box(out.width, out.height, xn, yn, ratio), outline=(255, 0, 0), width=6)
    cx, cy = xn * out.width, yn * out.height
    d.ellipse([cx - 14, cy - 14, cx + 14, cy + 14], fill=(255, 0, 0))
    d.rectangle([0, 0, 420, 54], fill=(0, 0, 0))
    d.text((10, 18), label, fill=(255, 255, 255))
    return out


def variants(pool: Dict[str, Any], args, w: int, h: int) -> None:
    """One frame per clip, drawn under every rotation/flip, as a 3x3 grid. Pick --transform by eye."""
    from PIL import Image
    project, _ = make_projector(w, h, args.projection, args.focal_px, args.hfov)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "variants")
    os.makedirs(out_dir, exist_ok=True)

    done = 0
    for r in pool["clips"]:
        if done >= args.dump_variants:
            break
        csv_path = None
        for dirpath, _, filenames in os.walk(args.gaze_root):
            if r["key"] + ".csv" in filenames:
                csv_path = os.path.join(dirpath, r["key"] + ".csv")
                break
        if csv_path is None:
            continue
        img = _mid_frame(r["video_path"], args.video_root)
        if img is None:
            continue
        # the frame we drew is the clip's middle, so use the gaze sample nearest that moment
        base, _, _, _, _ = parse_csv(csv_path, args.format, project, "none", w, h)
        base, _ = rebase(base)
        if not base:
            continue
        t_mid = 15.0
        s = min(base, key=lambda x: abs(x[0] - t_mid) if x[0] >= 0 else 1e9)
        tiles = []
        for name in VARIANTS:
            xn, yn = apply_transform(s[1], s[2], name)
            tiles.append(_draw(img, xn, yn, args.overlay_ratio, name))
        tw, th = tiles[0].width // 2, tiles[0].height // 2
        grid = Image.new("RGB", (tw * 3, th * 3), (20, 20, 20))
        for i, t in enumerate(tiles):
            grid.paste(t.resize((tw, th)), ((i % 3) * tw, (i // 3) * th))
        path = os.path.join(out_dir, f"{r['key']}_variants.jpg")
        grid.save(path, quality=85)
        print(f"variants: {path}")
        done += 1
    if done:
        print("Pick the tile where the box sits on what the wearer was plainly looking at, then "
              "rerun with --transform <that label>.")
    else:
        print("variants: no clip had both a CSV and a readable video")


def overlay(pool: Dict[str, Any], clips: Dict[str, Dict[str, Any]], args) -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "overlay")
    os.makedirs(out_dir, exist_ok=True)
    done = 0
    for r in pool["clips"]:
        if done >= args.dump_overlay or r["key"] not in clips:
            continue
        img = _mid_frame(r["video_path"], args.video_root)
        if img is None:
            continue
        xn, yn = clips[r["key"]]["median"]
        _draw(img, xn, yn, args.overlay_ratio, f"{r['key']}  transform={args.transform}").save(
            os.path.join(out_dir, f"{r['key']}.jpg"), quality=88)
        done += 1
    print(f"overlay: {done} frame(s) in {out_dir}")


if __name__ == "__main__":
    main()
