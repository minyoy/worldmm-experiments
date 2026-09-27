"""
Shared helpers for the gaze-crop retrieval experiment.

Pure Python except for numpy; nothing here imports torch or worldmm, so build_pool /
prepare_gaze / recall_eval's bookkeeping run on a laptop. Only embed_arms.py needs the GPU.

Paths, timestamps and the 120-question subset come from the visual-bottleneck experiment
(../visual_bottleneck/common.py) so both experiments talk about the same clips.
"""

import json
import math
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

# 이 실험이 쓰는 문항 목록: EgoLifeQA 500 전체 (build_questions.py 가 만든다).
# visual_bottleneck 의 subset.json(120문항)이 아니다 -- 저쪽은 문항마다 LLM 을 돌려야
# 해서 줄일 이유가 있었지만, recall 채점은 문항을 늘리는 비용이 사실상 0 이고 검정력이
# 문항 수에 직접 걸린다.
QUESTIONS_PATH = os.path.join(EXP_DIR, "questions_500.json")
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


# target_time.time 안에 "...DAY3_20222516..." 처럼 끼어 있는 (날짜, 시각) 쌍을 잡는다.
# 원본 common.py:108 은 같은 패턴을 .match() 로 써서 맨 앞 하나만 보고 나머지를 버린다.
# 여기서는 findall 로 전부 잡는 것이 핵심 차이.
_TARGET_CHAIN_RE = re.compile(r"DAY(\d)_(\d{6,8})", re.IGNORECASE)


def _abs_sec(day: Any, hhmmssff: Any) -> float:
    """EgoLife 의 (날짜, 시각) 표기를 '일주일 전체를 잇는 하나의 타임라인' 위 초로 바꾼다.

    날짜가 다른 두 시각을 빼서 간격을 구하려면 DAY 를 초에 녹여 넣어야 한다.
    DAY1 을 0 이 아니라 86400 으로 두는데(day * 86400), 비교에만 쓰므로 원점은 상관없다.

    입력:
        day       "DAY3" / 3 / "day3"  (숫자가 아닌 문자는 전부 무시)
        hhmmssff  "20222516" 또는 20222516  -> 20시 22분 25초 16/100
                  8자리 미만이면 왼쪽을 0 으로 채운다 ("930000" -> "00930000")
    출력:
        float 초.  예) _abs_sec("DAY3", "20222516") = 3*86400 + 73345.16 = 332545.16

    주의: 프레임 단위 두 자리(FF)는 1/100 초다. 1/30 초(프레임 번호)가 아니다.
    """
    d = int(re.sub(r"\D", "", str(day)) or 0)
    t = str(hhmmssff).zfill(8)
    return d * 86400 + int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6]) + int(t[6:8]) / 100.0


def target_spans(target_time: Optional[Dict[str, Any]],
                 fallback_date: Any = None) -> List[Tuple[float, float]]:
    """EgoLifeQA 의 target_time 주석을 [시작, 끝] 구간 리스트로 정규화한다.

    이 실험에서 '정답'의 유일한 원천. EgoLifeQA 원본에는 target_clips 같은 필드가 없고
    target_time 하나뿐이다. "어느 30초 클립이 정답인가" 는 전부 이 저장소의 파생물이므로,
    격자가 아니라 주석된 시점을 기준으로 삼는다.

    입력 target_time 은 세 가지 모양으로 온다. 셋 다 '시점 목록' 하나로 합쳐서 내보낸다.

      (1) 단일 시각 - 451문항
          {"date": "DAY1", "time": "11152408"}
          -> [(97296.08, 97296.08)]                        길이 0 구간 = 한 시점

      (2) time_list - 36문항
          {"date": "DAY1", "time_list": ["13394908", "13512718"]}
          -> [(135589.08, 135589.08), (136287.18, 136287.18)]

      (3) time 안에 여러 시점이 연쇄 - 16문항
          {"date": "DAY4", "time": "17044917DAY4_20593813DAY6_19483807"}
          -> 3개 시점 [(DAY4 17:04:49), (DAY4 20:59:38), (DAY6 19:48:38)]

    fallback_date 는 target_time 에 date 가 없을 때만 쓴다(보통 query_time 의 date).
    target_time 이 비어 있으면 빈 리스트를 돌려주고, 호출부가 기존 target_keys 로 폴백한다.

    ## (3)을 '구간'이 아니라 '개별 시점'으로 읽는 이유

    원본 common.py:130-137 은 이걸 연속 구간으로 읽어서 그 사이 클립을 전부 정답으로 잡는다.
    그러면 ID 279 는 70시간 떨어진 두 순간 때문에 정답 클립이 2,430개가 되어 사실상 못 틀린다.
    하지만 16문항 전부가 재발성(HabitInsight) 질문이고, reason 필드가 직접 그렇게 말한다.

        ID  17  "**Both times**, Shure wrote on the whiteboard"
        ID  21  "I sat on the small horse chair **twice**"
        ID 482  Q: "How many times has Shure played the guitar?"  A: "3 times"  스탬프 3개

    즉 두세 번의 별개 사건이지 그 사이 구간이 아니다. 고친 뒤 정답 클립 최대치가
    2,430개 -> 6개(tol=0)로 떨어진다.

    구간 형태를 아예 안 쓰는 건 아니다. 반환 타입이 (시작, 끝) 인 건 나중에 진짜 구간
    주석이 들어와도 호출부를 안 고치기 위해서다. 지금 데이터에서는 항상 시작 == 끝.
    """
    t = target_time or {}
    date = t.get("date") or fallback_date
    out: List[Tuple[float, float]] = []

    # (2) time_list: 가장 명시적인 형태. 각 원소가 그대로 한 시점.
    if t.get("time_list"):
        for x in t["time_list"]:
            a = _abs_sec(date, x)
            out.append((a, a))
        return out

    tv = str(t.get("time") or "").strip()
    if not tv:
        return out                       # 주석 없음 -> 호출부가 폴백

    # (1)(3) 공통 처리.
    # "17044917DAY4_20593813DAY6_19483807" 를 split 하면 맨 앞 "17044917" 만 남는다.
    # 이 앞머리는 DAY 표기가 없으므로 target_time["date"] 소속이다.
    head = _TARGET_CHAIN_RE.split(tv)[0]
    if head:
        a = _abs_sec(date, head)
        out.append((a, a))
    # 나머지는 각자 DAY 를 달고 있다. findall 이라 세 번째 이후도 안 버린다.
    for day, stamp in _TARGET_CHAIN_RE.findall(tv):
        a = _abs_sec("DAY" + day, stamp)
        out.append((a, a))
    return out


def clip_span(clip: Dict[str, Any]) -> Tuple[float, float]:
    """풀에 담긴 클립 하나의 [시작, 끝] 을 target_spans 와 같은 타임라인 위 초로 바꾼다.

    입력:  pool.json 의 clips 원소, 또는 캡션 파일의 원소.
           {"date": "DAY1", "start_time": "11150000", "end_time": "11152900", ...}
           (start_time/end_time 이 문자열이든 int 든 상관없다)
    출력:  (86400+40500.0, 86400+40529.0) 같은 (시작초, 끝초) 튜플

    주의: 클립 길이가 정확히 30초가 아니다. 중앙값 29초, 최소 1초, 최대 30초이고
    이웃 클립 사이에 1초 이상 틈이 2,242개(중앙값 3초) 있다. 그 틈에 target_time 이
    떨어지면 tol=0 에서 정답 클립이 0개가 된다(13문항). tol 을 7초만 줘도 복구된다.
    """
    return (_abs_sec(clip["date"], clip["start_time"]), _abs_sec(clip["date"], clip["end_time"]))


def exact_paired_p(a_wins: int, b_wins: int) -> float:
    """짝지은 비교의 정확검정 p값. 이긴 횟수의 쏠림만 본다(부호검정 / McNemar 정확검정).

    입력:  a_wins  a가 이긴 문항(또는 클립) 수
           b_wins  b가 이긴 수
           비긴 것(둘 다 맞힘 / 둘 다 틀림 / 값이 같음)은 넣지 않는다 -- 정보가 없다.
    출력:  양측 p값
           exact_paired_p(0, 3) = 0.250
           exact_paired_p(0, 4) = 0.125
           exact_paired_p(0, 6) = 0.031   <- 한 방향으로 6개부터 p<0.05
           exact_paired_p(2, 2) = 1.000

    쓰이는 곳이 둘이다. 수학은 같다 -- 공정한 동전을 n번 던져 이만큼 쏠릴 확률.
      - recall_eval: 이진 적중(맞힘/틀림)의 엇갈린 문항  -> McNemar 정확검정
      - compare_transforms: 클립별 코사인이 더 높은 쪽    -> 부호검정

    부트스트랩 신뢰구간을 쓰다가 뺐다. 관측된 쏠림을 사실로 놓고 표본을 재추출하므로,
    불일치 d개 중 하나도 안 뽑힐 확률이 약 exp(-d) 뿐이라 d 가 작으면 거의 자동으로 0 을
    벗어난다. 실제로 이 데이터에서 CI 는 [-6.7,-0.8] 로 "유의", 정확검정은 p=0.125 로
    "유의하지 않음" 이었고 정확검정이 맞다.

    ★ 대신 효과 크기를 주지 않는다. 이긴 횟수와 차이(pp)를 같이 볼 것.
      쏠림이 작으면 "차이 없음" 이 아니라 "표본이 작아 판정 불가" 다.
    """
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    k = min(a_wins, b_wins)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def decord_threads() -> int:
    """Threads for each decord VideoReader. 0 (the default) lets decord use every core.

    Video decode is the CPU-heavy half of this experiment, and decord's auto setting will happily
    fill a shared machine. DECORD_NUM_THREADS caps it; run_stage1.sh sets it from MAX_CPUS.
    """
    try:
        return max(0, int(os.environ.get("DECORD_NUM_THREADS", "0")))
    except ValueError:
        return 0


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
