"""
Shared helpers for the visual-bottleneck experiment.

Everything here is pure Python (no torch / worldmm imports) so that the
lightweight scripts (build_subset, seed_hipporag_cache, analyze) can run
without loading the model stack.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP_DIR, "..", ".."))

SUBJECT = "A1_JAKE"
DATA_DIR = os.path.join(REPO_ROOT, "data", "EgoLife")
QA_PATH = os.path.join(DATA_DIR, "EgoLifeQA", f"EgoLifeQA_{SUBJECT}.json")
CAPTION_DIR = os.path.join(DATA_DIR, "EgoLifeCap", SUBJECT)
CAPTION_30SEC = os.path.join(CAPTION_DIR, f"{SUBJECT}_30sec.json")
METADATA_DIR = os.path.join(REPO_ROOT, "output", "metadata")
CACHE_ROOT = os.path.join(REPO_ROOT, ".cache", "episodic_memory")

# 비디오 위치. 캡션 JSON의 video_path 는 "data/EgoLife/A1_JAKE/DAY1/....mp4" 처럼
# 저장소 기준 상대 경로로 적혀 있는데, 실제 비디오는 보통 다른 디스크에 있다.
# 아래 값이 그 "data/EgoLife" 부분을 대체한다. 환경변수 WORLDMM_VIDEO_ROOT 로 덮어쓸 수 있고,
# 각 스크립트의 --video-root 인자가 최우선이다.
VIDEO_ROOT = os.environ.get("WORLDMM_VIDEO_ROOT", "/datasets/EgoLife")

SUBSET_PATH = os.path.join(EXP_DIR, "subset.json")
FRAMES_DIR = os.path.join(EXP_DIR, "frames")
FRAMES16_DIR = os.path.join(EXP_DIR, "frames16")
TEXT_CONTEXT_DIR = os.path.join(EXP_DIR, "text_context")
MASKS_DIR = os.path.join(EXP_DIR, "masks")
MASKED_FRAMES_DIR = os.path.join(EXP_DIR, "masked_frames")
RESULTS_DIR = os.path.join(EXP_DIR, "results")
ANALYSIS_DIR = os.path.join(EXP_DIR, "analysis")

QA_TYPES = ["EntityLog", "RelationMap", "EventRecall", "TaskMaster", "HabitInsight"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Timestamps. WorldMM encodes a time as int(day_digit + HHMMSSFF), e.g.
# DAY1 11:09:43 -> 111094300. All comparisons in the codebase use this int.
# ---------------------------------------------------------------------------

def day_digit(date: str) -> str:
    return str(date).replace("DAY", "").replace("Day", "")[-1]


def ts_int(date: str, hhmmssff: Any) -> int:
    return int(day_digit(date) + str(hhmmssff).zfill(8))


def ts_display(ts: int) -> str:
    """111094300 -> 'DAY1 11:09:43' (same format as VisualMemory display keys)."""
    s = str(ts)
    return f"DAY{s[0]} {s[1:3]}:{s[3:5]}:{s[5:7]}"


def clip_display_key(cap: Dict[str, Any]) -> str:
    """Display key used by VisualMemory for a 30-sec caption entry."""
    start = ts_int(cap["date"], cap["start_time"])
    end = ts_int(cap["date"], cap["end_time"])
    return f"{ts_display(start)} - {ts_display(end)}"


def query_time_int(row: Dict[str, Any]) -> int:
    return ts_int(row["query_time"]["date"], row["query_time"]["time"])


def gap_days(row: Dict[str, Any]) -> Optional[int]:
    """query day - target day; None when target has no date."""
    t = row.get("target_time") or {}
    if not t.get("date"):
        return None
    return int(day_digit(row["query_time"]["date"])) - int(day_digit(t["date"]))


def gap_bin(gap: Optional[int]) -> str:
    if gap is None:
        return "na"
    if gap <= 0:
        return "0"
    if gap == 1:
        return "1"
    return "2+"


# ---------------------------------------------------------------------------
# target_time -> 30-sec caption clips. Same rules as eval/eval_egolife.py
# (parse_target_time + find_30s_segment), returning caption entries instead of
# (start, end) tuples so callers get the video_path too.
# ---------------------------------------------------------------------------

_RANGE_RE = re.compile(r"(\d+)DAY(\d)_(\d+)", re.IGNORECASE)


def _segments(captions_30sec: List[Dict[str, Any]]) -> List[Tuple[int, int, Dict[str, Any]]]:
    return [(ts_int(c["date"], c["start_time"]), ts_int(c["date"], c["end_time"]), c) for c in captions_30sec]


def target_clips(row: Dict[str, Any], captions_30sec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the 30-sec caption entries that contain the row's target_time(s), sorted by time."""
    t = row.get("target_time") or {}
    date = t.get("date") or row["query_time"]["date"]
    segs = _segments(captions_30sec)
    found: Dict[str, Dict[str, Any]] = {}

    def add_point(ts: int) -> None:
        for s, e, c in segs:
            if s <= ts <= e:
                found[c["video_path"]] = c

    if t.get("time_list"):
        for x in t["time_list"]:
            add_point(ts_int(date, x))
    elif t.get("time") and "DAY" in str(t["time"]).upper():
        m = _RANGE_RE.match(t["time"])
        if m:
            s0 = ts_int(date, m.group(1))
            e0 = ts_int("DAY" + m.group(2), m.group(3))
            for s, e, c in segs:
                if s <= e0 and e >= s0:
                    found[c["video_path"]] = c
    elif t.get("time"):
        add_point(ts_int(date, t["time"]))

    return sorted(found.values(), key=lambda c: ts_int(c["date"], c["start_time"]))


def build_choices(row: Dict[str, Any]) -> Dict[str, str]:
    choices = {}
    for key, label in [("choice_a", "A"), ("choice_b", "B"), ("choice_c", "C"), ("choice_d", "D")]:
        if key in row and row[key]:
            choices[label] = row[key]
    return choices


def load_subset(path: str = SUBSET_PATH) -> Dict[str, Any]:
    return load_json(path)


def subset_ids(subset: Dict[str, Any]) -> List[str]:
    return [e["ID"] for e in subset["entries"]]


def subset_by_id(subset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {e["ID"]: e for e in subset["entries"]}


def resolve_video_path(video_path: str, video_root: str = None) -> str:
    """캡션의 video_path 를 실제 파일 경로로 바꾼다.

    "data/EgoLife/A1_JAKE/DAY1/x.mp4" -> "<video_root>/A1_JAKE/DAY1/x.mp4"
    접두사가 다르거나 이미 절대 경로면 그대로 둔다.
    """
    root = video_root or VIDEO_ROOT
    prefix = "data/EgoLife/"
    if video_path.startswith(prefix):
        return os.path.join(root, video_path[len(prefix):])
    if os.path.isabs(video_path):
        return video_path
    return os.path.join(REPO_ROOT, video_path)


def remap_caption_video_paths(captions, video_root: str = None):
    """캡션 리스트의 video_path 를 제자리에서 실제 경로로 바꾸고, 바뀐 개수를 돌려준다.

    VisualMemory 는 이 값을 그대로 들고 있다가 프레임을 뽑을 때 파일을 연다.
    load_visual_clips() 에 넘기기 전에 이 함수를 통과시키면 src 수정 없이 경로가 교정된다.
    """
    n = 0
    for c in captions:
        vp = c.get("video_path")
        if not vp:
            continue
        new = resolve_video_path(vp, video_root)
        if new != vp:
            c["video_path"] = new
            n += 1
    return n
