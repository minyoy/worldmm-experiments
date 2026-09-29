#!/usr/bin/env python3
"""
EgoServe 이벤트를 gaze_crop 의 recall_eval.py 가 읽는 pool 형식으로 바꾼다.

EgoServe (data/EgoServe/EgoLife/A1_JAKE/*.json) 는 QA 가 아니라 proactive 어시스턴트 이벤트다.
영상·gaze 는 EgoLife 와 같은 것을 쓰므로, 클립 6,223 개와 그 임베딩(full / center / randf / gazef)은
../gaze_crop 것을 그대로 쓴다. 새로 만드는 것은 "질의"와 "정답 클립" 뿐이다.

검색 정답이 정의되는 이벤트만 쓴다 (과거 장면을 가리키는 것):

  episodic.json    56   current_* 장면에서 past_time_window 의 과거 장면을 떠올려야 한다
                        memory_recall 19 (forgotten_item / check_status / unfinished_plan / context_cue)
                        task_reminder 37
  long_term.json   32   linked_past_events 로 과거 장면이 명시된 30 개 이벤트 (과거 장면 32 개)

instant.json(70), short_term.json(121)은 과거 장면 지시가 없어 검색 정답이 없으므로 제외한다.

질의(query) = 정답 장면의 텍스트 묘사 (질의 품질을 이상적으로 고정한 상한선)
  episodic     past_observation   과거 장면을 묘사한 문장
  long_term    memory_key         연결된 과거 기억의 요약 ("Charge power banks in bedroom ..." 등)
  실제 시스템에서는 에이전트가 질의를 생성하지만, 여기서는 질의 품질 차이를 없애고 crop 조건별
  임베딩이 정답 장면을 얼마나 잘 찾는지만 본다(EgoLifeQA 에서 키워드 질의로 채점한 것과 같은 역할).
  현재 장면 묘사(current_observation)는 질의로 쓰지 않는다(정답과 시각적으로 닮아야만 찾아지는
  약한 질의였다). link_reason 은 현재 상황이 섞여 있어 쓰지 않는다. current_observation /
  current_local_context 는 gaze_cases 에서 보여 주려고 항목에 남겨 둔다.

pool 의 questions 원소는 gaze_crop/build_pool.py 와 같은 필드를 가진다. recall_eval.py 의
--query-source question 이 "question" 필드를 질의로 쓴다.

    python experiments/egoserve_gaze_crop/build_egoserve_pool.py
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GC = os.path.join(HERE, "..", "gaze_crop")
sys.path.insert(0, GC)
sys.path.insert(0, os.path.join(HERE, "..", "visual_bottleneck"))
from common import ts_int  # noqa: E402
from gaze_common import POOL_PATH  # noqa: E402

DATA = os.path.join(HERE, "..", "..", "data", "EgoServe", "EgoLife", "A1_JAKE")
POOL_ALL = os.path.join(GC, "pool_all.json")
_WIN = re.compile(r"(DAY\d+)\s+(\d{2}):(\d{2}):(\d{2})-(\d{2}):(\d{2}):(\d{2})")


def parse_window(win):
    """'DAY1 17:25:51-17:25:53' -> ('DAY1', '17255100', '17255300')"""
    m = _WIN.match(win)
    if not m:
        raise ValueError(win)
    day, h, mi, s, h2, mi2, s2 = m.groups()
    return day, f"{h}{mi}{s}00", f"{h2}{mi2}{s2}00"


def gap_bin(cur_day, past_day):
    g = int(cur_day[3:]) - int(past_day[3:])
    return "0" if g <= 0 else "1" if g == 1 else "2+"


def gap_seconds(cday, cs, pday, pe):
    """현재 장면 시작 - 과거 장면 끝 (초). 날짜가 다르면 하루를 86400초로 친다."""
    def sec(day, t):
        return int(day[3:]) * 86400 + int(t[0:2]) * 3600 + int(t[2:4]) * 60 + int(t[4:6])
    return sec(cday, cs) - sec(pday, pe)


NEAR_SEC = 120     # task_reminder 의 "방금 전" 기준
FAR_SEC = 600      # long_term 연결의 "아주 오래 전" 기준

# 그룹 = 떠올려야 할 과거 장면의 성격
#   own_action      내가 한 행동을 기억해야 하는 것 (memory_recall, 근거 i_do_steps)
#   heard_speech    누가 한 말을 기억해야 하는 것 (memory_recall, 근거 speakers_say). 화면에 답이 없을 수 있다
#   just_before     할 일 상기인데 과거가 2분 이내 (task_reminder)
#   minutes_before  할 일 상기인데 과거가 2분 이상 (task_reminder)
#   long_ago        10분 이상 전 또는 다른 날의 기억과 연결 (long_term linked_past_events)
GROUPS_VISUAL = ("own_action", "long_ago")          # 시각으로 찾을 수 있고 충분히 먼 것
GROUPS_OTHER = ("heard_speech", "just_before", "minutes_before")


def group_of(x):
    if x["type"] == "long_term_link":
        return "long_ago"
    if x["type"] == "task_reminder":
        return "just_before" if x["gap_sec"] < NEAR_SEC else "minutes_before"
    return "own_action" if x["past_source"] == "i_do_steps" else "heard_speech"


def make_entry(eid, kind, cur, past_win, past_source, query):
    cday, cs, _ = parse_window(cur["current_time_window"])
    pday, ps, pe = parse_window(past_win)
    return {
        "gap_sec": gap_seconds(cday, cs, pday, pe),
        "past_source": past_source,
        "ID": eid,
        "type": kind,
        # 과거 장면의 근거가 대화(speakers_say)면 오디오가 필요할 가능성이 높다. 근사 표기.
        "need_audio": past_source == "speakers_say",
        "gap_bin": gap_bin(cday, pday),
        "question": query,
        "keywords": "",
        "assistant_utterance": next((d["utterance"] for d in cur.get("proactive_dialogue", []) if d.get("role") == "assistant"), ""),
        "current_observation": cur.get("current_observation", ""),
        "current_local_context": cur.get("current_local_context", ""),
        # 질의 시각 = 현재 장면 시작. 그 이전에 끝난 클립만 후보가 된다(미래 차단).
        "query_time": ts_int(cday, cs),
        # 과거 장면의 시작·끝을 각각 한 시점으로 둔다. 어느 쪽이든 겹치는 클립이 정답.
        "target_time": {"date": pday, "time_list": [ps, pe]},
        "target_keys": [],
        "past_window": past_win,
        "current_window": cur["current_time_window"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=POOL_ALL, help="6,223 클립 풀(gaze_crop). 클립 목록만 가져온다")
    ap.add_argument("--out", default=os.path.join(HERE, "pool_egoserve.json"))
    args = ap.parse_args()

    base = json.load(open(args.pool))
    entries = []
    # 원본 recall_id / link_id 는 파일 안에서 중복되므로(erecall_001 이 두 번 나옴) 순번으로 유일한 ID 를 만든다.
    for i, e in enumerate(json.load(open(os.path.join(DATA, "episodic.json")))):
        eid = f"ep_{i:03d}_" + (e.get("recall_id") or e.get("reminder_id") or "")
        kind = e.get("recall_type") or "task_reminder"
        entries.append(make_entry(eid, kind, e, e["past_time_window"], e["past_supporting_source"],
                                  e["past_observation"].strip()))
    for j, e in enumerate(json.load(open(os.path.join(DATA, "long_term.json")))):
        for i, l in enumerate(e.get("linked_past_events", [])):
            eid = f"lt_{j:03d}_{e.get('link_id') or e.get('trigger_id') or ''}_{i}"
            entries.append(make_entry(eid, "long_term_link", e, l["past_time_window"], "unknown",
                                      l["memory_key"].strip()))

    # recall_eval.py 는 문항 ID 를 정수로 정렬한다(key=int). 사람이 읽는 ID 는 source_id 로 남기고
    # ID 는 "1".."N" 으로 다시 매긴다.
    for n, x in enumerate(entries, 1):
        x["source_id"] = x["ID"]
        x["ID"] = str(n)
        x["group"] = group_of(x)

    pool = {k: v for k, v in base.items() if k != "questions"}
    pool["questions"] = entries
    pool["note"] = ("EgoServe episodic + long_term(linked_past_events). question = description of the target "
                    "past scene (episodic: past_observation, long_term: memory_key). Clips and embeddings "
                    "are gaze_crop's pool_all.")
    json.dump(pool, open(args.out, "w"), ensure_ascii=False, indent=1)
    kinds = {}
    for x in entries:
        kinds[x["type"]] = kinds.get(x["type"], 0) + 1
    print(f"wrote {args.out}: {len(entries)} queries, {pool['n_clips']} clips")
    print("by type:", kinds)
    groups = {}
    for x in entries:
        groups[x["group"]] = groups.get(x["group"], 0) + 1
    print("by group:", groups)


if __name__ == "__main__":
    main()
