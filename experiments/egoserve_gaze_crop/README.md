# EgoServe gaze-crop 검색 실험 (A1_JAKE)

[`gaze_crop`](../gaze_crop/README.md)에서 본 `full` / `center` / `randf` / `gazef` 4개 조건을 EgoServe 이벤트에 그대로 적용해
검색 성능을 비교한다. EgoServe는 EgoLife와 같은 영상·gaze를 쓰고 주석만 다르므로, **클립 6,223개와 4개 조건의 임베딩은
`../gaze_crop/emb/`를 그대로 쓴다.** 새로 만드는 것은 질의와 정답뿐이다.

> **요약.** 88개 질의 전체로는 조건 간 차이를 말하기 어렵다. 하지만 **다른 날이나 10분 이상 전의 기억(`long_ago`, 32개)** 에서는
> `gazef`가 `full`보다 정답 순위가 일관되게 좋았다(tol 0/30/60에서 22:9, 24:7, 26:5, p=0.002~0.01). 다만 `center`도 `full`보다
> 나은 경향이라 시선 덕인지 자르기 덕인지는 가르지 못했다. 대화가 근거이거나 과거가 너무 가까운 그룹에서는 도움이 안 됐다.
> **이 결과는 질의 생성 능력을 뺀, 임베딩의 검색 능력만 잰 것이다**(아래 한계 참고).

## 데이터: QA가 아니라 proactive 이벤트

`data/EgoServe/EgoLife/A1_JAKE/`의 이벤트는 "지금 장면"에서 "과거 장면"을 떠올려 어시스턴트가 먼저 말을 거는 형식이다. 질문과 정답이
있는 QA가 아니어서 검색 정답을 이렇게 정의했다.

| 파일 | 이벤트 | 검색 정답 |
|---|---|---|
| `episodic.json` | 56 | `past_time_window`의 과거 장면 |
| `long_term.json` | 117 중 30 | `linked_past_events`의 과거 장면 (32개) |
| `instant.json`, `short_term.json` | 70, 121 | 과거 장면 지시가 없어 제외 |
| `long_term.json`의 `habit_coaching`, `routine_optimization` | 87 | 여러 날의 패턴이라 특정 과거 장면이 없어 제외 |

- 질의 88개 = episodic 56 + long_term 연결 32. **`A1_JAKE` 분량뿐**이다(EgoServe는 `A4_LUCIA`, `A5_KATRINA`도 있지만 그 영상·gaze가
  이 저장소에 없다).
- **질의 = 정답 장면의 텍스트 묘사.** episodic은 `past_observation`, long_term은 `memory_key`. 질의 품질을 이상적으로 고정한 상한선이다.
  (현재 장면 묘사 `current_observation`으로도 채점해 봤는데 과거 장면과 시각적으로 닮아야만 찾아지는 약한 질의라 버렸다.)
- 정답 클립: 과거 장면의 시작·끝 시각과 겹치는 클립. 후보는 현재 장면 시작 이전에 끝난 클립(미래 차단). tol 0/30/60초.
- tol 0에서는 정답이 질의 시각 이전에 없는 2개가 빠져 86개가 채점된다.

## 그룹: 떠올려야 할 과거 장면의 성격

| 그룹 | n | 과거 장면 |
|---|---|---|
| `own_action` | 12 | 내가 **한 행동** (memory_recall, 근거 i_do_steps) |
| `heard_speech` | 7 | 누가 **한 말** (memory_recall, 근거 speakers_say). 화면에 답이 없을 수 있어 사례 보기에서 제외 |
| `just_before` | 29 | 할 일 상기, 과거가 **2분 이내** |
| `minutes_before` | 8 | 할 일 상기, 과거가 2분 이상 |
| `long_ago` | 32 | **10분 이상 전 또는 다른 날**의 기억과 연결 (long_term) |

`visual_findable` = `own_action` + `long_ago`(시각으로 찾을 수 있고 충분히 먼 것, 44개), `other` = 나머지.

## 결과

`analysis/tables/`에 원본이 있다 (`recall_egoserve_pastobs_tol*.md`, 그룹별 `subsets_pastobs.md`).

**전체 88개, recall (%), tol 30**

| 조건 | R@3 | R@10 | R@20 | 순위 중앙값 |
|---|---|---|---|---|
| `full` | 4.5 | 13.6 | 23.9 | 152 |
| `center` | 8.0 | 17.0 | 21.6 | 126 |
| `randf` | 6.8 | 11.4 | 20.5 | 121 |
| `gazef` | 6.8 | 22.7 | 26.1 | 110 |

R@3은 88개 중 3~7개를 맞힌 것이라 의미가 없다. 조건 간 비교는 정답 순위를 문항별로 비교한다.

**`gazef` 대 `full`, 정답 순위가 더 좋은 질의 : 더 나쁜 질의 (Wilcoxon p)**

| 그룹 | n | tol 0 | tol 30 | tol 60 |
|---|---|---|---|---|
| 전체 | 86~88 | 48:37 (0.31) | 52:34 (0.03) | 52:35 (0.05) |
| `long_ago` | 32 | 22:9 (0.01) | 24:7 (0.004) | 26:5 (0.002) |
| `own_action` | 12 | 6:6 | 5:7 | 6:6 |
| `other` | 42~44 | 20:22 | 23:20 | 20:24 |

- `long_ago`는 같은 질의 문장이 최대 5번 반복되어 독립 표본이 32보다 적다. 질의 문장 19개로 줄여도 `gazef > full`이 유지된다(14:4, 14:5, 16:3, p=0.005~0.018).
- `center`, `randf`도 `long_ago`에서 `full`보다 나은 경향이다(tol 60에서 24:8, 20:10). `gazef > center`는 tol 0에서만 유의(24:8, p=0.025).
- `own_action`(12개)과 `heard_speech`(7개)는 표본이 너무 작아 판단할 수 없다.
- 그룹·tol·비교마다 검정을 여러 번 했고 다중비교 보정을 하지 않았다. 사후에 나눈 그룹이라 가설을 만드는 용도로만 읽는다.

## 사례 보기 (`egoserve_cases/`)

`build_egoserve_cases.py`가 `gaze_crop`의 gaze_cases와 같은 형태의 정적 페이지를 만든다(`heard_speech` 제외, tol 30 순위 기준).
- **A**: `gazef`의 정답 순위가 `full`보다 절반 이하로 줄어든 질의 23개
- **B**: 반대로 `full`이 크게 좋은 질의 15개

각 사례는 질의, 지금 장면 묘사와 프레임, 어시스턴트 발화, 정답 클립의 16프레임(시선 점, `gazef` crop 박스), tol 0/30/60 순위를 보여 준다.
판정(도움 됨/아님/모름)과 메모는 브라우저에 저장되고 JSON으로 내보낼 수 있다. 영상 프레임이 들어 있어 git에서 제외한다.

```bash
# 영상과 gaze 가 있는 서버에서 (CPU 만 사용)
python experiments/egoserve_gaze_crop/build_egoserve_cases.py
scp -r server:.../experiments/egoserve_gaze_crop/egoserve_cases .  &&  open egoserve_cases/index.html
```

## 한계

- **질의 생성 능력을 뺐다.** 이 벤치마크에서는 상황을 보고 무엇을 검색할지 질의를 만드는 것이 핵심 능력인데, 여기서는 정답 장면 묘사를
  질의로 고정했다. 실제 시스템의 검색 성능이 아니라 crop 조건별 임베딩이 정답 장면을 얼마나 잘 찾는지만 본 것이다. 모델이 질의를 만드는
  설정은 아직 하지 않았다.
- 표본이 작다(88개, 그룹은 7~44개). `long_ago`의 신호는 tol 30·60에서 더 뚜렷하고 tol 0(정확한 클립)에서는 약하다.
- `long_ago`의 질의는 `memory_key`(기억 요약)라 장면 묘사와 성격이 다르다. episodic 질의와 한 표에 묶을 때 주의한다.
- 자르는 것 자체의 효과와 시선 위치의 효과를 구분하지 못했다(`center` 대비 차이가 작음).
- `A1_JAKE` 한 사람의 분량이다.

## 재현

```bash
cd ~/WorldMM/experiments/egoserve_gaze_crop
python build_egoserve_pool.py                 # pool_egoserve.json (질의 88개, 클립은 gaze_crop/pool_all.json)
CUDA_VISIBLE_DEVICES=1 PY=/path/to/.venv/bin/python bash run_recall.sh   # 4개 조건 x tol 0/30/60 채점 (질의 임베딩에 GPU)
python analyze_subsets.py                     # 그룹별 표 (analysis/tables/subsets_pastobs.md)
python build_egoserve_cases.py                # 사례 보기 페이지
```

## 파일

| 경로 | 내용 |
|---|---|
| [`build_egoserve_pool.py`](build_egoserve_pool.py) | EgoServe 이벤트 → 검색 질의·정답 (`pool_egoserve.json`), 그룹 정의 |
| [`run_recall.sh`](run_recall.sh) | `../gaze_crop/recall_eval.py`로 4개 조건 채점 |
| [`analyze_subsets.py`](analyze_subsets.py) | 그룹별 recall과 `gazef` 대 다른 조건 순위 비교 |
| [`build_egoserve_cases.py`](build_egoserve_cases.py), `egoserve_cases_template.html` | 사례 보기 페이지 (`egoserve_cases/`, git 제외) |
| `results/recall_egoserve_pastobs_tol{0,30,60}.json` | 채점 원본 (문항별 순위 포함) |
| `analysis/tables/` | recall 표, 그룹별 표 |
