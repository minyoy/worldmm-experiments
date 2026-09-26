# Visual Bottleneck 진단 실험 (WorldMM, EgoLifeQA A1_JAKE)

> **한 줄 요약.** 정답 구간의 프레임을 직접 주면 답이 좋아진다(D−B **+9.2 %p**, 텍스트 검색이 놓친 문항에서는 **+12.2 %p**).
> 그런데 실제 WorldMM에서는 그 이득이 거의 사라진다(E−B **+0.9 %p**). 원인은 두 겹이다.
> **(1) 에이전트가 visual 검색을 거의 부르지 않고(2/120), (2) 강제로 불러도 target 클립을 top-3에 넣는 비율이 5%다.**
> 모델이 시각 근거를 못 쓰는 것(utilization)보다는 **찾아오지 못하는 것(retrieval)** 이 병목이다.
> 다음 실험인 [`../gaze_crop`](../gaze_crop/README.md)은 (2)의 원인 중 하나로 "프레이밍"을 검증한다.

설계 전문은 [`PLAN.md`](PLAN.md), 숫자 원본은 [`analysis/summary.md`](analysis/summary.md)에 있습니다.

---

## 1. 왜 이 실험을 했나

WorldMM 논문은 episodic + semantic 메모리에 visual 메모리를 더하면 성능이 오른다고 보고합니다
(E+S < E+S+V). 하지만 **오르는 폭이 작습니다.** 작은 이유는 서로 다른 두 가지일 수 있고, 어느 쪽이냐에
따라 고쳐야 할 곳이 완전히 다릅니다.

| 가설 | 뜻 | 맞다면 고칠 곳 |
|---|---|---|
| **Retrieval 문제** | visual 메모리가 정답에 필요한 클립을 찾아오지 못한다 | 메모리 구축과 검색 (예: gaze-aware visual memory) |
| **Utilization 문제** | 정답 클립을 줘도 응답 MLLM이 그 안의 시각 근거를 활용하지 못한다 | 모델 입력 (예: gaze-guided MLLM input) |

논문 설정 그대로는 두 가지가 섞여 있어서 구분할 수 없습니다. 그래서 **검색을 완벽하게 대체한 조건(oracle
visual)** 을 만들어, 검색이 성공했다면 얻었을 이득과 실제로 얻은 이득을 나란히 비교했습니다.

## 2. 실험 설계

### 조건

| 조건 | 입력 | 역할 |
|---|---|---|
| **A** | 질문만 | 사전지식 / 찍기 기준선 |
| **B** | 질문 + 텍스트 메모리 (episodic + semantic, visual 끔) | 논문의 E+S |
| **C** | 질문 + oracle 프레임 | 시각 근거만으로 풀 수 있나 |
| **D** | 질문 + B의 텍스트 컨텍스트 + oracle 프레임 | 텍스트가 있을 때 시각 근거가 더해 주는 것 |
| **E** | 원본 WorldMM (에이전트가 visual 검색 여부를 스스로 결정) | 논문의 E+S+V |
| **E′** | B의 텍스트 컨텍스트 + 질문으로 강제 검색한 visual top-3 | 텍스트를 고정하고 visual만 retrieved로 바꾼 통제 비교 |

- **Oracle 프레임**: 문항의 `target_time`을 30초 클립에 매핑하고, 그 클립에서 1 fps로 뽑은 프레임입니다
  (최대 64장, 실측 15~63장, 평균 32.7장). 모든 조건이 디스크에 캐시된 같은 JPEG를 씁니다.
- **텍스트 고정**: B를 한 번 돌려 문항별 `round_history`와 `retrieved_items`를 `text_context/{qid}.json`에
  저장하고, D와 E′가 이를 그대로 재사용합니다. 그래서 D−B와 E′−B는 순수하게 visual을 추가한 효과입니다.
- 비교는 모두 **같은 문항끼리 짝지은 paired 비교**이고, 유의성은 exact McNemar(양측)로 봤습니다.

### 모델과 메모리 (WorldMM-8B 설정)

| 구성요소 | 모델 | 출처 |
|---|---|---|
| Episodic 메모리 OpenIE (30sec / 3min / 10min / 1h) | GPT-5-mini | 30sec는 저자 배포본 시딩, 나머지 3개 스케일은 [`build_episodic_cache.py`](build_episodic_cache.py)로 직접 생성 |
| Semantic 메모리 | GPT-5-mini | 저자 배포본 (`output/metadata`) |
| Visual 임베딩 | VLM2Vec-V2.0 | 저자 배포본 |
| Retrieval agent / Response agent | **Qwen3-VL-8B-Instruct** (인스턴스 하나 공유) | 로컬 |

3min / 10min / 1h의 OpenIE 결과는 배포본에 없었습니다. 처음에는 Qwen으로 생성하려 했는데, 1h 캡션(평균
9,588자)에서 같은 단어를 반복 생성하다 JSON이 깨지며 **67%가 빈 개체**로 저장됐습니다. 그래서 논문 설정대로
GPT-5-mini로 새로 만들었고, 그 결과 빈 개체 비율은 10min과 1h 모두 0%였습니다.

### 데이터

A1_JAKE 500문항 중 oracle을 정의할 수 있는 478문항에서, **유형당 24개씩 총 120문항**을 층화 추출했습니다
(유형 × `need_audio` × query–target 일수 간격). 이 중 `need_audio=False`(시각 근거 문항)가 70개,
`True`(대화 근거 문항)가 50개입니다. 자세한 분포는 PLAN.md 3장에 있습니다.

## 3. 결과

### 3.1 정확도

| 조건 | 설명 | 정확도 (%) |
|---|---|---|
| A | 질문만 | 29.2 |
| B | 텍스트 메모리 | 40.8 |
| C | Oracle 프레임만 | 36.7 |
| **D** | **텍스트 + oracle 프레임** | **50.0** |
| E | 텍스트 + retrieved visual (원본 WorldMM) | 41.7 |
| E′ | B 텍스트 + retrieved visual (강제) | 45.8 |

### 3.2 핵심 비교 (paired, n=120)

| 비교 | 의미 | Δ (%p) | 좋아짐 / 나빠짐 | p |
|---|---|---|---|---|
| C − A | 시각 근거만으로 도움이 되나 | +7.5 | 19 / 10 | 0.136 |
| **D − B** | **텍스트 위에 oracle visual이 더하는 것** | **+9.2** | 19 / 8 | 0.052 |
| C − B | 어느 modality가 더 쓸 만한가 (진단용) | −4.1 | 22 / 27 | 0.568 |
| **E − B** | **텍스트 위에 retrieved visual이 더하는 것 (논문의 E+S+V − E+S)** | **+0.9** | 10 / 9 | 1.000 |
| D − E | retrieval bottleneck 크기 | +8.3 | 20 / 10 | 0.099 |
| E′ − B | 텍스트 고정 시 retrieved visual의 이득 | +5.0 | 11 / 5 | 0.210 |
| D − E′ | 텍스트 고정 시 oracle과 retrieved의 차이 | +4.2 | 11 / 6 | 0.332 |

### 3.3 Oracle visual은 어디서 도움이 되나

| 부분집합 | n | C − A | D − B |
|---|---|---|---|
| need_audio = False (시각 근거) | 70 | **+14.3** (p=0.031) | +11.4 (p=0.077) |
| need_audio = True (대화 근거) | 50 | −2.0 | +6.0 |
| EntityLog | 24 | +20.8 | **+25.0** (p=0.070) |
| RelationMap | 24 | +4.2 | +12.5 |
| EventRecall | 24 | +12.5 | 0.0 |
| TaskMaster | 24 | −8.3 | 0.0 |
| HabitInsight | 24 | +8.3 | +8.3 |
| gap 0 (당일) | 57 | +7.0 | +1.8 |
| **gap 1 (하루 전)** | 33 | +15.2 | **+21.2** (p=0.039) |
| gap 2+ | 30 | 0.0 | +10.0 |

**텍스트 검색이 target을 놓친 문항에서 더 크게 오릅니다.** B가 가져온 캡션 중 10분 이하 스케일의 어느 것도
target 시점을 덮지 못한 문항이 **82/120**이고, 여기서 D−B는 32.9 → 45.1 (**+12.2 %p**)입니다.
(기준을 "어느 스케일이든 덮지 못함"으로 바꾸면 64문항에서 +14.1 %p, p=0.035)

### 3.4 Retrieval 진단: 왜 E에서 이득이 사라졌나

| | E (에이전트 자율) | E′ (강제 호출) |
|---|---|---|
| visual 검색을 한 문항 | **2 / 120** | 120 / 120 |
| target 클립이 top-3에 든 비율 (recall@3) | 0 / 2 | **6 / 120 = 5.0%** |
| 정확도 | 41.7 | 45.8 |

- **라우팅 실패**: E에서 Qwen3-VL-8B는 visual 검색을 120문항 중 2문항에서만 호출했습니다. 라운드 결정의
  JSON 파서를 느슨하게 해도(`--robust-reasoning`, `results_robust/E.json`) 2문항 그대로였으므로, 파싱
  문제가 아니라 판단 자체의 문제입니다. 120문항 밖의 나머지 380문항에서도 결과가 같았습니다
  (`results_rest/E.json`: 정확도 44.2%, visual 호출 **3/380**).
- **검색 품질 실패**: 호출을 강제해도 target 클립은 5%만 top-3에 들어왔습니다. 그런데도 E′는 B보다
  +5.0 %p 높습니다. target을 거의 못 맞혔는데 정확도가 올랐으므로, 이 이득은 "정답 근거를 찾아와서"라기보다
  **근처 시각 컨텍스트가 있는 것 자체의 효과**일 가능성이 큽니다.

### 3.5 텍스트 검색 쪽 진단 ([`dpr_recall.py`](dpr_recall.py))

같은 문제가 텍스트 episodic 검색에도 있는지 확인했습니다. 에이전트가 실제로 낸 episodic 질의를, OpenIE /
그래프 / PPR / LLM 필터를 모두 빼고 **같은 텍스트 인코더의 단순 cosine**으로 30초 캡션에 대해 다시 돌렸습니다
(episodic 질의가 있었던 96문항).

| 검색 방식 | 30초 target 적중 |
|---|---|
| HippoRAG (그래프 + PPR, 30sec top-10) | 34 / 96 |
| DPR, 에이전트 질의 | @3 22 · @10 33 · @50 51 · @100 55 |
| DPR, 질문 원문 | @3 20 · @10 37 · @50 51 · @100 56 |
| DPR, target 캡션 자신 (sanity) | 96 / 96 |

HippoRAG와 단순 DPR의 성능이 비슷합니다(34 vs 33, @10). 그러니 그래프 단계가 target을 잃는 게 아니라,
**질의와 인코더 자체가 한계**입니다. 다만 @50~@100에서는 절반 이상이 잡히므로 target에 닿을 수는 있고,
top-k가 좁은 것도 원인의 일부입니다.

## 4. 해석

PLAN.md 1장의 판정표에 대입하면:

| Oracle visual gain (D−B) | Retrieved visual gain (E−B) | 판정 |
|---|---|---|
| **+9.2** (p=0.052, 텍스트 실패 문항 +12.2) | **+0.9** (p=1.0) | **큼 / 작음 → retrieval bottleneck** |

1. **Utilization은 (시각 문항에서는) 작동합니다.** 정답 프레임을 주면 시각 근거 문항에서 +14.3 %p(프레임만),
   EntityLog에서 +25 %p, 하루 전 사건에서 +21 %p가 오릅니다. 모델은 시각 근거를 쓸 줄 압니다.
2. **병목은 retrieval이고, 두 층입니다.**
   - (a) **라우팅**: 에이전트가 visual 검색을 부를 생각을 하지 않습니다 (2/120, 나머지 380문항에서 3/380).
   - (b) **검색 품질**: 불러도 target을 거의 찾지 못합니다 (recall@3 5%).

   둘은 서로 독립이라, (b)만 고쳐서는 E의 최종 점수가 움직이지 않습니다. 그래서 검색 개선 효과는 (a)를
   제거한 E′를 기준으로 재야 합니다.
3. **Text dominance는 뚜렷하지 않습니다.** C−A(+7.5)가 크고 D−B가 작은 패턴이 아니라, D−B(+9.2)가 오히려
   조금 더 큽니다. 텍스트가 있어도 시각 근거가 더해질 여지가 남아 있습니다.
4. **시각이 도울 수 있는 천장이 있습니다.** 대화 근거 문항(50/120)에서는 oracle 프레임만으로 −2.0 %p이고,
   EventRecall과 TaskMaster는 D−B가 0입니다. 이 구간은 visual을 아무리 잘 찾아도 오르지 않습니다.
   당일 문항(gap 0)도 텍스트가 이미 잘 잡고 있어 +1.8 %p에 그칩니다.

**후속 방향.** 판정표대로면 "gaze-aware visual memory"(검색 쪽 개선)입니다. 다만 위 2(a) 때문에, 검색 개선
효과는 E가 아니라 E′ 기준으로 측정해야 하고, 라우팅은 프롬프트나 정책 문제로 따로 다뤄야 합니다.

## 5. 한계와 주의사항

- **표본이 작습니다.** n=120이면 5 %p가 문항 6개입니다. 핵심 비교 중 p<0.05는 부분집합 몇 개뿐이고,
  전체 D−B(0.052)와 D−E(0.099)는 경계선입니다. 방향은 일관되지만 효과 크기는 넓게 읽어야 합니다.
- **피험자 1명(A1_JAKE), 응답 모델 1개(Qwen3-VL-8B).** 더 큰 모델에서는 라우팅 행태가 다를 수 있습니다.
- **D와 E′의 프레임 예산이 다릅니다.** D는 평균 33장, E′는 중앙값 63장입니다(top-3 클립 × 1 fps).
  D−E′를 "순수 검색 차이"로 읽을 때 이 점을 감안해야 합니다.
- **Episodic 캐시 출처가 섞여 있습니다.** 30sec는 저자의 GPT-5-mini 결과, 3min/10min/1h는 우리가 만든
  GPT-5-mini 결과입니다. 모든 조건이 같은 캐시를 공유하므로 조건 간 비교는 유효하지만, 논문 수치와 절대값이
  다를 수 있습니다.
- **Exp.3 (visual masking)은 아직 돌리지 않았습니다.** "오른 이득이 실제로 target 영역을 보고 얻은 것인가"는
  아직 검증되지 않았습니다.
- **결론에서 제외한 실행**
  - `results/B_leaky.json` (35.8%): 전체 주간 인덱스가 미리 만들어진 상태로 시작해서, 초반 문항에 미래 캡션이
    새어 들어간 실행입니다. 이후 [`eval_egolife.py`](eval_egolife.py)가 시작할 때 인덱스를 비우도록
    고쳤습니다(`reset_episodic_index`). OpenIE 캐시는 그대로 유지됩니다.
  - `results/E_novisual.json` (41.7%): visual 유사도 검색이 항상 0프레임을 돌려주던 버그를 고치기 전의
    E입니다. E와 점수가 같은데, visual을 2문항에서만 부르니 버그 유무가 결과에 거의 영향을 주지 않았습니다.

## 6. 다음 실험: [`gaze_crop`](../gaze_crop/README.md)

**질문.** retrieval 실패(recall 5%)의 일부는 **프레이밍 실패**가 아닐까? 지금은 30초 클립 하나를 전체 화면
16프레임으로 보고 임베딩 하나로 압축합니다. 그러면 질문과 무관한 배경이 임베딩을 지배할 수 있습니다.
착용자가 **실제로 보고 있던 곳(eye gaze)** 중심으로 잘라서 임베딩하면 검색이 나아질까?

**이 실험에서 이어받는 것**
- 헤드룸: 텍스트 검색이 놓친 82문항에서 oracle visual +12.2 %p. gaze crop이 되찾으려는 몫입니다.
- 기준선: E′ 45.8% (recall@3 5%). 2(a)의 라우팅 문제를 피하려고 **E가 아니라 E′ 위에서** 잽니다.
  넘어야 할 선은 B의 40.8%가 아니라 45.8%입니다.
- 천장: 대화 근거 문항에서는 visual이 도움이 안 되므로 recall을 `need_audio`로 나눠 봅니다.

**설계 요약**

| 단계 | 재는 것 | 비교 | 풀 |
|---|---|---|---|
| 1단계 | visual retrieval recall@k (LLM 없음) | `gaze@0.5` vs `full`, 대조군 `center@0.5`와 random crop | 623클립 (target 123 + 같은 날 distractor 500) |
| 게이트 | 1단계에서 `gaze − full > 0`일 때만 진행 | | |
| 2단계 | E′ 정확도 (visual 인덱스만 교체) | `gaze@0.5` vs `full` | 전체 6,223클립 |

`center@0.5`는 "crop 자체의 효과"와 "gaze의 효과"를 구분하기 위한 대조군입니다. 1인칭 영상은 중심 편향이
강해서, 중앙만 잘라도 대상이 커지고 배경이 빠지기 때문입니다.

| 1단계 결과 | 해석 |
|---|---|
| `gaze` > `center` > `full` | gaze가 crop 이상으로 기여 → 2단계로 |
| `gaze` ≈ `center` > `full` | crop만 효과, gaze는 아님 |
| 모두 ≈ `full` | 프레이밍은 병목이 아님 → 인코더나 질의 쪽을 봄 (3.5의 DPR 진단과 연결) |

**진행 상황.** 2026-09-26 01:17에 `gpu2`에서 [`run_overnight.sh`](../gaze_crop/run_overnight.sh)로
1단계 → 게이트 → 2단계를 무인 실행하기 시작했습니다(CPU 2코어로 제한). 결과는
`../gaze_crop/analysis/DIGEST.md`에 정리됩니다.

## 7. 재현

저장소 루트에서 실행합니다. 전체 절차는 PLAN.md 5장에 있습니다.

```bash
# 준비 (1회)
python experiments/visual_bottleneck/build_subset.py
python experiments/visual_bottleneck/seed_hipporag_cache.py        # 30sec OpenIE 시딩
python experiments/visual_bottleneck/build_episodic_cache.py       # 3min/10min/1h OpenIE (GPT-5-mini, OPENAI_API_KEY 필요)
python experiments/visual_bottleneck/extract_oracle_frames.py

# 조건 실행 (B가 먼저: D, E'가 B의 text_context를 쓴다)
for c in B A C D E E_prime; do
  python experiments/visual_bottleneck/eval_egolife.py --condition $c --resume
done

# 집계
python experiments/visual_bottleneck/analyze.py      # -> analysis/summary.md, summary.json
python experiments/visual_bottleneck/dpr_recall.py   # -> analysis/dpr_recall*.json
```

## 8. 파일

| 경로 | 내용 |
|---|---|
| `PLAN.md` | 실험 설계 전문 (배경, 조건, 판정 기준, 실행 프로토콜) |
| `subset.json` / `subset_rest.json` | 120문항 main set / 나머지 380문항 (E 재현용) |
| `eval_egolife.py` | 모든 조건의 러너 (`--condition A~E, E_prime`) |
| `build_episodic_cache.py`, `seed_hipporag_cache.py` | episodic OpenIE 캐시 구축 |
| `extract_oracle_frames.py` | `frames/` (1 fps, 최대 64장), `frames16/` (Exp.3용 16장) |
| `analyze.py` | 정확도 표, paired McNemar, 부분집합 분해, retrieval 진단 |
| `dpr_recall.py` | 텍스트 episodic 검색의 HippoRAG vs 단순 DPR 비교 |
| `make_masks.py`, `apply_masks.py` | Exp.3 마스킹 (미실행) |
| `results/` | 조건별 문항 단위 결과 |
| `results_robust/`, `results_rest/` | E의 lenient 파서 실행 / 나머지 380문항 실행 |
| `text_context/` | 조건 B의 문항별 텍스트 컨텍스트 (D, E′가 재사용) |
| `analysis/` | `summary.md`, `summary.json`, `dpr_recall*.json` |
