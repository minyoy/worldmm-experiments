# Visual Bottleneck 진단 실험 계획

이 문서 하나로 실험의 배경, 설계, 데이터, 구현, 실행 순서, 해석 기준을 모두 담는다.
이 폴더에는 실험용 스크립트와 산출물이 들어가며, 코드베이스 본체(`src/worldmm`)는 최소한만 수정한다.

---

## 0. 배경과 문제 의식

WorldMM 논문에서 이미 알려진 사실:

```
E+S  <  E+S+V      (Episodic + Semantic  <  Episodic + Semantic + Visual)
```

즉 retrieved visual이 전체적으로 어느 정도는 도움이 된다. 그런데 **그 이득이 작다.** 원인은 둘 중 하나이거나 둘 다일 수 있다.

1. **Retrieval 문제** — visual memory가 정답에 필요한 클립을 못 찾아온다.
2. **Utilization 문제** — 정답 클립을 줘도 응답 MLLM이 그 안의 visual evidence를 못 쓴다.

이 둘을 분리하기 위해 세 실험을 한다.

| 실험 | 비교 | 답하려는 질문 |
|---|---|---|
| Exp.1 Oracle Visual | Text only vs Text + Oracle visual | retrieval을 완벽하게 했을 때 text 대비 gain이 얼마나 커지는가 |
| Exp.2 Retrieved Visual | Text + Retrieved visual vs Text + Oracle visual | retrieval bottleneck의 크기 |
| Exp.3 Visual Masking | Original vs Relevant-mask vs Irrelevant-mask oracle | 응답 MLLM이 실제로 task-relevant visual evidence에 의존하는가 |

결론에 따른 후속 방향:

- **retrieval 문제** → gaze-aware visual memory 구축 (gaze crop, region feature, object-centric memory)
- **utilization 문제** → gaze-guided MLLM input (gaze heatmap 등)
- **둘 다** → gaze를 메모리 구축과 입력 양쪽에 쓰는 end-to-end 구조

---

## 1. 판정 기준

핵심 조건 다섯 개 (자세한 정의는 2장):

```
A  Question only
B  Question + Text memory
C  Question + Oracle visual
D  Question + Text memory + Oracle visual
E  Question + Text memory + Retrieved visual   (= 원본 WorldMM)
```

| Oracle visual gain (D−B) | Retrieved visual gain (E−B) | 해석 | 후속 방향 |
|---|---|---|---|
| 큼 | 큼 | retrieval, utilization 모두 정상 | - |
| 큼 | 작음 | **retrieval bottleneck** | gaze-aware visual memory |
| 작음 | 작음 | **utilization bottleneck** 가능성 | gaze-guided MLLM input |
| C−A 큼, D−B 작음 | - | **text dominance / modality fusion 문제** 가능성 | 둘 다 |

### 해석 예시

**Exp.1에서**

```
A  Question only         35
B  Text memory           60
C  Oracle visual         66
D  Text + Oracle visual  62
```
→ visual만 있을 때는 잘 푸는데 text가 함께 들어오면 visual의 추가 이득이 거의 없다. **text dominance / multimodal fusion 문제** 가능성.

```
A  Question only         35
B  Text memory           60
C  Oracle visual         40
D  Text + Oracle visual  61
```
→ 정답 visual을 직접 줘도 활용을 못 한다. **visual utilization bottleneck** 가능성.

**Exp.2에서**

```
B  Text memory              60
D  Text + Oracle visual     72
E  Text + Retrieved visual  62
```
→ 올바른 visual을 주면 +12인데 실제 retrieval에서는 +2. **retrieval bottleneck이 큼.**

```
B  Text memory              60
D  Text + Oracle visual     62
E  Text + Retrieved visual  61
```
→ oracle retrieval로 바꿔도 차이가 없다. **retrieval보다 utilization이 병목**일 가능성.

`C vs B`는 "visual이 text보다 우수하다"를 판정하는 비교가 아니라, **현재 모델에게 어느 modality가 더 usable한 representation인지 보는 진단**으로만 해석한다.

---

## 2. 실험 조건과 코드 매핑

실험은 논문의 **WorldMM-8B 설정**을 따른다.

| 단계 | 구성요소 | 모델 | 이 실험에서 |
|---|---|---|---|
| Memory Construction | Episodic memory 생성 (캡션, OpenIE) | GPT-5-mini | `caption.zip` + `output/metadata` + `build_episodic_cache.py` (아래) |
| Memory Construction | Semantic memory 생성 | GPT-5-mini | `output/metadata` 그대로 |
| Memory Construction | Visual embedding | VLM2Vec-V2 | `output/metadata` 그대로 |
| Inference | Retrieval agent (질의 NER, multiscale filter) | Qwen3-VL-8B-Instruct | `--retriever-model qwen3vl-8b` |
| Inference | Response agent (라운드 결정, 최종 답변) | Qwen3-VL-8B-Instruct | `--respond-model qwen3vl-8b` |

메모리 구축은 사실상 끝나 있고, 실험은 inference 단계만 돌린다. 프롬프트는 `qa_egolife` 템플릿을 그대로 쓴다. Text memory는 **WorldMM의 episodic/semantic 메모리를 그대로** 쓰고 visual만 교체한다. 그래야 조건 간 나머지가 동일하게 유지된다.

모든 조건은 **`experiments/visual_bottleneck/eval_egolife.py --condition <X>`** 하나로 실행한다. 이 파일은 `eval/eval_egolife.py`의 복사본에 `--condition` 분기를 얹은 것이다 (아래 "구현 방식" 참조). 원본 `eval/eval_egolife.py`는 건드리지 않는다.

| 조건 | 입력 | `--condition` 분기에서 하는 일 |
|---|---|---|
| A | Question only | `world_memory.answer()` 대신 QA 프롬프트에 질문/선택지만 넣어 응답 모델 호출 |
| B | Question + Text memory | reasoning 템플릿에서 Visual 항목을 뺀 버전으로 `answer()` 실행. 결과의 `round_history`·`retrieved_items`를 `text_context/{qid}.json`에 저장 |
| C | Question + Oracle visual | `frames/{qid}/`의 프레임을 QA 프롬프트에 이미지로 넣어 호출 |
| D | Question + Text memory + Oracle visual | `text_context/{qid}.json`의 텍스트 + C의 프레임 |
| E | Question + Text memory + Retrieved visual | 원본 그대로 `answer()` 실행 (분기 없음). 논문의 E+S+V 설정 |
| E′ (보조) | B 컨텍스트 + Retrieved visual | `text_context/{qid}.json`의 텍스트 + `visual_memory.retrieve(question)` top-3 프레임 |
| C1 / D1 | Exp.3 원본 (16장 oracle) | C/D와 같되 `frames16/{qid}`의 고정 16장 사용. 마스킹 조건과 프레임 수를 맞추기 위한 기준선 |
| C2/C3, D2/D3 | 마스킹 oracle | C1/D1과 같되 `masked_frames/{relevant,irrelevant}/{qid}`의 프레임 사용 |

**B/D의 텍스트 컨텍스트를 공유하는 이유**: D−B 차이가 순수하게 visual 추가 효과가 되도록 텍스트 입력을 고정한다. B를 한 번만 돌리고 `round_history`와 `retrieved_items`를 질문별로 저장해 D, E′에서 재사용한다.

**E와 D의 차이는 두 가지가 섞여 있다** (에이전트가 다른 검색 경로를 택함 + visual이 retrieved임). 그래서 E′를 두어 visual만 바꾼 통제 비교를 따로 본다. 주 결과는 D vs E(논문 설정과 정합), 보조로 D vs E′.

### 구현 방식: `experiments/visual_bottleneck/eval_egolife.py`

`eval/eval_egolife.py`를 그대로 복사한 뒤 아래만 추가한다. `src/worldmm`도 6장의 버그 수정 한 건 외에는 손대지 않는다.

1. **인자 추가**: `--condition {A,B,C,D,E,E_prime,C2,C3,D2,D3}`, `--subset subset.json`, `--frames-dir`, `--mask-dir`, `--text-context-dir`, `--semantic-path`(하드코딩 경로 대체).
2. **서브셋 필터**: `eval_data`를 로드한 직후 `subset.json`의 ID로 걸러낸다.
3. **Visual 비활성화 (B)**: `PromptTemplateManager.templates`는 런타임에 덮어쓸 수 있는 dict이다. `memory_reasoning` 템플릿에서 "3. Visual" 항목과 few-shot의 visual 라운드를 뺀 문자열을 만들어 `prompt_template_manager.templates["memory_reasoning"]`에 넣는다. 안전장치로 `world_memory.retrieve_from_visual`을 빈 결과를 돌려주는 함수로 바꿔 끼운다.
4. **QA 직접 호출 헬퍼 (A/C/D/E′)**: `WorldMemory.answer()`의 마지막 부분(QA 프롬프트 조립 → `respond_llm_model.generate`)을 스크립트 안의 함수 `answer_with_context(world_memory, question, choices, text_items, images)`로 옮겨 쓴다. 기존 `_render_retrieved_items_for_qa`와 `qa_egolife` 템플릿을 그대로 재사용하므로 프롬프트 형식이 원본과 동일하다.
5. **루프 안 분기**: 기존 `world_memory.answer(...)` 호출 자리에서 `--condition`에 따라 위 헬퍼 또는 원본 `answer()`를 호출한다. E는 원본 호출 그대로.
6. **결과 필드 추가**: `result_entry`에 `condition, need_audio, gap, num_frames, visual_used, visual_query_kind, retrieved_clips, target_clips`를 더한다. 출력 경로는 `results/{condition}.json`.
7. **인덱싱 시각**: 모든 조건에서 원본과 같이 `until_time=query_time`으로 `world_memory.index()`를 호출한다. A/C처럼 메모리를 안 쓰는 조건도 같은 순서로 돌려 조건 간 실행 경로를 맞춘다.

B → D/E′ 순서 의존성이 있으므로 B를 먼저 돌려 `text_context/`를 채운다.

### Oracle visual 정의

- `target_time`(single / time_list / range)을 `eval_egolife.py`의 `parse_target_time`과 같은 규칙으로 30초 클립에 매핑한다. GT evidence timestamp가 포함된 WorldMM visual segment를 직접 고르는 것.
- 프레임 추출은 WorldMM 검색 경로와 동일하게 **1fps, 총 64장 상한**, 클립이 여러 개면 비례 샘플링. 이렇게 해야 D와 E의 프레임 예산이 같다.
- Exp.3용 서브셋은 마스크가 프레임 단위라 **클립당 고정 16장**을 별도로 추출한다.
- 프레임은 질문별로 디스크에 JPEG로 캐시해 C/D/E′/Exp.3이 **동일한 픽셀**을 쓰게 한다.

### 메모리 소스: `output/metadata` (gpt-5-mini로 사전 구축, 로컬에 있음)

`wgcyeo/WorldMM-EgoLife`를 통째로 받은 상태 (6명분, 약 780 MB). 실험은 A1_JAKE 것만 쓴다. 이 메모리들은 저장소 `caption.zip`의 캡션 텍스트를 기준으로 만들어졌으므로 그 캡션과 반드시 같이 써야 한다 (3장 참조).

| 파일 | 평가 시 사용 방식 |
|---|---|
| `semantic_memory/A1_JAKE/semantic_consolidation_results_gpt-5-mini.json` | 그대로 로드. 타임스탬프별 스냅샷 → PPR 그래프 |
| `visual_memory/A1_JAKE/visual_embeddings.pkl` | 그대로 로드. `video_path` 키 → 1536차원 벡터 |
| `episodic_memory/A1_JAKE/openie_results_gpt-5-mini.json` | **평가 코드가 직접 읽지 않음.** 30초 캡션 6,223개의 NER/트리플이 캡션 텍스트 해시를 키로 저장됨. 캡션 텍스트와 합쳐 HippoRAG 캐시로 변환하면 재사용 가능 |
| `episodic_memory/A1_JAKE/episodic_triple_results_gpt-5-mini.json` | 시맨틱 추출용 중간 산출물. 이번 실험에선 미사용 |

**에피소딕 메모리 주의점**: 평가 시 HippoRAG가 캡션을 받아 OpenIE를 새로 돌려 `.cache/episodic_memory/{granularity}/openie_results_ner_gpt-5-mini.json`에 저장한다. 이 파일명은 retriever 인자와 무관하게 HippoRAG 기본 설정(`llm_name=gpt-5-mini`)에서 나온다. 사전 구축 `openie_results_gpt-5-mini.json`은 형식이 다르지만 청크 해시(`chunk-` + MD5) 규칙이 같아서 변환이 가능하다.

- `seed_hipporag_cache.py`로 30초 캡션 6,223개분을 HippoRAG 형식(`{"docs": [{idx, passage, extracted_entities, extracted_triples}]}`)으로 바꿔 `.cache/episodic_memory/30sec/`에 심는다. 사전 구축 파일에는 해시만 있고 캡션 원문(`passage`)이 없으므로, `caption.zip`의 `A1_JAKE_30sec.json` 텍스트를 해시로 다시 계산해 합친다. **로컬에서 확인 결과 6,223개 해시가 전부 일치한다.** 이후 첫 실행 시 OpenIE는 `caption.zip`의 3min/10min/1h 캡션 약 1,400개에 대해서만 돈다 (이 세 파일은 사전 구축 OpenIE에 포함되어 있지 않음).
- **코드 구조**: `--retriever-model` 하나가 에피소딕 메모리의 **구축(OpenIE)과 검색(질의 NER, multiscale filter) 양쪽**에 쓰인다. HippoRAG는 캐시 파일명이 retriever와 무관하게 `gpt-5-mini`로 고정되어 있어, 캐시에 있는 캡션은 retriever가 무엇이든 건너뛰고 없는 캡션만 그 retriever로 OpenIE를 돌린다.
- **에피소딕 캐시는 GPT-5-mini로 미리 만든다.** `build_episodic_cache.py`가 4개 스케일을 DAY7 끝까지 한 번에 인덱싱한다. 논문 표의 "Episodic memory: GPT-5-mini"와 일치하고, 30초 스케일(시딩)과 나머지 세 스케일의 생성 모델이 같아진다. 비용 약 $1~2, 소요 30분 안팎.
- **이후 모든 조건은 Qwen으로 돌려도 이 캐시를 그대로 쓴다.** HippoRAG는 캐시 파일명을 자기 기본 설정(`llm_name=gpt-5-mini`)으로 정하므로 `--retriever-model qwen3vl-8b`에서도 캐시가 히트한다. Qwen은 질의 검색, 라운드 결정, multiscale filter, 답변 생성만 담당한다. 저자들도 이 성질로 구축과 inference를 분리한 것으로 보인다.
- **Qwen으로 OpenIE를 돌리는 안은 시도했다가 철회했다.** 아래 "관측된 문제" 두 절이 그 기록이다. 요약하면 1시간 캡션에서 품질이 무너지고(빈 노드 67%) 20시간이 걸린다.
- 한 번 만들어진 `.cache/episodic_memory/`는 모든 조건이 공유한다. 조건 간 에피소딕 인덱스가 같아야 하므로 **중간에 캐시를 지우지 않는다.** 지우면 OpenIE가 다시 돌아 (LLM 출력이 매번 조금씩 달라) 조건 간 인덱스가 어긋난다.

---

## 3. 데이터와 서브셋

### 필요한 데이터

| 항목 | 상태 | 비고 |
|---|---|---|
| A1_JAKE 비디오 DAY1~7 | **전체 필요** (약 102 GB, 6,223 클립) | E 조건의 retrieval이 전체 클립을 대상으로 하므로 전부 필요. 저장소 밖에 둬도 된다 (아래 "비디오 위치") |
| `EgoLifeQA/EgoLifeQA_A1_JAKE.json` | HF에서 수신 필요 | 500문항 |
| `EgoLifeCap/A1_JAKE/*_{30sec,3min,10min,1h}.json` | 있음 (저장소의 `data/EgoLife/caption.zip`에서 풀림) | 에피소딕 캡션, 비주얼 클립 목록. 아래 설명 참조 |
| `output/metadata/` | 있음 | 2장 참조 |

**캡션 4개 파일은 원본 EgoLife에 없는 WorldMM 저자 생성물이다.** 원본 `EgoLifeCap`에는 중국어 DenseCaption과 Transcript(srt)만 있고, 저자들이 번역 → 자막 동기화 → 1인칭 재작성 → 30초/3분/10분/1시간 요약(모두 gpt-5-mini)을 거쳐 만든 결과를 `caption.zip`으로 저장소에 넣어둔 것이다. 이 캡션과 `output/metadata`는 **한 세트**다.

- 에피소딕 OpenIE 결과는 캡션 텍스트의 MD5 해시를 키로 쓴다.
- 비주얼 임베딩은 캡션의 `video_path`를 키로 쓴다.
- 시맨틱 트리플은 캡션의 타임스탬프를 키로 쓴다.

따라서 `2_preprocess.sh`와 `3_build_memory.sh`의 캡션 생성 단계는 **절대 다시 돌리지 않는다.** 다시 생성하면 LLM 출력이 달라져 해시가 어긋나고 사전 구축 메모리를 쓸 수 없게 된다. 원본 EgoLife의 DenseCaption, Transcript, Sync 폴더도 이번 실험에는 불필요하다.

### 비디오 위치

캡션 JSON의 `video_path`는 `data/EgoLife/A1_JAKE/DAY1/....mp4` 처럼 저장소 기준 상대 경로로 적혀 있지만, 실제 비디오는 보통 다른 디스크에 있다. `common.py`의 `VIDEO_ROOT`가 그 `data/EgoLife` 부분을 대체하며 기본값은 `/datasets/EgoLife`다. 우선순위는 이렇다.

1. 스크립트의 `--video-root` 인자
2. 환경변수 `WORLDMM_VIDEO_ROOT`
3. `common.py`의 `VIDEO_ROOT` 기본값

`eval_egolife.py`는 캡션을 `VisualMemory`에 넘기기 전에 경로를 교정하므로 `src` 수정이 필요 없다. 비디오를 읽는 것은 `extract_oracle_frames.py`(oracle 프레임 캐시)와 조건 E / E′(검색된 클립의 프레임)뿐이다. **조건 A, B, C, D는 비디오 없이 돌아간다** (C, D는 미리 캐시한 프레임을 읽는다).

### 전체 QA 분포 (A1_JAKE 500문항 분석 결과)

| 항목 | 값 |
|---|---|
| 유형 | EntityLog 125, EventRecall 126, RelationMap 125, TaskMaster 63, HabitInsight 61 |
| `need_audio=False` | 298 (EntityLog 106, RelationMap 94, EventRecall 66, HabitInsight 28, TaskMaster 4) |
| target_time 형식 | single 451, time_list 36, range 13 |
| target이 30초 캡션에 매핑 안 됨 | 13문항 (캡션 공백 구간) → 제외 |
| range가 수십~수천 클립에 걸침 | 9문항 → oracle 정의 불가, 제외 |
| **Oracle 정의 가능 (target 클립 1~10개)** | **478문항** |
| 그중 `need_audio=False` | 279문항 (EntityLog 103, RelationMap 90, EventRecall 61, HabitInsight 21, TaskMaster 4) |

Oracle 정의 가능한 478문항의 유형 × `need_audio` × query−target 일수 간격(gap 0 / 1 / ≥2) 분포:

| 유형 | need_audio=False | need_audio=True | 합 |
|---|---|---|---|
| EntityLog | 103 (55 / 33 / 15) | 19 (11 / 4 / 4) | 122 |
| RelationMap | 90 (55 / 13 / 22) | 31 (25 / 1 / 5) | 121 |
| EventRecall | 61 (33 / 14 / 14) | 60 (44 / 8 / 8) | 121 |
| TaskMaster | 4 (2 / 1 / 1) | 57 (42 / 8 / 7) | 61 |
| HabitInsight | 21 (10 / 7 / 4) | 32 (24 / 4 / 4) | 53 |

`need_audio`는 정답 근거가 대화에 있는지를 나타내는 원본 주석이다. TaskMaster는 94%, HabitInsight는 60%가 audio 기반이라, 이 축은 **유형과 강하게 얽혀 있다.** 그래서 `need_audio=False`로 미리 자르면 특정 유형이 통째로 빠지고, "어디서 성능이 오르는가"를 실험 전에 가정하는 셈이 된다. 따라서 필터로 쓰지 않고 **층화 축이자 사후 분석 축**으로만 쓴다.

### Main set — 120문항

478문항에서 **유형당 24개**를 뽑는다. 유형 안에서 `need_audio`와 gap으로 층화한다. visual이 도움될 수 있는 `need_audio=False` 문항을 70개 확보해 핵심 비교(D−B, D−E)의 검정력을 지키고, `need_audio=True` 50개로 audio 기반 문항에서의 visual 효과와 text dominance를 같이 본다. long-video 특성을 위해 gap ≥1을 pool 비율(약 45%)보다 높게(약 53%) 잡는다. 시드 고정.

| 유형 | need_audio=False (gap 0 / 1 / ≥2) | need_audio=True (gap 0 / 1 / ≥2) | 합 |
|---|---|---|---|
| EntityLog | 19 (7 / 6 / 6) | 5 (3 / 1 / 1) | 24 |
| RelationMap | 17 (7 / 4 / 6) | 7 (5 / 1 / 1) | 24 |
| EventRecall | 13 (5 / 4 / 4) | 11 (6 / 3 / 2) | 24 |
| TaskMaster | 4 (2 / 1 / 1) | 20 (10 / 5 / 5) | 24 |
| HabitInsight | 17 (8 / 6 / 3) | 7 (4 / 2 / 1) | 24 |
| **합** | **70** | **50** | **120** |

gap 분포는 0 / 1 / ≥2 = 57 / 33 / 30. 모든 칸이 pool 안에 들어간다 (가장 빠듯한 칸: TaskMaster audio=False 4/4, HabitInsight audio=False gap 0 8/10, gap 1 6/7, gap ≥2 3/4).

**분석 구조** (사전 등록):

1. **전체 120문항** — 주 결과. A~E, E′ 정확도와 문항 단위 paired 비교(McNemar, flip 수). 120문항이면 5%p 차이가 문항 6개다.
2. **`need_audio` 분할 (70 vs 50)** — visual gain이 70쪽에서만 나오는지, 50쪽에서도 나오는지. "어디서 오르는가"에 대한 직접 답.
3. **유형별 (각 24)** — 방향성 참고. 유형별 결론은 약하게 표현한다.
4. **gap별 (57 / 33 / 30)** — long-video 효과.

### Masking set — 40문항 ⊂ Main set

마스킹은 정의상 프레임 안에 evidence가 있어야 성립하므로, 여기만은 **`need_audio=False`이고 target 클립이 정확히 1개**인 문항으로 제한한다. 유형별 EntityLog 14, RelationMap 12, EventRecall 9, HabitInsight 5. HabitInsight는 단일 클립 문항이 적으므로(pool 9개) `build_subset.py`에서 HabitInsight `need_audio=False`를 뽑을 때 단일 클립을 우선한다. 사람이 마스크를 검수한다.

---

## 4. 실험별 세부

### Exp.1 — Oracle Visual (utilization 분리)

목적: **retrieval 실패를 제거한 상태에서** MLLM이 visual evidence 자체를 얼마나 활용하는지 본다.

조건 A, B, C, D. 핵심 비교:

- `C − A`: visual evidence만으로 정답을 찾을 수 있는가
- `D − B`: text memory가 이미 있을 때 visual이 추가 정보를 제공하는가
- `C vs B`: 현재 모델에게 어느 modality가 더 usable한가 (진단용, 우열 판정 아님)

### Exp.2 — Retrieved Visual (retrieval 분리)

목적: WorldMM의 visual retrieval이 oracle 수준의 usable evidence를 가져오는가.

조건 B, D, E, E′. 핵심 비교:

- `D − E`: oracle 대비 실제 retrieval의 손실 = **retrieval bottleneck 크기**. 가장 중요한 비교
- `E − B`: 논문의 E+S+V − E+S 재현
- `D − E′`: 텍스트 고정 상태에서 oracle vs retrieved

부가 분석: E의 `round_history`에서 (1) visual 검색이 실제로 선택된 문항 비율, (2) 검색된 클립이 target 클립과 겹친 비율(**retrieval recall@3**), (3) 검색 질의가 텍스트였는지 시간 범위였는지를 기록한다. recall이 낮으면 retrieval bottleneck의 직접 증거다.

### Exp.3 — Visual Masking (reliance 확인)

main performance 실험이 아니라 **post-hoc diagnostic**이다. 목적: 성능 향상이 실제로 task-relevant visual evidence를 이용해서 발생했는가.

Masking set 40문항, 클립당 16프레임 고정. Visual only와 Text + Visual 두 환경에서 모두 평가:

```
C1. Original oracle visual          (frames16, 클립당 16장)
C2. Relevant-region masked oracle visual
C3. Irrelevant-region masked oracle visual

D1. Text + Original oracle visual   (frames16)
D2. Text + Relevant-mask oracle visual
D3. Text + Irrelevant-mask oracle visual
```

메트릭:

```
Relevant Mask Drop   = Acc(original) − Acc(relevant-mask)
Irrelevant Mask Drop = Acc(original) − Acc(irrelevant-mask)
```

이상적인 결과:

```
Original                72
Irrelevant-region mask  71
Relevant-region mask    61
```
→ 단순히 visual 일부가 사라져서가 아니라, **task-relevant evidence를 제거했을 때 선택적으로** 성능이 떨어졌다.

반대로:

```
Original                72
Irrelevant mask         71
Relevant mask           71
```
→ relevant region을 가려도 영향이 없다. 모델이 text memory, 다른 프레임의 visual cue, question prior, 주변 context에 의존했을 가능성. 이때 "visual을 전혀 쓰지 않는다"로 결론내리지 말고 **"해당 evidence region에 대한 reliance가 낮다"** 로 표현한다.

**마스킹 구현**

- 프레임 수는 원본과 동일하게 유지한다 (Original 16장 ↔ Masked 동일한 16장).
- relevant object/region이 등장하는 sampled frame마다 동일하게 마스크를 적용한다.
- irrelevant-region mask는 **relevant mask와 비슷한 면적**으로 맞춘다. 가린 픽셀 양 차이 때문에 성능이 달라지는 것을 막기 위함. relevant와 겹치지 않는 영역에서 같은 면적으로 무작위 생성하고, 프레임 간 위치를 고정한다.
- 마스크는 회색 채움.

파이프라인:

```
Target evidence 정의 (질문 유형별 규칙, keywords/reason 필드 활용)
        ↓
GroundingDINO로 텍스트 프롬프트 → bbox 후보 검출
        ↓
SAM2로 bbox 보정 / segmentation / 프레임 간 tracking
        ↓
사람 검수 (40문항 × 16프레임, 검수 페이지)
        ↓
relevant / irrelevant region masking
        ↓
동일 프레임 수로 Qwen3-VL 입력
```

질문 유형별 relevant region 규칙:

| 질문 성격 | relevant region |
|---|---|
| Object identity / attribute | target object |
| Object location | target object + 주변 지지면 |
| Person-object interaction | object + 손/사람 상호작용 영역 |
| Spatial relation | 관련 object 두 개 |
| Fine-grained action | 손 + 조작 중인 object |

---

## 5. 실행 프로토콜

### 서버 실행 명령 (요약)

```sh
cd ~/WorldMM && source .venv/bin/activate

# src 수정 3건이 적용되어 있는지 확인 (6장 참조). 셋 다 숫자가 나와야 한다
grep -c "start_sec=None"        src/worldmm/memory/visual/memory.py       # 1
grep -c "openie_info_for_index" src/HippoRAG/src/hipporag/HippoRAG.py     # 2
grep -c "model_fields"          src/worldmm/llm/qwen3vl.py                # 1
git diff --stat src/worldmm/llm/templates/   # 비어 있어야 한다 (프롬프트 무수정)

# 데이터 (1회)
hf download lmms-lab/EgoLife --repo-type=dataset --local-dir data/EgoLife --include "EgoLifeQA/*"
ls -l output/metadata/semantic_memory/A1_JAKE/   # consolidation 파일이 120,569,130 바이트인지 확인

# 비디오 위치. 기본값은 /datasets/EgoLife 이며, 다르면 아래 한 줄로 바꾼다
export WORLDMM_VIDEO_ROOT=/datasets/EgoLife
ls $WORLDMM_VIDEO_ROOT/A1_JAKE/DAY1 | head -3   # mp4가 보여야 한다

# 준비 (1회)
python experiments/visual_bottleneck/build_subset.py
python experiments/visual_bottleneck/seed_hipporag_cache.py      # 30sec 캐시 심기
python experiments/visual_bottleneck/extract_oracle_frames.py

# 에피소딕 캐시 구축 (1회, GPT-5-mini, 약 30분 / $1~2)
export OPENAI_API_KEY="..."
python experiments/visual_bottleneck/build_episodic_cache.py
#   실행 전후로 스케일별 문서 수와 entity 빈 비율을 출력한다.
#   1h의 빈 비율이 30sec(7%) 근처면 정상. 이 단계 이후 OPENAI_API_KEY는 불필요

# 스모크 테스트 → 조건 B (첫 실행: 3min/10min/1h OpenIE + 임베딩 포함, 가장 오래 걸림)
python experiments/visual_bottleneck/eval_egolife.py --condition B --limit 3
python experiments/visual_bottleneck/eval_egolife.py --condition B --resume
ls .cache/episodic_memory/*/openie_results_ner_gpt-5-mini.json   # 4개 있어야 함

# Exp.1, Exp.2
for c in A C D E E_prime; do
  python experiments/visual_bottleneck/eval_egolife.py --condition $c --resume
done
python experiments/visual_bottleneck/analyze.py

# Exp.3 (마스킹)
python experiments/visual_bottleneck/make_masks.py --init-prompts   # mask_prompts.json 편집
python experiments/visual_bottleneck/make_masks.py
python experiments/visual_bottleneck/make_masks.py --review          # review_masks.html 검수 → review_decisions.json 내보내기
python experiments/visual_bottleneck/apply_masks.py
for c in C1 C2 C3 D1 D2 D3; do
  python experiments/visual_bottleneck/eval_egolife.py --condition $c --resume
done
python experiments/visual_bottleneck/analyze.py
```

`--retriever-model`, `--respond-model`은 기본값이 `qwen3vl-8b`라 생략 가능하다. 중단되면 같은 명령을 `--resume`으로 다시 실행하면 이어서 돈다.

### 단계별 설명

1. **환경**: `uv sync` (홈이 NFS면 `UV_CACHE_DIR`을 로컬 디스크로). 모델 가중치 사전 다운로드: `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-Embedding-4B`, `VLM2Vec/VLM2Vec-V2.0`. `OPENAI_API_KEY`는 6단계 캐시 구축에만 필요.
2. **데이터**: A1_JAKE 비디오 전체 + `EgoLifeQA_A1_JAKE.json` 다운로드.
3. `build_subset.py` → `subset.json` (120문항 + masking 40 태그, need_audio, target 클립 목록, gap, 시드).
4. `extract_oracle_frames.py` → `frames/{qid}/` (1fps, 64장 상한), `frames16/{qid}/` (masking set, 클립당 16장).
5. `seed_hipporag_cache.py` → `output/metadata`의 30초 OpenIE 결과를 `.cache/episodic_memory/30sec/`에 HippoRAG 형식으로 심기.
6. `build_episodic_cache.py --model gpt-5-mini` → 3min/10min/1h OpenIE(약 1,478건)와 4개 스케일 임베딩. **1회만 한다.** 완료 후 `.cache/episodic_memory/*/openie_results_ner_gpt-5-mini.json` 4개와 스케일별 빈 개체 비율을 확인.
7. 이후 모든 조건은 `--retriever-model qwen3vl-8b --respond-model qwen3vl-8b`. 먼저 `eval_egolife.py --condition B` → `results/B.json` + `text_context/{qid}.json`. 캐시 히트라 OpenIE는 돌지 않는다.
8. `eval_egolife.py --condition A`, `C`, `D` → `results/{condition}.json`.
9. `eval_egolife.py --condition E`, `E_prime`.
10. Exp.1, Exp.2 중간 판정.
11. Exp.3: `make_masks.py --init-prompts` → `mask_prompts.json` 편집 → `make_masks.py` → `make_masks.py --review` → 브라우저 검수 후 `review_decisions.json` 내보내기 → `apply_masks.py` → `eval_egolife.py --condition C1|C2|C3|D1|D2|D3`.
12. `analyze.py` → 표, 델타, 유형별·gap별 분해, McNemar, retrieval recall, flip 분석.

**고정할 것**: 응답 모델, 생성 파라미터, 프롬프트 템플릿, 프레임 해상도, 선택지 순서, `.cache/episodic_memory/`. 조건 간에 바뀌는 건 컨텍스트 내용뿐이어야 한다.

**생성 파라미터는 원본 그대로 둔다 (아무것도 넘기지 않음).** 한 번 `do_sample=False`로 greedy 디코딩을 강제했다가 되돌렸다. 원본이 파라미터를 넘기지 않아 모델의 `generation_config`를 쓰므로, 그것을 바꾸면 재현이 아니기 때문이다. 대가로 디코딩이 비결정적이 되지만, 조건 간 비교의 핵심인 텍스트 컨텍스트는 B가 만든 `text_context/`를 D와 E′가 재사용하므로 고정된다. 재실행 시 `--resume`으로 기존 결과를 보존해 조건 내 일관성을 지킨다.

### 관측된 문제: Qwen OpenIE의 degenerate repetition

Qwen3-VL-8B로 긴 캡션의 NER을 돌리면 같은 토큰을 수백 번 반복하다 `max_new_tokens=2048`에 잘려 JSON이 깨진다. 재시도가 소진되면 그 캡션은 빈 개체·트리플로 저장되어 그래프에 기여하지 못한다. 캡션이 길수록 심하다.

| 스케일 | 캐시 빈 문서 비율 |
|---|---|
| 30sec (GPT-5-mini 시딩) | 7% |
| 3min | 0.9% |
| 10min | 1% |
| 1h | **67%** |

greedy 디코딩 때문이라고 보고 `do_sample=False`를 제거했으나, sampling에서도 동일하게 재현되어 **가설은 기각되었다.** 모델 고유의 동작이다. `repetition_penalty`를 넣으면 완화되겠지만 원본에 없는 파라미터라 또 다른 편차가 된다.

### 관측된 문제: Qwen OpenIE의 처리 속도

실측 기준 캡션 하나에 3분 스케일 약 40초, 10분 스케일 약 65초가 걸린다 (NER + 트리플 추출). 남은 1,478개를 처리하면 **OpenIE에만 약 20시간**이고 여기에 120문항의 추론 라운드가 더해진다. OpenIE의 `ThreadPoolExecutor`는 API 동시 호출용이라 GPU에서는 요청이 직렬화되어 병렬화 이득이 없다.

**산출물 구조**

```
experiments/visual_bottleneck/
├── PLAN.md
├── subset.json
├── frames/{qid}/{clip_idx}_{frame_idx}.jpg
├── frames16/{qid}/...              # Exp.3용 고정 16장
├── text_context/{qid}.json         # 조건 B의 round_history + retrieved_items
├── masks/{qid}/{frame}.png
├── masked_frames/{relevant,irrelevant}/{qid}/...
├── results/{A,B,C,D,E,E_prime,C2,C3,D2,D3}.json
└── analysis/                       # 표, 그림
```

결과 JSON은 문항별로 `ID, type, need_audio, gap, condition, response, correct, num_frames, num_rounds, visual_used, visual_query_kind, retrieved_clips, target_clips`를 담는다.

---

## 6. 기존 코드 수정 범위

원칙: **재현 실험이므로 방법을 바꾸는 수정은 하지 않는다.** 프롬프트 템플릿(`ner`, `triple_extraction`, `memory_reasoning`, `qa_egolife`), 검색 알고리즘, 하이퍼파라미터는 원본 그대로 쓴다. `eval/eval_egolife.py`도 무수정이고, 실험 로직은 전부 `experiments/visual_bottleneck/eval_egolife.py` 안에서 해결한다.

src 수정은 아래 2건뿐이며, 각각 재현 충실도에 미치는 영향을 함께 기록한다. 결과 보고 시 이 표를 같이 싣는다.

| 수정 | 성격 | 재현 영향 |
|---|---|---|
| `visual/memory.py` 프레임 버그 | 버그 수정 | **있음.** 아래 1번 |
| `HippoRAG.py` assert 완화 | 인프라 | 없음. 만들어지는 그래프 동일 |

**되돌린 것 3건** (모두 한 번 넣었다가 뺐다):

1. `templates/ner.py`, `templates/triple_extraction.py`의 출력 형식 지시 문구. 원본이 "Respond with a JSON list"라고 적어 놓고 파서는 객체를 기대하는 불일치가 있지만, 프롬프트를 바꾸면 에피소딕 메모리 생성 방법 자체가 달라져 재현이 아니다.
2. `qwen3vl.py`의 구조화 출력 파서 관대화. Qwen OpenIE를 포기했으므로 불필요해졌다. 추론 단계(검색, 라운드 결정, multiscale filter, 답변)는 `text_format`을 쓰지 않고 평문 생성 후 정규식으로 JSON을 뽑기 때문에 Qwen에게는 이 경로가 없다.
3. `eval_egolife.py`의 `do_sample=False`. 5장 "생성 파라미터" 참조.

1. **[적용됨, src 수정] Visual similarity 검색 프레임 0장 버그** — `src/worldmm/memory/visual/memory.py`의 `_retrieve_by_similarity`가 하루 기준 초(`clip_start_sec`)를 클립 내부 오프셋으로 넘겨서 텍스트 질의 시 항상 빈 결과가 나왔다. 클립 전체를 뽑도록 `start_sec=None, end_sec=None`으로 수정함. 시간 범위 질의 경로는 원래 정상.
   **재현 영향**: 원본 상태라면 논문의 E+S+V도 "장면 설명" 질의에서는 이미지를 한 장도 보지 못하고, 시간 범위 질의일 때만 봤다는 뜻이 된다. 고치지 않으면 조건 E가 사실상 B와 같아져 Exp.2가 성립하지 않으므로 고치는 쪽을 택한다. 논문 수치와 E가 다르게 나오면 이 수정이 유력한 원인이다. 참고로 조건 E의 `visual_query_kind` 기록으로 질의가 텍스트였는지 시간 범위였는지 사후 분리가 가능하다.
2. **[적용됨, src 수정] HippoRAG 인덱싱 assert** — `src/HippoRAG/src/hipporag/HippoRAG.py`의 `index()`가 OpenIE 캐시 문서 수와 현재 인덱싱 청크 수가 같다고 assert한다. 시딩으로 캐시에 6,223건이 먼저 들어가면 첫 질문(청크 96개)에서 깨진다. 캐시에서 현재 청크에 해당하는 항목만 골라 그래프를 만들도록 두 줄 수정. 캐시 파일 저장은 전체를 유지하므로 시딩 결과는 보존된다.
3. **[스크립트 안에서 해결] Visual 비활성화(B)** — reasoning 템플릿 덮어쓰기 + `retrieve_from_visual` 교체. 2장 "구현 방식" 3번.
4. **[스크립트 안에서 해결] QA 직접 호출(A/C/D/E′)** — `answer_with_context` 헬퍼. 2장 "구현 방식" 4번.
5. **[스크립트 안에서 해결] 시맨틱 경로 하드코딩** — `--semantic-path` 인자로 대체.
6. **[스크립트 안에서 해결] Qwen 인스턴스 공유** — retriever와 respond가 둘 다 `qwen3vl-8b`이므로 원본대로면 8B가 두 벌 올라간다. 복사본 `eval_egolife.py`에서 두 이름이 같으면 `LLMModel` 하나를 둘 다에 넘긴다 (`fps=1`은 retriever 쪽에 붙여도 무해). GPU 메모리 약 16 GB 절약.
7. **[불필요] `extract_visual_features.py`의 `--num_frames` 무시** — 사전 구축 임베딩을 쓰므로 영향 없음.

---

## 7. 이 폴더에 작성할 스크립트

| 파일 | 역할 |
|---|---|
| `common.py` | 경로 상수, 타임스탬프 변환, `target_time` → 30초 클립 매핑 등 공용 헬퍼. torch/worldmm 미의존 |
| `build_subset.py` | QA 로드, target_time → 클립 매핑, 유형 × need_audio × gap 층화 추출(120), masking 40 태그, `subset.json` 출력 |
| `extract_oracle_frames.py` | 1fps/64장 및 고정 16장 프레임 캐시 |
| `seed_hipporag_cache.py` | `output/metadata/episodic_memory/A1_JAKE/openie_results_gpt-5-mini.json` → `.cache/episodic_memory/30sec/openie_results_ner_gpt-5-mini.json` 변환. 캡션 텍스트 해시 6,223개가 모두 매칭되는지 검증 |
| `build_episodic_cache.py` | 4개 스케일을 DAY7 끝까지 1회 인덱싱해 `.cache/episodic_memory/`를 완성. 기본 `--model gpt-5-mini`. 실행 전후 스케일별 문서 수와 entity 빈 비율 출력 |
| `eval_egolife.py` | **모든 조건의 러너.** `eval/eval_egolife.py` 복사본 + `--condition` 분기 (2장 "구현 방식"). B는 `text_context/` 캐시도 생성. E는 원본 루프 그대로에 retrieved 클립 키·visual 사용 여부·질의 종류만 기록. `--resume`(중단 지점부터), `--limit N`(스모크 테스트) 지원. 결과는 문항마다 즉시 저장 |
| `make_masks.py` | Exp.3 마스크 생성. `--init-prompts`로 `mask_prompts.json` 템플릿 생성 → 프롬프트 편집 → 실행하면 GroundingDINO(transformers) + SAM2/SAM(transformers)으로 `masks/{qid}/*.png`와 overlay 생성 → `--review`로 `review_masks.html` 생성 |
| `review_masks.html` | 브라우저에서 문항별 approve / reject / fix 선택과 메모. "export decisions JSON"으로 `review_decisions.json` 저장 |
| `apply_masks.py` | `review_decisions.json` 반영(reject 제외, fix는 메모의 박스 JSON 사용) 후 `masked_frames/{relevant,irrelevant}/{qid}/` 생성. irrelevant는 relevant와 같은 면적, 겹치지 않는 위치, 문항 내 프레임 간 고정 |
| `analyze.py` | 정확도 표, 델타, 유형·gap별 분해, McNemar, retrieval recall, flip 분석 |

---

## 8. 리소스 추정

| 항목 | 추정 |
|---|---|
| GPU 메모리 | Qwen3-VL-8B bf16 약 16 GB (인스턴스 공유 시 한 벌) + Qwen3-Embedding-4B 약 8 GB + VLM2Vec 약 5 GB. A100 40 GB 1장으로 가능. 공유 안 하면 약 45 GB |
| 디스크 | 비디오 102 GB, 프레임 캐시 약 0.5 GB, HippoRAG 캐시 수 GB |
| 캐시 구축 (`build_episodic_cache.py`) | 3min/10min/1h OpenIE 약 1,478건 × 2회 호출(gpt-5-mini API, 입력 2.6M·출력 0.3M 토큰, **약 $1~2**, 30분 안팎) + 캡션 약 7,700개 임베딩(로컬 GPU). 1회성 |
| 조건 B | 문항당 최대 5라운드, 라운드마다 reasoning 1회 + 질의 NER 1회 + multiscale filter 1회, 전부 로컬 Qwen. 120문항 기준 1~2시간 |
| 조건 A/C/D/E′ | 문항당 forward 1회. 64장 이미지 입력이라 C/D가 가장 무거움. 120문항 기준 각 30~40분 |
| 조건 E | B와 유사 + visual 검색 시 프레임 추출 |
| Exp.3 | 40문항 × 6조건, 16장 입력. 1시간 내외. 사람 검수 640장, 2~3시간 |

API 비용은 캐시 구축 1회($1~2)뿐이고, 이후 inference는 전부 로컬 Qwen이라 비용이 없다. **모든 조건은 같은 `.cache/episodic_memory/`를 공유해야 한다.**

---

## 9. 전체 흐름

```
          EgoLifeQA A1_JAKE (500)  →  oracle 정의 가능 (478)  →  Main set (120, 유형×need_audio×gap 층화)
                                                                     │
                                                                     ▼
                                                           Experiment 1
                                                           Oracle Visual
                                                                     │
                              ┌──────────────────┬───────────────────┼──────────────────┐
                              ▼                  ▼                   ▼                  ▼
                       A. Question only   B. Text memory     C. Oracle visual   D. Text + Oracle
                                                                     │
                                                                     ▼
                                                     Visual utilization 진단 (C−A, D−B, C vs B)
                                                                     │
                                                                     ▼
                                                           Experiment 2
                                                           Retrieved Visual
                                                                     │
                                                   E. Text + Retrieved visual (원본 WorldMM)
                                                   E′. B 컨텍스트 + Retrieved visual (통제)
                                                                     │
                                                                     ▼
                                                     Oracle과 직접 비교 (D−E, E−B, recall@3)
                                                                     │
                                       ┌─────────────────────────────┴─────────────────────────┐
                                       ▼                                                       ▼
                              Retrieval bottleneck                                   Utilization bottleneck
                                                                     │
                                                                     ▼
                                                           Experiment 3
                                                           Visual Masking (40)
                                                                     │
                                   ┌─────────────────────┬───────────┴───────────┐
                                   ▼                     ▼                       ▼
                               Original          Relevant mask           Irrelevant mask
                                                                     │
                                                                     ▼
                                                     실제 visual reliance 분석 (mask drop)
                                                                     │
                                                                     ▼
                                                  후속 방향 결정 (gaze memory / gaze input / 둘 다)
```

---

## 10. 체크리스트

- [ ] 환경 구축, 모델 가중치 다운로드
- [ ] A1_JAKE 비디오 전체 + `EgoLifeQA_A1_JAKE.json` 다운로드
- [ ] 6장 코드 수정 1~3 적용, visual similarity가 프레임을 반환하는지 확인
- [ ] `build_subset.py` → Main 120 / Masking 40 확정
- [ ] 프레임 캐시 (`frames/`, `frames16/`)
- [ ] `seed_hipporag_cache.py`로 30초 OpenIE 캐시 시딩, 해시 6,223개 전부 매칭 확인
- [ ] `build_episodic_cache.py`로 4개 스케일 캐시 완성. 파일 4개와 1h 빈 개체 비율 확인. 이후 `OPENAI_API_KEY` 불필요
- [ ] 조건 B 실행 (OpenIE 없이 추론만), 컨텍스트 캐시 검증 (round_history에 visual이 없는지)
- [ ] 조건 A, C, D 실행 → Exp.1 판정 (C−A, D−B, C vs B)
- [ ] 조건 E, E′ 실행 → Exp.2 판정 (D−E, E−B, recall@3)
- [ ] 마스크 생성·검수·적용
- [ ] C1~3, D1~3 실행 → mask drop 계산
- [ ] 최종 표와 해석, 후속 방향(gaze) 결정
