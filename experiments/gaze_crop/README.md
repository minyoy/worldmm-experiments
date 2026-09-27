# Gaze-crop 검색 실험 (WorldMM, EgoLifeQA A1_JAKE)

> **한 줄 요약.** 착용자의 시선 지점을 프레임마다 따라가며 자른 뒤 임베딩하면(`gazef@0.5`)
> visual retrieval이 **전체 화면(`full`)보다 확실히 좋아진다**: A1_JAKE 500문항 · 실제 인덱스
> 6,223클립에서 **recall@3 9.2% → 13.4%** (+4.2 %p, McNemar p=0.006), 정답 순위 중앙값 99위 → 76위,
> 문항 단위로는 **290문항 개선 / 194문항 악화** (sign test p=1.5e-5). 엄격 채점(tol 0s, 487문항)에서도
> **3.1% → 5.1%** (+2.1 %p, p=0.087)로 방향이 같고 유의성은 경계선이다.
> 그런데 **이 이득은 아직 답 정확도로 바뀌지 않았고**(E′ 45.0% vs `full` 45.0%), 두 가지가 미해결이다:
> **(1) crop 대조군(`center@0.5`)을 이 규모에서 측정하지 않아 "gaze 덕"인지 "자르기 덕"인지 아직 못 가른다,**
> **(2) 정확도를 잰 120문항 subset은 두 arm 모두 recall@3 ≈ 0.8%인 극단적으로 어려운 표본이었다.**

설계 전문은 [`PLAN.md`](PLAN.md), 이전 실험은 [`../visual_bottleneck/README.md`](../visual_bottleneck/README.md).
`analysis/DIGEST.md`는 500문항 결과까지 읽도록 `digest.py`를 고쳐 다시 생성했습니다 — 맨 위 헤드라인이
가장 검정력 높은 측정(500문항)을 가리키고, tolerance와 chance가 표마다 붙습니다. `gaze_points.json`과
`results_qa/`는 git에 없어서(재생성 가능 산출물) 실행 머신이 아닌 곳에서는 transform·정확도 절이
`analysis/stage2_recorded.json`의 기록본으로 채워지고, 출력에 **[기록본]**으로 표시됩니다. 원본이 있는
머신에서 다시 돌리면 원본이 항상 우선합니다.

---

## 1. 왜 이 실험을 했나

[visual_bottleneck](../visual_bottleneck/README.md) 실험의 결론은 **병목이 utilization이 아니라 retrieval**
이라는 것이었습니다.

| 관찰 | 값 |
|---|---|
| 텍스트 검색이 놓친 82/120 문항에서 oracle visual 프레임의 기여 (D−B) | **+12.2 %p** |
| 실제 visual 검색이 target 30초 클립을 top-3에 넣는 비율 (조건 E′) | **5.0%** (6/120) |
| 에이전트가 스스로 visual을 부른 문항 (조건 E) | 2/120 |

즉 **쓸 만한 시각 정보는 있는데(oracle +12.2 %p) 검색이 못 찾아옵니다(recall@3 5%).**
이 실험은 그 검색 실패의 일부가 **프레이밍 실패**인지 묻습니다. 지금 파이프라인은 30초 클립 하나를
전체 화면 16프레임으로 보고 임베딩 하나로 압축하므로, 질문과 무관한 배경(천장·벽·빈 책상)이 임베딩을
지배할 수 있습니다. 1인칭 영상에는 "그 순간 사람이 실제로 보고 있던 곳"이라는 공짜 신호(eye gaze)가
있으니, 그 점 주위만 잘라서 임베딩하면 클립 벡터가 행동/객체 쪽으로 옮겨갈 것이라는 가설입니다.

측정 순서는 **recall 먼저, 정확도 나중**입니다. crop이 직접 움직이는 양은 recall이고, 정확도는 그 recall을
LLM 비용을 들여 나중에 사 오는 것이기 때문입니다.

## 2. 어떻게 돌렸나

### 2.1 Arm (픽셀만 다르고 나머지는 전부 동일)

arm은 **crop 박스를 어디에 두는가**로만 갈립니다. `gazef`는 "박스를 좌표 파일이 말하는 위치로 프레임마다
옮긴다"는 코드 경로일 뿐이고, **거기에 어떤 좌표 파일을 주느냐가 arm을 만듭니다** — 실제 시선이면 실험군,
무작위 좌표면 대조군입니다. 구분되는 축은 둘입니다: **박스가 프레임마다 움직이나**, **그 위치가 시선인가**.

박스를 **고정**하는 gaze 계열 arm(클립 median 시선, 고정 무작위 위치)은 제거했습니다 — 클립별 median
시선이 (0.515, 0.561)이라 `center@0.5`와 거의 같은 픽셀을 자르고, `center@0.5`가 이미 답하는 것을 되물을
뿐이었습니다. 코드에서도 `gaze@R` arm과 `--pseudo random`을 지웠습니다.

| 임베딩 파일 (`emb/`) | 박스 위치 | 프레임마다 이동 | 무엇을 말해 주나 |
|---|---|---|---|
| `full` | crop 없음 (전체 화면) | — | 베이스라인 (저자 임베딩과 같은 픽셀) |
| `center_05` | 화면 중앙 고정 | ✗ | **자르기 자체의 효과** |
| **`gazef_05`** | **그 순간의 시선** | **✓** | **실험군(본진)** |
| `gazef_05_randf` | 무작위 (매 샘플 새로 뽑음) | ✓ | **움직임 자체의 효과** (움직이지만 시선은 아님) |
| `pkl` | crop 없음, 저자 임베딩 행 | — | 참조용 (프레임 샘플링 경로가 달라 arm 비교 대상 아님) |

만드는 명령은 이렇게 대응합니다 — `gazef_05`와 `gazef_05_randf`는 **같은 `--arms gazef@0.5`**이고
`--gaze`(좌표 파일)와 `--arm-suffix`(파일 이름)만 다릅니다:

```bash
embed_arms.py --arms gazef@0.5                                              # -> emb/gazef_05.npz
embed_arms.py --arms gazef@0.5 --gaze gaze_random_frames.json --arm-suffix _randf   # -> gazef_05_randf.npz
```

`gazef_05`가 `full`을 이겼을 때 "왜 이겼나"의 후보가 셋이고, 위 표의 각 대조군이 하나씩 끊어 냅니다:

| 후보 설명 | 끊어 내는 비교 |
|---|---|
| ① 배경이 빠지고 대상이 커져서 (자르기) | `gazef_05` vs `center_05` |
| ② 박스가 프레임마다 옮겨다녀 16프레임이 서로 다른 영역을 덮어서 (움직임) | `gazef_05` vs `gazef_05_randf` |
| ③ 그 움직임이 하필 **시선** 위치라서 | ①②가 다 설명 못 하고 남는 부분 |

모델(VLM2Vec-V2.0), 16프레임 샘플링 규칙, `max_pixels = 360*420`, `encode_video` 호출, 쿼리, candidate set은
전부 고정입니다. 즉 **바뀌는 것은 crop 박스 하나**입니다.

### 2.2 gaze를 픽셀 좌표로 옮기기 ([`prepare_gaze.py`](prepare_gaze.py))

gaze는 별도 HF 데이터셋(`Wangtwohappy/EgoLife_EyeTracking_EyeGaze`)이고, 클립 mp4와 같은 이름의 CSV
(30초당 300행, 10 Hz)에 **픽셀이 아니라 CPF 기준 yaw/pitch 라디안**이 들어 있습니다. 처리 과정:

- **좌/우 눈 결합**: Aria 공식 경로(`compute_depth_and_combined_gaze_direction`)와 같은 기하 —
  결합 yaw = `atan((tan yaw_L + tan yaw_R)/2)`, vergence depth = `0.063/(tan yaw_R − tan yaw_L)`.
  검증: 역산한 depth와 CSV `depth_m`의 중앙 오차 **0.022 m** → 컬럼과 부호를 문서대로 읽고 있음.
- **투영**: equidistant(`r = f·θ`), focal 611 px @1408을 프레임 폭에 스케일(≈HFOV 98°).
  VRS/calibration 파일이 두 릴리스 어디에도 없어서(`.vrs` 0개) 공식 fisheye624 재투영은 **불가능**하고,
  단일 focal 근사가 유일한 선택입니다. 프레임 밖 시선은 버리지 않고 가장자리로 clamp —
  실측 clamp 비율 **0/185,424 = 0.0%**.
- **좌표계 transform**: mp4가 센서 기준으로 어떻게 돌아갔는지 문서로 알 수 없으므로 후보 6개
  (`none, flipx, flipy, flipx+flipy, rot90cw, rot90ccw`)를 놓고 **사람이 그림을 보고 골랐습니다**
  ([`compare_transforms.py`](compare_transforms.py) → `variants/*.jpg`). 결정: **`flipx+flipy`**
  (= 180° 회전). 캡션 코사인 점수는 끝까지 무승부였고(1·2위 격차 95% CI −0.009…+0.012), 판정은 그림
  12장(교과서적 시선 목표 — 얼굴·투사 화면·물건 집는 손)으로 했습니다. **실험 지표(recall)로 transform을
  고르는 것은 금지**했습니다. 증거는 `overlay/`, `variants/`에 남아 있습니다.
- 결과 확인: 클립별 median gaze는 x 0.515, y 0.561 (p10–p90 x 0.48–0.55 / y 0.51–0.62) — 30초 median은
  거의 중앙으로 수렴합니다. 그래서 시선의 위치 정보는 **프레임별로 따라갈 때만** 살아남습니다 —
  클립당 한 점으로 박스를 고정하면 `center@0.5`와 거의 같은 픽셀이 되고, 그것이 고정 gaze arm을
  아예 제거한 이유입니다(§2.1).

### 2.3 풀과 채점 정의

| | 1단계 (kill switch) | 2단계 (본 측정) |
|---|---|---|
| 풀 | 623클립 (target 123 + 같은 날 distractor 500) | **6,223클립 전부 = 실제 인덱스 크기** |
| 문항 | visual_bottleneck의 120문항 subset | 처음 120, 이후 **A1_JAKE 500문항 전부** |
| 문항당 경쟁 클립(중앙값) | 323 | 2,585 |
| 비용 | arm당 ~10분 | arm당 ~2시간 (+QA 시 arm당 ~34분) |

- **미래 차단**: 문항별 후보는 `ts_end <= query_time`인 클립뿐이라 문항마다 후보 수가 다릅니다(96~5,966).
- **정답 판정은 `target_time` 주석에서 직접** 계산합니다. `common.py`의 파생 `target_clips`는 연쇄
  타임스탬프를 구간으로 잘못 읽어 한 문항의 정답 칸을 2,430개까지 부풀리는 버그가 있어 쓰지 않습니다.
- **tolerance**: `--target-tolerance-sec`. `0`은 EgoLife가 태그한 30초 칸 하나만 정답으로 보는 엄격한 정의,
  `60`은 "근처에 갔나"를 묻는 다른 질문입니다(정답 칸 중앙값 1 → 5개). tolerance를 올리면 recall이 저절로
  오르므로 **항상 같은 tolerance끼리만** 비교하고, 아래 표에는 chance 수준을 같이 적었습니다.
  tol=0에서 정답 칸이 0개가 되는 문항 13개(주석 시각이 클립 사이 틈에 떨어짐)는 채점 불가로 빠집니다.
- **문항을 500개로 늘린 이유**: recall 채점에는 LLM도 캐시도 필요 없어서 문항 추가 비용이 거의 0인데,
  검정력은 문항 수가 전부입니다. 120문항에서는 짝지은 불일치가 **4건**뿐이라 어떤 통계로도 결론이 안 났고,
  500문항에서는 **55건**이 되어 결론이 납니다([`build_questions.py`](build_questions.py)).

### 2.4 정확도 측정 (2단계)

평가기를 새로 쓰지 않고 `visual_bottleneck/eval_egolife.py`의 **조건 E′**(B가 캐시한 텍스트 컨텍스트 +
질문으로 강제 검색한 visual top-k)에서 **visual 인덱스만** arm 임베딩으로 바꿔 끼웠습니다
([`export_pkl.py`](export_pkl.py) → `--visual-path`). `--robust-reasoning`은 기존 E′와 플래그를 맞추려고
켜지 않았고, LLM에 가는 프레임은 잘리지 않은 전체 프레임입니다(E′ 원래 동작). E가 아니라 E′를 쓰는 이유는
E에서 8B가 visual을 120문항 중 2문항에서만 불러서 검색 품질이 아무리 올라도 end-to-end 숫자가 움직이지
않기 때문입니다(라우팅 문제는 이 실험 범위 밖).

## 3. 결과

### 3.1 1단계 — 623클립 풀, 120문항 (kill switch)

`--query-source question`(E′가 실제로 쓰는 질의):

| arm | R@1 | R@3 | R@5 | R@10 | R@50 | med. rank |
|---|---|---|---|---|---|---|
| `center_05` | 0.8% | **3.3%** | 5.0% | 7.5% | 22.5% | 118 |
| `full` | 0.8% | 2.5% | 2.5% | 6.7% | 27.5% | 114 |
| `gazef_05` | 1.7% | **3.3%** | 4.2% | 7.5% | 28.3% | 109 |
| `pkl` | 1.7% | 1.7% | 4.2% | 8.3% | 30.8% | 111 |

`gazef@0.5 − full = +0.8 %p`(불일치 1:0)로 게이트를 통과했지만, 이 규모에서는 **사실상 아무 정보가 없는
숫자**입니다. 같은 실행의 `--query-source keywords`(QA 자체 키워드, 더 친절한 질의)에서는 신호가 훨씬
크게 나왔고 **순위가 뒤집힙니다**:

| arm | R@3 | R@10 | med. rank |
|---|---|---|---|
| `center_05` | **19.2%** | 30.8% | 35 |
| `gazef_05` | 15.0% | 30.0% | 41 |
| `pkl` | 16.7% | 28.3% | 47 |
| `full` | 12.5% | 26.7% | 48 |

- `center_05 − full = +6.7 %p` (95% CI +1.7…+12.5) — **자르기만으로도 오릅니다.**
- `center_05 − gazef_05 = +4.2 %p` (CI −0.8…+9.2) — gaze 추종이 중앙 crop을 못 이깁니다.
- 이 표에서 `center_05`가 1위입니다. **1단계가 남긴 유일한 실질적 경고**이고, 그래서 500문항 규모의
  `center_05` 측정이 다음 단계 1순위입니다.

### 3.2 2단계 — 6,223클립(실제 인덱스), 120문항: 무결과

| tolerance | arm | R@3 | R@10 | med. rank | chance@3 |
|---|---|---|---|---|---|
| 0s (엄격) | `full` | 0.8% | 1.7% | 878 | 0.29% |
| 0s | `gazef_05` | 0.8% | 2.5% | 909 | 0.29% |
| 60s | `full` | 0.8% | 3.3% | 415 | 1.38% |
| 60s | `gazef_05` | **4.2%** | 5.0% | 368 | 1.38% |

tol=0에서는 **두 arm 모두 정답을 120문항 중 1개**밖에 못 올렸습니다(불일치 2건, p=1.000). tol=60에서
방향은 gaze 쪽이지만 불일치 4건(p=0.125)으로 역시 결론 불가입니다. 여기서 멈추면 "gaze는 효과 없음"으로
기록될 상황이었는데, **표본이 너무 작고 이 120문항 자체가 이상하게 어려웠습니다**(3.5).

### 3.3 500문항으로 확장 — gaze가 이긴다

같은 임베딩, 같은 풀(6,223), 문항만 A1_JAKE 전체 500개로. tolerance 60s:

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | med. rank |
|---|---|---|---|---|---|---|---|
| `full` | 4.8% | 9.2% | 11.8% | 20.4% | 26.8% | 39.6% | 99 |
| **`gazef_05`** | **7.2%** | **13.4%** | **17.4%** | **25.4%** | **33.2%** | **44.4%** | **76** |
| _chance_ | _0.55%_ | _1.60%_ | _2.58%_ | _4.84%_ | _8.75%_ | _17.92%_ | _428_ |

- **`gazef_05` − `full` = +4.2 %p at k=3**, 불일치 55건(38 : 17), **McNemar p = 0.006**.
- 정답 순위가 개선된 문항 **290 / 악화 194 / 동일 16**, sign test **p = 1.5e-5**. k=1부터 k=50까지
  모든 깊이에서 일관되게 앞섭니다(상대적으로 +12~47%).
- 두 arm 모두 chance(1.6% @3)의 5.7~8.4배 → 둘 다 실제로 검색을 하고 있고, 그 위에서 gaze가 더 잘합니다.

120문항을 뺀 **held-out 380문항**만으로도 같은 결론이고, 엄격한 tol=0에서도 방향과 크기가 유지됩니다
(120문항은 어떤 튜닝에도 쓰이지 않았지만, 그래도 분리해서 확인했습니다):

| 문항 집합 | tol | arm | R@3 | R@10 | med. rank | chance@3 | paired at k=3 |
|---|---|---|---|---|---|---|---|
| **500문항 (487 채점)** | **0s** | `full` | 3.1% | 8.4% | 289 | 0.34% | — |
| **500문항 (487 채점)** | **0s** | **`gazef_05`** | **5.1%** | 12.7% | **249** | 0.34% | +2.1 %p, 19:9, p=0.087 |
| holdout 380 | 60s | `full` | 11.8% | 25.8% | 57 | 1.67% | — |
| holdout 380 | 60s | `gazef_05` | **16.3%** | 31.8% | **35** | 1.67% | +4.5 %p, 34:17, **p=0.024** |
| holdout 367¹ | 0s | `full` | 3.8% | 10.6% | 204 | 0.36% | — |
| holdout 367¹ | 0s | `gazef_05` | **6.5%** | 16.1% | **142** | 0.36% | +2.7 %p, 18:8, p=0.076 |

¹ tol=0에서 정답 칸이 0개가 되는 13문항 제외(500문항 기준으로도 같은 13문항).

500문항 tol=0 행은 `main_tol0`(120문항)과 `holdout_tol0`(380문항)을 [`merge_recall.py`](merge_recall.py)로
합친 것입니다 — 같은 임베딩·같은 풀·같은 tolerance·겹치지 않는 문항이라 한 번에 돌린 것과 산술적으로
동일하고(검산 완료), GPU가 필요 없습니다.

### 3.4 어디서 이기나 (500문항, tol=60, k=3)

절대 %p는 베이스라인에 끌려다니므로 **배율과 불일치 건수를 같이** 봐야 합니다. 층별 불일치가 5~40건이라
대부분의 층은 그 자체로는 판정되지 않습니다.

| 분해 | n | `full` | `gazef_05` | Δ | 배율 | 불일치 (gazef:full) | p |
|---|---|---|---|---|---|---|---|
| need_audio = False (시각 근거) | 298 | 12.4% | 15.1% | +2.7 | 1.22× | 40 (24:16) | 0.268 |
| **need_audio = True (대화 근거)** | 202 | 4.5% | 10.9% | **+6.4** | **2.44×** | 15 (14:1) | **0.001** |
| EntityLog | 125 | 9.6% | 12.8% | +3.2 | 1.33× | 12 (8:4) | 0.388 |
| EventRecall | 126 | 5.6% | 10.3% | +4.8 | 1.86× | 8 (7:1) | 0.070 |
| RelationMap | 125 | 14.4% | 21.6% | +7.2 | 1.50× | 25 (17:8) | 0.108 |
| TaskMaster | 63 | 3.2% | 11.1% | +7.9 | 3.50× | 5 (5:0) | 0.062 |
| HabitInsight | 61 | 11.5% | 6.6% | −4.9 | 0.57× | 5 (1:4) | 0.375 |
| gap 0 (당일) | 314 | 10.2% | 15.9% | +5.7 | 1.56× | — | — |
| gap 1 | 94 | 4.3% | 7.4% | +3.1 | 1.72× | — | — |
| gap 2+ | 92 | 10.9% | 10.9% | 0.0 | 1.00× | — | — |

**읽는 법이 중요합니다.** 층 하나하나는 전체(불일치 55건)의 일부를 나눠 쓰는 것이라, 예를 들어
HabitInsight의 −4.9 %p는 **문항 3개 차이**(불일치 5건, p=0.375)입니다 — "반복 습관에서는 crop이 해롭다"로
읽을 근거가 없습니다. EntityLog의 +3.2 %p도 p=0.388입니다. 층별로 살아 있는 결론은 하나뿐입니다:

- **`need_audio=True`에서 +6.4 %p, 불일치 15건 중 14:1, p=0.001.** 전체 이득의 대부분이 여기서 나옵니다.
  가설(시각 근거 문항에서 더 커야 한다)과 정반대입니다.
- 가설대로 움직여야 했던 `need_audio=False`는 +2.7 %p이지만 불일치 40건에 24:16, p=0.268 — **방향만 맞고
  크기는 확정되지 않습니다.**
- 사람 이름이 들어간 질문(n=155)은 `full` 3.9% → `gazef` 9.7%(배율 2.50×), `Who`로 시작하는 질문(n=158)은
  14.6% → 20.3%(+5.7 %p). 배율은 모든 하위집합에서 1.2~3.5× 사이로, **특정 유형에 국한된 이득이 아닙니다.**

### 3.5 정확도(E′)는 아직 움직이지 않았다

6,223클립 인덱스, 120문항, 나머지 전부 동일:

| arm / 조건 | 정확도 |
|---|---|
| _B (텍스트만)_ | _40.8%_ |
| _E (에이전트 자율)_ | _41.7%_ |
| _E′ (저자 임베딩, visual 강제)_ | _45.8%_ |
| `full`, visual top-3 | 45.0% |
| `gazef_05`, visual top-3 | 45.0% |
| `full`, visual top-20 | **46.7%** |
| `gazef_05`, visual top-20 | 45.0% |

k=3에서 Δ = 0.0 %p, k=20에서 −1.7 %p(문항 2개). **이 표는 gaze를 반박하지 못합니다** — 이 120문항에서는
두 arm의 recall@3이 0.8%(정답 1개)로 같았으므로, 애초에 정확도가 달라질 재료가 없었습니다.
recall 이득이 실제로 있는 380문항에는 E′를 아직 돌리지 않았습니다(조건 B의 텍스트 컨텍스트가 120개만
캐시돼 있음).

### 3.6 덤으로 나온 발견: 120문항 subset은 retrieval이 비정상적으로 어렵다

같은 실행 안에서 문항 집합만 갈라 보면:

| 문항 집합 | n | `full` R@3 | `gazef_05` R@3 | `full` med. rank |
|---|---|---|---|---|
| visual_bottleneck 120문항 subset | 120 | **0.8%** | 4.2% | 415 |
| 나머지 380문항 | 380 | **11.8%** | 16.3% | 57 |

500문항 중 `full`의 적중 46건이 무작위로 흩어졌다면 120문항에는 11건이 들어가야 하는데 **1건**입니다
(hypergeometric P(X≤1) = 2.9e-5; `gazef`도 5 vs 기대 16, P = 1.9e-4). 유형·`need_audio`·gap 모든 층에서
같은 방향이므로 층화 구성만으로는 설명되지 않습니다. subset은 유형당 24개씩 균등 층화하면서
`need_audio=True`와 gap≥1을 자연 분포보다 많이 뽑았고, `target_clips` 1~10칸 조건으로 걸렀습니다 —
이 중 무엇이 retrieval 난이도와 얽혔는지는 아직 모릅니다.

**이게 중요한 이유**: visual_bottleneck의 "recall@3 = 5%"와 gaze 2단계의 무결과가 모두 이 120문항에서
나온 숫자입니다. 나머지 380문항에서 같은 검색기가 R@3 11.8%(tol=60) / 3.8%(tol=0)를 내므로,
**"WorldMM의 visual retrieval은 5%다"는 subset 특성이 섞인 값**으로 읽어야 합니다.

## 4. 해석

1. **프레이밍은 실제로 병목의 일부입니다.** 모델·프레임·질의·풀을 전부 고정하고 crop 박스만 바꿨는데
   recall@3이 9.2% → 13.4%(+46% 상대)로 오르고 순위 중앙값이 99 → 76으로 내려갑니다. 임베딩이 배경에
   희석되고 있었다는 증거입니다.
2. **그러나 "gaze 덕"이라고는 아직 말할 수 없습니다.** 1단계(623클립, keywords 질의)에서는
   `center@0.5`가 `gazef@0.5`보다 높았습니다(19.2% vs 15.0%). 즉 **이득의 상당 부분이 자르기 자체**일
   가능성이 살아 있고, 그것을 가르는 `center@0.5`를 500문항·6,223클립에서 **아직 측정하지 않았습니다.**
   이 실험이 현재 지지하는 주장은 "gaze crop > full"까지이고, "gaze > center"는 미결입니다.
   "움직임이 이유인가"는 `gazef_05_randf`로만 끊을 수 있고, 그것도 아직 측정하지 않았습니다.
3. **검정력이 결론을 뒤집었습니다.** 120문항에서는 불일치 2~4건으로 "효과 없음"처럼 보였고, 같은 임베딩을
   500문항으로 채점하니 불일치 55건에 p=0.006이 나왔습니다. recall 채점은 LLM이 필요 없어 문항을 늘리는
   비용이 거의 0이므로, **작은 subset으로 검색 실험을 판정하면 안 된다**는 것이 이 실험의 방법론적 교훈입니다.
4. **recall 이득이 답으로 바뀔지는 아직 모릅니다.** 정확도는 recall이 둘 다 ~0인 120문항에서만 재봤습니다.
   또한 이전 실험에서 E′가 recall 5%인 상태로도 B보다 +5.0 %p였던 것은 "정답을 찾아와서"가 아니라
   "근처 시각 컨텍스트가 있어서"일 가능성이 있었는데, 그렇다면 recall +4 %p가 정확도로 얼마나 환산되는지는
   선형적이지 않을 수 있습니다.
5. **이득이 가설과 반대인 층에서 나옵니다 — 그리고 그게 메커니즘을 다시 생각하게 합니다.**
   기대는 "시각 근거 문항(`need_audio=False`)에서 더 크다"였는데, 유의한 이득은 `need_audio=True`
   (대화 근거, +6.4 %p, p=0.001) 쪽입니다. 가능한 설명 셋:
   - **헤드룸**: `need_audio=False`는 `full`이 이미 12.4%에서 출발합니다(요리·물건 조작처럼 장면 자체가
     특이해서 전체 프레임으로도 잡히는 순간). `need_audio=True`는 4.5%에서 출발합니다 — 오를 자리가 다릅니다.
   - **변별이지 인식이 아님**: 대화 장면은 같은 집·같은 방·같은 테이블이 반복돼 전체 프레임끼리 서로
     구별되지 않습니다. gaze crop은 그 순간의 화자·얼굴·손으로 좁혀 **distractor와의 차이**를 만듭니다.
     즉 gaze의 기여가 "쿼리의 객체를 찾아 준다"보다 **"수천 개의 비슷한 클립 중 이 클립을 구별해 준다"**
     쪽일 수 있습니다. 사람 이름 질문에서 배율이 2.5×인 것이 이와 같은 방향입니다.
   - **층별 검정력 부족**: EntityLog(+3.2, p=0.388)와 HabitInsight(−4.9, p=0.375)는 각각 불일치가
     12건·5건뿐입니다. "EntityLog에서 안 올랐다"도 "HabitInsight에서 손해다"도 아직 읽을 수 없습니다.

   둘째 설명이 맞는지는 값싸게 갈릴 수 있습니다: 쿼리에서 객체 명사만 남긴 질의와 사람 이름만 남긴 질의로
   각각 recall을 재면 됩니다(임베딩 재사용, GPU 불필요).

## 5. 한계와 주의사항

- **피험자 1명(A1_JAKE), 임베딩 모델 1개(VLM2Vec-V2.0), ratio 1개(0.5).** 스윕이 없습니다.
- **gaze 기하는 근사입니다.** 결합 방향과 vergence는 Aria 문서와 일치하고 depth 역산 오차 0.022 m로
  검증했지만, 픽셀 투영은 calibration 없이 단일 focal + equidistant이고 CPF→RGB 평행이동(~3 cm)을
  무시합니다 → median depth 0.59 m에서 ~2.9° ≈ 31 px 오차(crop 박스는 704 px).
- **transform은 사람이 그림으로 골랐습니다**(`flipx+flipy`). 캡션 점수로는 끝까지 무승부였고, 다른
  피험자로 옮길 때는 `--axis x --min-offset 0.08`로 12장 이상 보고 다시 정해야 합니다. 증거는 `overlay/`.
- **tolerance를 섞어 읽으면 안 됩니다.** 헤드라인 +4.2 %p는 tol=60(정답 칸 중앙값 5개, chance@3 1.6%)이고,
  엄격한 tol=0에서는 500문항 +2.1 %p (p=0.087), holdout 380문항 +2.7 %p (p=0.076)입니다. 방향은 세 곳에서
  모두 같고 절대값도 chance의 6~15배지만, **엄격 기준으로는 아직 p<0.05가 아닙니다.** A1_JAKE 500문항이
  문항의 전부이므로 남은 길은 arm 수를 줄여 검정력을 몰아주거나 피험자를 늘리는 것입니다.
- **tolerance가 옛 결과 파일에는 기록돼 있지 않습니다.** `recall_eval.py`가 이제 `target_tolerance_sec`와
  `chance`를 결과 JSON·마크다운에 남기지만, 9/27에 만든 파일들은 그 이전 형식이라 `digest.py`가
  실행 명령에서 추정한 값(`BLOCKS`의 폴백)을 "추정"으로 표시해 씁니다. 다음 채점 때 자동으로 사라집니다.
- **2단계 정확도와 transform은 기록본에서 읽습니다**(git에 `results_qa/`·`gaze_points.json`이 없는 곳).
  숫자의 출처는 2026-09-27 stage-2 실행이며 `analysis/stage2_recorded.json`에 적혀 있습니다.
- **`pkl` arm은 참조점입니다.** 저자 임베딩은 mp4 경로로 프레임 샘플링을 스스로 하므로 `full`과 경로가
  다릅니다. 비교는 항상 `full` 기준으로 했습니다.
- gaze CSV가 없는 클립은 gaze arm에서 조용히 중앙 crop으로 대체되지 않고 빠집니다(`unscorable` 0건이었음).

## 6. 다음 단계

우선순위 순. 1~2는 LLM이 필요 없고 GPU만 씁니다.

1. **`center@0.5`와 `gazef_05_randf`를 500문항·6,223클립에서 측정.** 이 실험에 남은 단 하나의 미결
   질문(자르기냐 / 움직임이냐 / 시선이냐)을 가르는 실험입니다. 임베딩만 arm당 ~5,600클립(각 ~2시간)
   늘리면 채점은 무료이고, 1단계 keywords에서 `center > gazef`였으므로 **결과가 뒤집힐 실질적 위험**이
   있습니다. [`run_controls.sh`](run_controls.sh)가 preflight → 두 arm 임베딩 → 4 arm 채점(tol 0/60) →
   다이제스트까지 이어서 합니다:

   ```bash
   cd ~/WorldMM && source .venv/bin/activate
   nohup bash experiments/gaze_crop/run_controls.sh > /dev/null 2>&1 &
   tail -f experiments/gaze_crop/logs/controls_*.log
   ```

   죽어도 같은 명령을 다시 부르면 이어받습니다(`embed_arms.py`가 클립 키 기준). 한쪽만 돌릴 때는
   `SKIP_CENTER=1` / `SKIP_RANDF=1`, 채점만 다시 할 때는 둘 다 1로 주면 됩니다.

   판정은 **두 쌍만** 봅니다 — `gazef − center`(자르기), `gazef − randf`(움직임). arm 4개면 pair가 6개고,
   6개를 다 본 뒤 이긴 쌍을 고르면 그게 다중비교입니다.

2. **인덱스 확장(가장 값싸고 유망): `full` + `gazef` 둘 다 인덱스에 넣고 클립 점수 = max(두 벡터).**
   현재는 arm을 **교체**하고 있어 `full`만 맞히던 문항(17건)을 잃습니다. 불일치 55건 중 38:17이라는
   구조는 두 벡터가 서로 다른 문항을 맞힌다는 뜻이고, 합치면 이론상 R@3 ≈ 20%까지 올라갑니다.
   새 임베딩이 전혀 필요 없고 채점만 바꾸면 됩니다.
3. **ratio 스윕과 crop 변형**: R = 0.3 / 0.5 / 0.7, 프레임별 gaze의 시간 smoothing(흔들림 감소),
   `gazef` 16프레임 중 일부만 crop하는 혼합. 3.4의 "사람이 커져서 이득"이 맞다면 작은 R이 더 좋아야 합니다.
4. **정확도를 recall 이득이 있는 곳에서 재기**: held-out 380문항에 조건 B를 돌려 `text_context`를 만들고
   (LLM 비용 발생), 그 위에서 E′를 `full` / `gazef` / (1의 승자)로 실행. 지금까지의 정확도 표는 recall이
   둘 다 0인 표본에서 나온 것이라 사실상 정보가 없습니다.
5. **120문항 subset이 왜 retrieval-hard인지 진단**(3.6). visual_bottleneck의 "recall@3 5%"와 D−B +12.2 %p가
   모두 이 표본에서 나왔으므로, 편향의 출처(층화? `target_clips` 필터? 질문 문구?)를 찾으면 이전 결론의
   일반화 범위가 정리됩니다. 380문항에 대한 oracle/DPR 진단을 같이 돌리면 값싸게 확인됩니다.
6. **gaze의 기여가 "객체 인식"인지 "클립 변별"인지 가르기**(4-5의 둘째 설명): 같은 임베딩에
   `--query-source`를 객체 명사만 / 사람 이름만으로 바꿔 넣어 recall을 재봅니다. 전자에서만 오르면
   "중요한 객체를 크게 만들어 주는 것"이고, 후자에서도 오르면 "비슷한 장면들 사이의 변별을 만드는 것"입니다.
   후자라면 crop 반경을 더 줄이는 쪽이 유망합니다. GPU 불필요.
7. **다른 피험자로 외적 타당성 확인**: A2 등에서 transform을 `--axis x` 절차로 다시 정하고 같은 측정.
   지금 결론은 전부 A1_JAKE 한 명, 한 카메라 기하에서 나온 것입니다.
8. **tolerance 정책 확정**: 논문/보고용 헤드라인은 tol=0(엄격) + chance 대비로 쓰고, tol=60은 보조로만.
   500문항 엄격 채점이 +2.1 %p·p=0.087이므로, 문항을 더 늘릴 수 없는 상황에서는 arm 수를 줄여 검정력을
   몰아주는 편이 낫습니다.

## 7. 재현

```bash
cd ~/WorldMM && source .venv/bin/activate

# 1단계 (623클립, ~40분). GAZE_TRANSFORM 없이 부르면 후보 그림만 그리고 멈춘다
bash experiments/gaze_crop/run_stage1.sh                      # -> variants/*.jpg 보고 transform 고르기
GAZE_TRANSFORM=flipx+flipy bash experiments/gaze_crop/run_stage1.sh

# 2단계 (전체 6,223클립 + E′, 3~4시간). 1단계 승자를 자동으로 가져간다
VISUAL_TOP_K="3 20" bash experiments/gaze_crop/run_stage2.sh

# 500문항 확장 채점 (LLM/GPU 불필요, 임베딩만 있으면 됨)
python experiments/gaze_crop/build_questions.py                                     # questions_500.json
python experiments/gaze_crop/build_pool.py --all-clips --questions experiments/gaze_crop/questions_500.json \
    --out experiments/gaze_crop/pool_all.json
python experiments/gaze_crop/recall_eval.py --pool experiments/gaze_crop/pool_all.json \
    --arms full gazef@0.5 --target-tolerance-sec 60 \
    --out experiments/gaze_crop/results/recall_all_tol60.json \
    --markdown experiments/gaze_crop/analysis/recall_all_tol60.md
```

## 8. 파일

파일 이름 규칙: `recall_{문항집합}_{tolerance}.{json,md}` — 문항집합은 `stage1`(623클립 풀) ·
`main`(120문항 subset) · `holdout`(나머지 380) · `all`(500문항 전부), tolerance는 `tol0`(엄격) ·
`tol60`(±60초). `recall_eval.py`의 기본 출력은 `recall_scratch_*`로 떨어지므로 이름 붙일 결과는
`--out`/`--markdown`을 항상 명시합니다.

| 경로 | 내용 |
|---|---|
| [`PLAN.md`](PLAN.md) | 설계 전문 (arm 구성 근거, gaze 기하, transform 결정 로그, 판정 기준) |
| `analysis/recall_all_tol60.md` · `results/recall_all_tol60.json` | **헤드라인**: 500문항 · 6,223클립 · tol=60 |
| `analysis/recall_all_tol0.md` · `results/recall_all_tol0.json` | 같은 500문항, 엄격 채점. 두 실행을 합친 것 |
| `analysis/recall_holdout_tol60.md`, `recall_holdout_tol0.md` | held-out 380문항 (tol=60 / tol=0) |
| `analysis/recall_main_tol60.md`, `recall_main_tol0.md` | 120문항, 6,223클립 (tol=60 / tol=0) |
| `results/recall_stage1_question_tol0.json`, `analysis/recall_stage1_keywords_tol0.md` | 1단계 623클립, 6개 arm (질문 질의 / 키워드 질의) |
| `analysis/DIGEST.md` | **먼저 볼 것.** 헤드라인 + 모든 recall 표 + 정확도. `digest.py`로 재생성 |
| `analysis/stage2_recorded.json` | git에 없는 산출물(transform, 2단계 정확도)의 기록본. 원본이 있으면 무시됨 |
| `questions_500.json`, `pool.json`, `pool_holdout.json` | 문항 전체 / 623클립 풀 / holdout 풀 |
| `variants/*.jpg`, `overlay/*.jpg`¹ | transform 후보 그림과 선택 증거 |
| [`prepare_gaze.py`](prepare_gaze.py) | yaw/pitch CSV → 정규화된 `gaze_points.json` (+overlay) |
| [`embed_arms.py`](embed_arms.py) | 1회 디코딩 → arm별 crop → `encode_video` → `emb/<arm>.npz` (이어받기) |
| [`recall_eval.py`](recall_eval.py) | recall@k, paired McNemar, chance 수준, 분해표 |
| [`compare_transforms.py`](compare_transforms.py) | transform 후보 그림 + 캡션 코사인 점수 (**고르는 건 사람**) |
| [`build_questions.py`](build_questions.py), [`build_pool.py`](build_pool.py) | 500문항 / 클립 풀 구성 |
| [`merge_recall.py`](merge_recall.py) | 문항이 겹치지 않는 recall 실행 둘을 합쳐 한 표로 (GPU 불필요) |
| [`export_pkl.py`](export_pkl.py), [`rescore.py`](rescore.py) | arm 임베딩 → E′ 입력 / 저장된 응답 재채점 |
| [`gate.py`](gate.py), [`digest.py`](digest.py) | 1단계 판정(종료 코드) / 요약 생성 |
| [`run_controls.sh`](run_controls.sh) | 대조군(`center@0.5`, `gazef_05_randf`) 임베딩 + 4 arm 채점 (~4시간) |
| `logs/stage{1,2}_*.log` | 전체 실행 로그 (clamp 비율, depth 검증, ETA 포함) |

¹ `emb/`, `overlay/`, `gaze_points.json`, `pool_all.json`, `results_qa/`는 재생성 가능해서 git에 없습니다.
