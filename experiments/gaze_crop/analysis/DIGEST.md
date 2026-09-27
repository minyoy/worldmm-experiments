# gaze-crop 실험 다이제스트

### 헤드라인

가장 검정력 높은 측정: **★ A1_JAKE 500문항 전부 (6,223클립), tolerance 60s** — 500문항, tolerance 60s (정답 칸이 늘어나므로 엄격 표와 직접 비교 금지) **[파일에 기록 없음 — 실행 명령에서 추정]** (`results/recall_all_tol60.json`)
  - k=3: `gazef_05` 가 `full` 보다 **+4.2pp** (불일치 55건, p=0.006)
같은 비교, 엄격 채점(tol 0s): **★ A1_JAKE 500문항 전부 (6,223클립), tolerance 0s (엄격)** — 487문항, tolerance 0s (엄격: EgoLife 가 태그한 30초 칸만 정답) (`results/recall_all_tol0.json`)
  - k=3: `gazef_05` 가 `full` 보다 **+2.1pp** (불일치 28건, p=0.087 — **유의하지 않음**)

### gaze→이미지 transform

- 실제로 쓴 값: **flipx+flipy** — 사람이 variants/*.jpg 를 보고 선택 (12장). 캡션 코사인 점수는 tie **[기록본: stage2_recorded.json, `gaze_points.json` 없음]**
- 캡션 코사인 점수(참고): 점수 1위 flipy > 2위 flipx+flipy / 36클립, 1·2위 격차 95% CI -0.0090..+0.0124 (구분 안 됨)

### 1단계 recall (623클립 풀, 120문항, 질문 질의)

pool pool.json, 120/120 문항 채점, tolerance 0s (엄격: EgoLife 가 태그한 30초 칸만 정답) **[파일에 기록 없음 — 실행 명령에서 추정]**

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `center_05` | 0.8% | 3.3% | 5.0% | 7.5% | 12.5% | 22.5% | 118 |
| `full` | 0.8% | 2.5% | 2.5% | 6.7% | 15.8% | 27.5% | 114 |
| `gaze_05` | 0.8% | 2.5% | 3.3% | 6.7% | 12.5% | 26.7% | 98 |
| `gazef_05` | 1.7% | 3.3% | 4.2% | 7.5% | 12.5% | 28.3% | 109 |
| `gazef_05_rand` | 0.8% | 2.5% | 2.5% | 5.8% | 7.5% | 24.2% | 120 |
| `pkl` | 1.7% | 1.7% | 4.2% | 8.3% | 14.2% | 30.8% | 111 |

paired diff at k=3:
- `center_05` − `full` = **+0.8pp** (center_05 only 1, full only 0 (옛 형식: 정확검정 p값 없음))
- `center_05` − `gaze_05` = **+0.8pp** (center_05 only 2, gaze_05 only 1 (옛 형식: 정확검정 p값 없음))
- `center_05` − `gazef_05` = **+0.0pp** (center_05 only 1, gazef_05 only 1 (옛 형식: 정확검정 p값 없음))
- `center_05` − `gazef_05_rand` = **+0.8pp** (center_05 only 2, gazef_05_rand only 1 (옛 형식: 정확검정 p값 없음))
- `center_05` − `pkl` = **+1.7pp** (center_05 only 2, pkl only 0 (옛 형식: 정확검정 p값 없음))
- `full` − `gaze_05` = **+0.0pp** (full only 1, gaze_05 only 1 (옛 형식: 정확검정 p값 없음))
- `full` − `gazef_05` = **-0.8pp** (full only 0, gazef_05 only 1 (옛 형식: 정확검정 p값 없음))
- `full` − `gazef_05_rand` = **+0.0pp** (full only 1, gazef_05_rand only 1 (옛 형식: 정확검정 p값 없음))
- `full` − `pkl` = **+0.8pp** (full only 1, pkl only 0 (옛 형식: 정확검정 p값 없음))
- `gaze_05` − `gazef_05` = **-0.8pp** (gaze_05 only 1, gazef_05 only 2 (옛 형식: 정확검정 p값 없음))
- `gaze_05` − `gazef_05_rand` = **+0.0pp** (gaze_05 only 1, gazef_05_rand only 1 (옛 형식: 정확검정 p값 없음))
- `gaze_05` − `pkl` = **+0.8pp** (gaze_05 only 1, pkl only 0 (옛 형식: 정확검정 p값 없음))
- `gazef_05` − `gazef_05_rand` = **+0.8pp** (gazef_05 only 2, gazef_05_rand only 1 (옛 형식: 정확검정 p값 없음))
- `gazef_05` − `pkl` = **+1.7pp** (gazef_05 only 2, pkl only 0 (옛 형식: 정확검정 p값 없음))
- `gazef_05_rand` − `pkl` = **+0.8pp** (gazef_05_rand only 1, pkl only 0 (옛 형식: 정확검정 p값 없음))

need_audio 분해 (False 행이 실제 여유분):
- False (n=70): center_05 2.9%, full 1.4%, gaze_05 1.4%, gazef_05 1.4%, gazef_05_rand 1.4%, pkl 1.4%
- True (n=50): center_05 4.0%, full 4.0%, gaze_05 4.0%, gazef_05 6.0%, gazef_05_rand 4.0%, pkl 2.0%

### 1단계 recall (623클립 풀, 120문항, 키워드 질의)

pool pool.json, 120/120 문항 채점, tolerance 0s (엄격: EgoLife 가 태그한 30초 칸만 정답) **[파일에 기록 없음 — 실행 명령에서 추정]**

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `center_05` | 8.3% | 19.2% | 22.5% | 30.8% | 36.7% | 53.3% | 35 |
| `full` | 10.8% | 12.5% | 15.0% | 26.7% | 36.7% | 50.8% | 48 |
| `gaze_05` | 10.8% | 17.5% | 20.0% | 27.5% | 39.2% | 57.5% | 31 |
| `gazef_05` | 8.3% | 15.0% | 20.0% | 30.0% | 38.3% | 53.3% | 41 |
| `gazef_05_rand` | 7.5% | 11.7% | 17.5% | 28.3% | 34.2% | 49.2% | 53 |
| `pkl` | 10.0% | 16.7% | 18.3% | 28.3% | 36.7% | 51.7% | 47 |

paired diff at k=3:
- `center_05` − `full` = **+6.7pp** (center_05 only 10, full only 2 (옛 형식: 정확검정 p값 없음))
- `center_05` − `gaze_05` = **+1.7pp** (center_05 only 6, gaze_05 only 4 (옛 형식: 정확검정 p값 없음))
- `center_05` − `gazef_05` = **+4.2pp** (center_05 only 8, gazef_05 only 3 (옛 형식: 정확검정 p값 없음))
- `center_05` − `gazef_05_rand` = **+7.5pp** (center_05 only 11, gazef_05_rand only 2 (옛 형식: 정확검정 p값 없음))
- `center_05` − `pkl` = **+2.5pp** (center_05 only 9, pkl only 6 (옛 형식: 정확검정 p값 없음))
- `full` − `gaze_05` = **-5.0pp** (full only 2, gaze_05 only 8 (옛 형식: 정확검정 p값 없음))
- `full` − `gazef_05` = **-2.5pp** (full only 2, gazef_05 only 5 (옛 형식: 정확검정 p값 없음))
- `full` − `gazef_05_rand` = **+0.8pp** (full only 6, gazef_05_rand only 5 (옛 형식: 정확검정 p값 없음))
- `full` − `pkl` = **-4.2pp** (full only 0, pkl only 5 (옛 형식: 정확검정 p값 없음))
- `gaze_05` − `gazef_05` = **+2.5pp** (gaze_05 only 6, gazef_05 only 3 (옛 형식: 정확검정 p값 없음))
- `gaze_05` − `gazef_05_rand` = **+5.8pp** (gaze_05 only 9, gazef_05_rand only 2 (옛 형식: 정확검정 p값 없음))
- `gaze_05` − `pkl` = **+0.8pp** (gaze_05 only 6, pkl only 5 (옛 형식: 정확검정 p값 없음))
- `gazef_05` − `gazef_05_rand` = **+3.3pp** (gazef_05 only 9, gazef_05_rand only 5 (옛 형식: 정확검정 p값 없음))
- `gazef_05` − `pkl` = **-1.7pp** (gazef_05 only 4, pkl only 6 (옛 형식: 정확검정 p값 없음))
- `gazef_05_rand` − `pkl` = **-5.0pp** (gazef_05_rand only 5, pkl only 11 (옛 형식: 정확검정 p값 없음))

need_audio 분해 (False 행이 실제 여유분):
- False (n=70): center_05 25.7%, full 18.6%, gaze_05 25.7%, gazef_05 21.4%, gazef_05_rand 14.3%, pkl 24.3%
- True (n=50): center_05 10.0%, full 4.0%, gaze_05 6.0%, gazef_05 6.0%, gazef_05_rand 8.0%, pkl 6.0%

### 2단계 recall (6,223클립 = 실제 인덱스, 120문항, tol 0s)

pool pool_all.json, 120/120 문항 채점, tolerance 0s (엄격: EgoLife 가 태그한 30초 칸만 정답) **[파일에 기록 없음 — 실행 명령에서 추정]**

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 0.0% | 0.8% | 0.8% | 1.7% | 2.5% | 3.3% | 878 |
| `gazef_05` | 0.8% | 0.8% | 1.7% | 2.5% | 2.5% | 5.8% | 909 |

paired diff at k=3:
- `full` − `gazef_05` = **+0.0pp** (불일치 2개 중 full 1 : gazef_05 1, p=1.000 — **유의하지 않음**)

need_audio 분해 (False 행이 실제 여유분):
- False (n=70): full 1.4%, gazef_05 0.0%
- True (n=50): full 0.0%, gazef_05 2.0%

### 같은 120문항, tolerance 60s

pool pool_all.json, 120/120 문항 채점, tolerance 60s (정답 칸이 늘어나므로 엄격 표와 직접 비교 금지) **[파일에 기록 없음 — 실행 명령에서 추정]**

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 0.0% | 0.8% | 0.8% | 3.3% | 6.7% | 11.7% | 415 |
| `gazef_05` | 0.8% | 4.2% | 4.2% | 5.0% | 7.5% | 12.5% | 368 |

paired diff at k=3:
- `full` − `gazef_05` = **-3.3pp** (불일치 4개 중 full 0 : gazef_05 4, p=0.125 — **유의하지 않음**)

need_audio 분해 (False 행이 실제 여유분):
- False (n=70): full 1.4%, gazef_05 2.9%
- True (n=50): full 0.0%, gazef_05 6.0%

### held-out 380문항 (120문항 제외), tolerance 0s

pool pool_holdout.json, 367/380 문항 채점, tolerance 0s (엄격: EgoLife 가 태그한 30초 칸만 정답) **[파일에 기록 없음 — 실행 명령에서 추정]**

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 1.6% | 3.8% | 5.7% | 10.6% | 16.1% | 30.0% | 204 |
| `gazef_05` | 3.0% | 6.5% | 7.9% | 16.1% | 22.3% | 35.4% | 142 |

paired diff at k=3:
- `full` − `gazef_05` = **-2.7pp** (불일치 26개 중 full 8 : gazef_05 18, p=0.076 — **유의하지 않음**)

need_audio 분해 (False 행이 실제 여유분):
- False (n=217): full 4.6%, gazef_05 6.9%
- True (n=150): full 2.7%, gazef_05 6.0%

### held-out 380문항, tolerance 60s

pool pool_holdout.json, 380/380 문항 채점, tolerance 60s (정답 칸이 늘어나므로 엄격 표와 직접 비교 금지) **[파일에 기록 없음 — 실행 명령에서 추정]**

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 6.3% | 11.8% | 15.3% | 25.8% | 33.2% | 48.4% | 57 |
| `gazef_05` | 9.2% | 16.3% | 21.6% | 31.8% | 41.3% | 54.5% | 35 |

paired diff at k=3:
- `full` − `gazef_05` = **-4.5pp** (불일치 51개 중 full 17 : gazef_05 34, p=0.024)

need_audio 분해 (False 행이 실제 여유분):
- False (n=228): full 15.8%, gazef_05 18.9%
- True (n=152): full 5.9%, gazef_05 12.5%

### ★ A1_JAKE 500문항 전부 (6,223클립), tolerance 0s (엄격)

pool pool_all.json, 487/500 문항 채점, tolerance 0s (엄격: EgoLife 가 태그한 30초 칸만 정답)

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 1.2% | 3.1% | 4.5% | 8.4% | 12.7% | 23.4% | 289 |
| `gazef_05` | 2.5% | 5.1% | 6.4% | 12.7% | 17.5% | 28.1% | 249 |
| _chance_ | _0.11%_ | _0.34%_ | _0.57%_ | _1.14%_ | _2.27%_ | _5.39%_ | _1273_ |

paired diff at k=3:
- `full` − `gazef_05` = **-2.1pp** (불일치 28개 중 full 9 : gazef_05 19, p=0.087 — **유의하지 않음**)

need_audio 분해 (False 행이 실제 여유분):
- False (n=287): full 3.8%, gazef_05 5.2%
- True (n=200): full 2.0%, gazef_05 5.0%

### ★ A1_JAKE 500문항 전부 (6,223클립), tolerance 60s

pool pool_all.json, 500/500 문항 채점, tolerance 60s (정답 칸이 늘어나므로 엄격 표와 직접 비교 금지) **[파일에 기록 없음 — 실행 명령에서 추정]**

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 4.8% | 9.2% | 11.8% | 20.4% | 26.8% | 39.6% | 99 |
| `gazef_05` | 7.2% | 13.4% | 17.4% | 25.4% | 33.2% | 44.4% | 76 |

paired diff at k=3:
- `full` − `gazef_05` = **-4.2pp** (불일치 55개 중 full 17 : gazef_05 38, p=0.006)

need_audio 분해 (False 행이 실제 여유분):
- False (n=298): full 12.4%, gazef_05 15.1%
- True (n=202): full 4.5%, gazef_05 10.9%

### ★★ 대조군 포함 4 arm, 500문항, tolerance 0s (엄격)

아직 없음 — 만들려면: `bash experiments/gaze_crop/run_controls.sh  (center@0.5 + gazef_05_randf 임베딩 후 채점)`

### ★★ 대조군 포함 4 arm, 500문항, tolerance 60s

아직 없음 — 만들려면: `bash experiments/gaze_crop/run_controls.sh`

### 2단계 정확도 (조건 E′, 인덱스만 교체)

**[기록본: stage2_recorded.json — `results_qa/` 가 이 머신에 없음. 출처: visual_bottleneck/subset.json (120문항), pool pool_all.json (6,223 clips)]**

| arm | 정확도 |
|---|---|
| _B (텍스트만)_ | _40.8%_ |
| _E (에이전트 자율)_ | _41.7%_ |
| _E' (저자 임베딩, visual 강제)_ | _45.8%_ |
| `full_k20` | **46.7%** |
| `full_k3` | **45.0%** |
| `gazef_05_k20` | **45.0%** |
| `gazef_05_k3` | **45.0%** |

- k=3: `gazef_05` − `full` = **+0.0pp** (같은 풀·같은 플래그·같은 k의 자체 대조)
- k=20: `gazef_05` − `full` = **-1.7pp** (같은 풀·같은 플래그·같은 k의 자체 대조)

- ⚠ 이 정확도는 visual_bottleneck 의 120문항 subset 에서만 잰 값이다. 그 표본에서는 두 arm 의 recall@3 이 거의 0 이라(위 120문항 블록) 정확도가 달라질 재료 자체가 없다. recall 이득이 있는 문항에서 재려면 나머지 380문항에 조건 B 를 먼저 돌려 `visual_bottleneck/text_context/` 를 채워야 한다 (README 3.5, 6-4).
