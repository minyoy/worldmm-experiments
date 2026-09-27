# gaze-crop 실험 다이제스트

### gaze→이미지 transform

- 실제로 쓴 값: **flipx+flipy** (사람이 고름; 근거 이미지는 `overlay/`, `variants/`)
- 캡션 코사인 순위(참고): flipy > flipx+flipy / 36클립
- 1위 `flipy` vs 2위 `flipx+flipy`: **tie(구분 안 됨)** (95% CI -0.0090..+0.0124)
- 참고: 점수 1위는 `flipy` 인데 `flipx+flipy` 를 골랐음 — 이미지 판단을 우선한 결과

### 1단계 recall (623클립 풀)

pool pool.json, 120/120 문항 채점

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `center_05` | 0.8% | 3.3% | 5.0% | 7.5% | 12.5% | 22.5% | 118 |
| `full` | 0.8% | 2.5% | 2.5% | 6.7% | 15.8% | 27.5% | 114 |
| `gaze_05` | 0.8% | 2.5% | 3.3% | 6.7% | 12.5% | 26.7% | 98 |
| `gazef_05` | 1.7% | 3.3% | 4.2% | 7.5% | 12.5% | 28.3% | 109 |
| `gazef_05_rand` | 0.8% | 2.5% | 2.5% | 5.8% | 7.5% | 24.2% | 120 |
| `pkl` | 1.7% | 1.7% | 4.2% | 8.3% | 14.2% | 30.8% | 111 |

paired diff at k=3:
- `center_05` − `full` = **+0.8pp** (95% CI +0.0..+2.5; center_05 only 1, full only 0)
- `center_05` − `gaze_05` = **+0.8pp** (95% CI -1.7..+4.2; center_05 only 2, gaze_05 only 1)
- `center_05` − `gazef_05` = **+0.0pp** (95% CI -2.5..+2.5; center_05 only 1, gazef_05 only 1)
- `center_05` − `gazef_05_rand` = **+0.8pp** (95% CI -1.7..+3.3; center_05 only 2, gazef_05_rand only 1)
- `center_05` − `pkl` = **+1.7pp** (95% CI +0.0..+4.2; center_05 only 2, pkl only 0)
- `full` − `gaze_05` = **+0.0pp** (95% CI -2.5..+2.5; full only 1, gaze_05 only 1)
- `full` − `gazef_05` = **-0.8pp** (95% CI -2.5..+0.0; full only 0, gazef_05 only 1)
- `full` − `gazef_05_rand` = **+0.0pp** (95% CI -2.5..+2.5; full only 1, gazef_05_rand only 1)
- `full` − `pkl` = **+0.8pp** (95% CI +0.0..+2.5; full only 1, pkl only 0)
- `gaze_05` − `gazef_05` = **-0.8pp** (95% CI -3.3..+1.7; gaze_05 only 1, gazef_05 only 2)
- `gaze_05` − `gazef_05_rand` = **+0.0pp** (95% CI -2.5..+2.5; gaze_05 only 1, gazef_05_rand only 1)
- `gaze_05` − `pkl` = **+0.8pp** (95% CI +0.0..+2.5; gaze_05 only 1, pkl only 0)
- `gazef_05` − `gazef_05_rand` = **+0.8pp** (95% CI -1.7..+3.3; gazef_05 only 2, gazef_05_rand only 1)
- `gazef_05` − `pkl` = **+1.7pp** (95% CI +0.0..+4.2; gazef_05 only 2, pkl only 0)
- `gazef_05_rand` − `pkl` = **+0.8pp** (95% CI +0.0..+2.5; gazef_05_rand only 1, pkl only 0)

need_audio 분해 (False 행이 실제 여유분):
- False (n=70): center_05 2.9%, full 1.4%, gaze_05 1.4%, gazef_05 1.4%, gazef_05_rand 1.4%, pkl 1.4%
- True (n=50): center_05 4.0%, full 4.0%, gaze_05 4.0%, gazef_05 6.0%, gazef_05_rand 4.0%, pkl 2.0%

### 2단계 recall (6,223클립 = 실제 인덱스 크기)

pool pool_all.json, 120/120 문항 채점

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 0.0% | 0.8% | 0.8% | 1.7% | 2.5% | 3.3% | 878 |
| `gazef_05` | 0.8% | 0.8% | 1.7% | 2.5% | 2.5% | 5.8% | 909 |

paired diff at k=3:
- `full` − `gazef_05` = **+0.0pp** (95% CI -2.5..+2.5; full only 1, gazef_05 only 1)

need_audio 분해 (False 행이 실제 여유분):
- False (n=70): full 1.4%, gazef_05 0.0%
- True (n=50): full 0.0%, gazef_05 2.0%

### 2단계 정확도 (조건 E′, 인덱스만 교체)

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

파서 확인: `python experiments/gaze_crop/rescore.py /home/minyoy/WorldMM/experiments/gaze_crop/results_qa/*/E_prime.json`
