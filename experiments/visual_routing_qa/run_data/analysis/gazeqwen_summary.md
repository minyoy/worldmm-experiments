# GazeQwen oracle 실험 요약

생성: `analyze_gazeqwen.py`. 문항별 결과 JSON(`run_data/results/oracle_gq_*_{sdpa,eager}`)에서 계산. 짝지은 부트스트랩(4,000회), 정확 부호검정. 다중비교 보정 없음.

- sdpa 완료 487문항 (실패: 없음)
- eager 완료 483문항 (실패: ['86', '338', '465', '484'])

## 1. sdpa

### 전체 (N=487)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 216 | 44.4% |
| `gq_lora_only` | 200 | 41.1% |
| `gq_inject_only` | 197 | 40.5% |
| `gq_gaze` | 199 | 40.9% |
| `gq_placebo` | 196 | 40.2% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -3.3 | [-7.4, +0.8] | 43 / 59 | 0.137 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -0.6 | [-3.7, +2.5] | 28 / 31 | 0.795 |
| `gq_gaze − gq_inject_only` | gaze content | +0.4 | [-0.6, +1.6] | 5 / 3 | 0.727 |
| `gq_gaze − gq_placebo` | gaze position | +0.6 | [+0.0, +1.4] | 3 / 0 | 0.250 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -0.2 | [-3.3, +2.9] | 27 / 28 | 1.000 |

답이 바뀐 문항: gaze vs inject_only 15, gaze vs placebo 3, gaze vs lora_only 79

fixation이 있는 문항 396개, 없는 문항 91개 (없는 문항에서는 `gq_gaze`가 구조적으로 `gq_lora_only`와 같다).

### fixation이 1개 이상인 문항 (N=396)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 177 | 44.7% |
| `gq_lora_only` | 163 | 41.2% |
| `gq_inject_only` | 160 | 40.4% |
| `gq_gaze` | 162 | 40.9% |
| `gq_placebo` | 159 | 40.2% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -3.5 | [-7.8, +0.8] | 33 / 47 | 0.146 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -0.8 | [-4.5, +2.8] | 25 / 28 | 0.784 |
| `gq_gaze − gq_inject_only` | gaze content | +0.5 | [+0.0, +1.3] | 2 / 0 | 0.500 |
| `gq_gaze − gq_placebo` | gaze position | +0.8 | [+0.0, +1.8] | 3 / 0 | 0.250 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -0.3 | [-3.8, +3.5] | 27 / 28 | 1.000 |

답이 바뀐 문항: gaze vs inject_only 3, gaze vs placebo 3, gaze vs lora_only 79

## 2. eager

### 전체 (성공한 문항) (N=483)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 208 | 43.1% |
| `gq_lora_only` | 200 | 41.4% |
| `gq_inject_only` | 184 | 38.1% |
| `gq_gaze` | 187 | 38.7% |
| `gq_placebo` | 188 | 38.9% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -1.7 | [-6.2, +3.1] | 62 / 70 | 0.543 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -3.3 | [-6.4, -0.4] | 19 / 35 | 0.040 |
| `gq_gaze − gq_inject_only` | gaze content | +0.6 | [-1.0, +2.1] | 9 / 6 | 0.607 |
| `gq_gaze − gq_placebo` | gaze position | -0.2 | [-1.7, +1.2] | 5 / 6 | 1.000 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -2.7 | [-5.4, +0.0] | 16 / 29 | 0.072 |

답이 바뀐 문항: gaze vs inject_only 25, gaze vs placebo 14, gaze vs lora_only 83

### fixation이 1개 이상인 문항 (N=393)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 168 | 42.7% |
| `gq_lora_only` | 157 | 39.9% |
| `gq_inject_only` | 142 | 36.1% |
| `gq_gaze` | 144 | 36.6% |
| `gq_placebo` | 145 | 36.9% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -2.8 | [-8.1, +2.3] | 48 / 59 | 0.334 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -3.8 | [-7.1, -0.5] | 15 / 30 | 0.036 |
| `gq_gaze − gq_inject_only` | gaze content | +0.5 | [-0.8, +1.8] | 4 / 2 | 0.688 |
| `gq_gaze − gq_placebo` | gaze position | -0.3 | [-1.8, +1.5] | 5 / 6 | 1.000 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -3.3 | [-6.6, +0.0] | 16 / 29 | 0.072 |

답이 바뀐 문항: gaze vs inject_only 8, gaze vs placebo 14, gaze vs lora_only 83

## 3. 구현 차이 (eager − sdpa, 같은 문항)

N=483

| 조건 | sdpa | eager | Δ %p | 95% CI | 개선 / 악화 | p | 답 일치 |
|---|---:|---:|---:|---|---:|---:|---:|
| `gq_q25_full` | 44.3% | 43.1% | -1.2 | [-5.0, +2.5] | 37 / 43 | 0.576 | 75.4% |
| `gq_lora_only` | 41.4% | 41.4% | +0.0 | [-4.1, +4.1] | 49 / 49 | 1.000 | 69.2% |
| `gq_inject_only` | 40.6% | 38.1% | -2.5 | [-6.6, +1.4] | 42 / 54 | 0.261 | 67.5% |
| `gq_gaze` | 41.0% | 38.7% | -2.3 | [-6.2, +1.7] | 42 / 53 | 0.305 | 68.3% |
| `gq_placebo` | 40.4% | 38.9% | -1.4 | [-5.2, +2.5] | 44 / 51 | 0.538 | 68.1% |

## 4. 같은 문항 부분집합에서의 구현별 비교

### sdpa, 양쪽 모두 성공한 문항 (N=483)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 214 | 44.3% |
| `gq_lora_only` | 200 | 41.4% |
| `gq_inject_only` | 196 | 40.6% |
| `gq_gaze` | 198 | 41.0% |
| `gq_placebo` | 195 | 40.4% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -2.9 | [-7.0, +1.2] | 43 / 57 | 0.193 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -0.8 | [-3.9, +2.3] | 27 / 31 | 0.694 |
| `gq_gaze − gq_inject_only` | gaze content | +0.4 | [-0.6, +1.7] | 5 / 3 | 0.727 |
| `gq_gaze − gq_placebo` | gaze position | +0.6 | [+0.0, +1.4] | 3 / 0 | 0.250 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -0.4 | [-3.5, +2.5] | 26 / 28 | 0.892 |

답이 바뀐 문항: gaze vs inject_only 15, gaze vs placebo 3, gaze vs lora_only 76

### eager, 양쪽 모두 성공한 문항 (N=483)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 208 | 43.1% |
| `gq_lora_only` | 200 | 41.4% |
| `gq_inject_only` | 184 | 38.1% |
| `gq_gaze` | 187 | 38.7% |
| `gq_placebo` | 188 | 38.9% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -1.7 | [-6.2, +3.1] | 62 / 70 | 0.543 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -3.3 | [-6.4, -0.4] | 19 / 35 | 0.040 |
| `gq_gaze − gq_inject_only` | gaze content | +0.6 | [-1.0, +2.1] | 9 / 6 | 0.607 |
| `gq_gaze − gq_placebo` | gaze position | -0.2 | [-1.7, +1.2] | 5 / 6 | 1.000 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -2.7 | [-5.4, +0.0] | 16 / 29 | 0.072 |

답이 바뀐 문항: gaze vs inject_only 25, gaze vs placebo 14, gaze vs lora_only 83

## 5. 시각 근거 문항 (`need_audio=False`)

정답 근거에 대화가 필요 없다고 라벨된 문항이다(ORACLE_REPORT의 분석 2와 같은 기준). "화면 하나로 풀린다"는 뜻은 아니다. 사후 분할이며 보정하지 않았다.

### 5.1 sdpa

### 시각 근거 전체 (N=287)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 131 | 45.6% |
| `gq_lora_only` | 120 | 41.8% |
| `gq_inject_only` | 115 | 40.1% |
| `gq_gaze` | 115 | 40.1% |
| `gq_placebo` | 113 | 39.4% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -3.8 | [-9.4, +1.4] | 26 / 37 | 0.207 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -1.7 | [-5.9, +2.4] | 15 / 20 | 0.500 |
| `gq_gaze − gq_inject_only` | gaze content | +0.0 | [-1.7, +1.7] | 3 / 3 | 1.000 |
| `gq_gaze − gq_placebo` | gaze position | +0.7 | [+0.0, +1.7] | 2 / 0 | 0.500 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -1.7 | [-5.6, +2.1] | 13 / 18 | 0.473 |

답이 바뀐 문항: gaze vs inject_only 8, gaze vs placebo 2, gaze vs lora_only 45

### 시각 근거 + fixation 1개 이상 (N=236)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 106 | 44.9% |
| `gq_lora_only` | 98 | 41.5% |
| `gq_inject_only` | 92 | 39.0% |
| `gq_gaze` | 93 | 39.4% |
| `gq_placebo` | 91 | 38.6% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -3.4 | [-8.9, +2.1] | 21 / 29 | 0.322 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -2.5 | [-7.2, +1.7] | 12 / 18 | 0.362 |
| `gq_gaze − gq_inject_only` | gaze content | +0.4 | [+0.0, +1.3] | 1 / 0 | 1.000 |
| `gq_gaze − gq_placebo` | gaze position | +0.8 | [+0.0, +2.1] | 2 / 0 | 0.500 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -2.1 | [-6.8, +2.5] | 13 / 18 | 0.473 |

답이 바뀐 문항: gaze vs inject_only 2, gaze vs placebo 2, gaze vs lora_only 45

### 5.2 eager

### 시각 근거 전체 (성공한 문항) (N=284)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 120 | 42.3% |
| `gq_lora_only` | 116 | 40.8% |
| `gq_inject_only` | 107 | 37.7% |
| `gq_gaze` | 105 | 37.0% |
| `gq_placebo` | 106 | 37.3% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -1.4 | [-7.4, +5.3] | 41 / 45 | 0.747 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -3.2 | [-7.4, +1.1] | 15 / 24 | 0.200 |
| `gq_gaze − gq_inject_only` | gaze content | -0.7 | [-2.8, +1.1] | 3 / 5 | 0.727 |
| `gq_gaze − gq_placebo` | gaze position | -0.4 | [-2.1, +1.4] | 3 / 4 | 1.000 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -3.9 | [-8.1, +0.0] | 11 / 22 | 0.080 |

답이 바뀐 문항: gaze vs inject_only 12, gaze vs placebo 10, gaze vs lora_only 54

### 시각 근거 + fixation 1개 이상 (N=233)

| 조건 | 정답 | 정확도 |
|---|---:|---:|
| `gq_q25_full` | 97 | 41.6% |
| `gq_lora_only` | 91 | 39.1% |
| `gq_inject_only` | 79 | 33.9% |
| `gq_gaze` | 80 | 34.3% |
| `gq_placebo` | 81 | 34.8% |

| 비교 | 의미 | Δ %p | 95% CI | 개선 / 악화 | p |
|---|---|---:|---|---:|---:|
| `gq_lora_only − gq_q25_full` | LoRA | -2.6 | [-9.4, +4.3] | 32 / 38 | 0.550 |
| `gq_inject_only − gq_lora_only` | V-JEPA injection | -5.2 | [-9.9, -0.4] | 11 / 23 | 0.058 |
| `gq_gaze − gq_inject_only` | gaze content | +0.4 | [-0.9, +2.1] | 2 / 1 | 1.000 |
| `gq_gaze − gq_placebo` | gaze position | -0.4 | [-2.6, +1.7] | 3 / 4 | 1.000 |
| `gq_gaze − gq_lora_only` | GazeQwen vs LoRA only | -4.7 | [-9.9, +0.0] | 11 / 22 | 0.080 |

답이 바뀐 문항: gaze vs inject_only 4, gaze vs placebo 10, gaze vs lora_only 54

### 5.3 구현 차이 (시각 근거, eager − sdpa)

N=284

| 조건 | sdpa | eager | Δ %p | 95% CI | 개선 / 악화 | p | 답 일치 |
|---|---:|---:|---:|---|---:|---:|---:|
| `gq_q25_full` | 45.4% | 42.3% | -3.2 | [-8.1, +1.4] | 19 / 28 | 0.243 | 72.9% |
| `gq_lora_only` | 42.3% | 40.8% | -1.4 | [-6.7, +4.2] | 30 / 34 | 0.708 | 66.9% |
| `gq_inject_only` | 40.1% | 37.7% | -2.5 | [-7.7, +2.8] | 27 / 34 | 0.443 | 67.3% |
| `gq_gaze` | 40.1% | 37.0% | -3.2 | [-8.5, +2.1] | 25 / 34 | 0.298 | 68.0% |
| `gq_placebo` | 39.4% | 37.3% | -2.1 | [-7.4, +3.2] | 27 / 33 | 0.519 | 67.3% |
