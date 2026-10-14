# Oracle-frame crop QA

Questions with all 7 conditions finished: 487/487; excluded: 0.

Target clips (tol=0), 1 fps, <= 32 timestamps; crop ratio 0.5; off-center = gaze distance from (0.5, 0.5) >= 0.212. Same timestamps, text memory, QA model and greedy decoding in every condition. Analyses 1-3 were fixed before the results.

## Analysis 1 (primary): all questions  (N=487)

| Condition | Correct | Rate | Mean images |
|---|---:|---:|---:|
| full | 234 | 48.0% | 30.1 |
| gaze | 242 | 49.7% | 30.1 |
| center | 249 | 51.1% | 30.1 |
| dual_center | 246 | 50.5% | 60.2 |
| dual_gaze | 245 | 50.3% | 60.2 |
| dual_offcenter | 237 | 48.7% | 32.6 |
| dual_offcenter_center | 236 | 48.5% | 32.6 |

**Primary (Holm over these three)**

| Pair | Δ pp | 95% CI | A only / B only | exact p | Holm p |
|---|---:|---|---:|---:|---:|
| gaze − center | -1.4 | [-3.7, +0.8] | 12 / 19 | 0.281 | 0.843 |
| dual_gaze − dual_center | -0.2 | [-2.3, +1.8] | 13 / 14 | 1.000 | 1.000 |
| dual_offcenter − full | +0.6 | [-0.8, +2.1] | 8 / 5 | 0.581 | 1.000 |

**Secondary (descriptive, not corrected)**

| Pair | Δ pp | 95% CI | A only / B only | exact p |
|---|---:|---|---:|---:|
| gaze − full | +1.6 | [-1.0, +4.3] | 26 / 18 | 0.291 |
| center − full | +3.1 | [+0.4, +5.7] | 29 / 14 | 0.032 |
| dual_center − full | +2.5 | [+0.2, +4.7] | 22 / 10 | 0.050 |
| dual_gaze − full | +2.3 | [-0.4, +4.9] | 28 / 17 | 0.135 |
| dual_offcenter_center − full | +0.4 | [-1.2, +2.1] | 9 / 7 | 0.804 |
| dual_gaze − gaze | +0.6 | [-2.3, +3.3] | 26 / 23 | 0.775 |
| dual_center − center | -0.6 | [-3.1, +1.8] | 17 / 20 | 0.743 |
| dual_offcenter − dual_gaze | -1.6 | [-4.3, +1.0] | 17 / 25 | 0.280 |
| dual_offcenter − dual_offcenter_center | +0.2 | [-1.0, +1.4] | 5 / 4 | 1.000 |

## Analysis 2: need_audio=False  (N=287)

| Condition | Correct | Rate | Mean images |
|---|---:|---:|---:|
| full | 138 | 48.1% | 30.1 |
| gaze | 150 | 52.3% | 30.1 |
| center | 153 | 53.3% | 30.1 |
| dual_center | 145 | 50.5% | 60.2 |
| dual_gaze | 145 | 50.5% | 60.2 |
| dual_offcenter | 140 | 48.8% | 32.7 |
| dual_offcenter_center | 137 | 47.7% | 32.7 |

**Secondary (descriptive, not corrected)**

| Pair | Δ pp | 95% CI | A only / B only | exact p |
|---|---:|---|---:|---:|
| gaze − full | +4.2 | [+0.3, +8.0] | 21 / 9 | 0.043 |
| center − full | +5.2 | [+1.7, +8.7] | 21 / 6 | 0.006 |
| dual_center − full | +2.4 | [-0.7, +5.6] | 14 / 7 | 0.189 |
| dual_gaze − full | +2.4 | [-1.4, +5.9] | 18 / 11 | 0.265 |
| dual_offcenter_center − full | -0.3 | [-2.4, +1.7] | 4 / 5 | 1.000 |
| dual_gaze − gaze | -1.7 | [-5.6, +2.1] | 14 / 19 | 0.487 |
| dual_center − center | -2.8 | [-6.3, +0.3] | 8 / 16 | 0.152 |
| dual_offcenter − dual_gaze | -1.7 | [-5.2, +1.7] | 11 / 16 | 0.442 |
| dual_offcenter − dual_offcenter_center | +1.0 | [-0.7, +2.8] | 5 / 2 | 0.453 |

## Analysis 3: >=1 off-center timestamp  (N=370)

| Condition | Correct | Rate | Mean images |
|---|---:|---:|---:|
| full | 187 | 50.5% | 30.1 |
| gaze | 193 | 52.2% | 30.1 |
| center | 200 | 54.1% | 30.1 |
| dual_center | 191 | 51.6% | 60.2 |
| dual_gaze | 190 | 51.4% | 60.2 |
| dual_offcenter | 190 | 51.4% | 33.4 |
| dual_offcenter_center | 189 | 51.1% | 33.4 |

| Pair | Δ pp | 95% CI | A only / B only | exact p |
|---|---:|---|---:|---:|
| dual_offcenter − full | +0.8 | [-1.1, +2.7] | 8 / 5 | 0.581 |
| dual_offcenter_center − full | +0.5 | [-1.6, +2.7] | 9 / 7 | 0.804 |
| dual_offcenter − dual_offcenter_center | +0.3 | [-1.4, +1.9] | 5 / 4 | 1.000 |
| dual_offcenter − dual_gaze | +0.0 | [-3.0, +3.0] | 15 / 15 | 1.000 |

## >=5 off-center timestamps (descriptive)  (N=76)

| Condition | Correct | Rate | Mean images |
|---|---:|---:|---:|
| full | 33 | 43.4% | 30.1 |
| gaze | 37 | 48.7% | 30.1 |
| center | 37 | 48.7% | 30.1 |
| dual_center | 39 | 51.3% | 60.2 |
| dual_gaze | 37 | 48.7% | 60.2 |
| dual_offcenter | 36 | 47.4% | 37.6 |
| dual_offcenter_center | 37 | 48.7% | 37.6 |

| Pair | Δ pp | 95% CI | A only / B only | exact p |
|---|---:|---|---:|---:|
| dual_offcenter − full | +3.9 | [+0.0, +9.2] | 3 / 0 | 0.250 |
| dual_offcenter_center − full | +5.3 | [+1.3, +10.5] | 4 / 0 | 0.125 |
| dual_offcenter − dual_offcenter_center | -1.3 | [-6.6, +3.9] | 2 / 3 | 1.000 |
| dual_offcenter − dual_gaze | -1.3 | [-7.9, +5.3] | 3 / 4 | 1.000 |

## By question type (descriptive, post-hoc, not tested)

| Type | N | full | gaze | center | dual_center | dual_gaze | dual_offcenter | dual_offcenter_center |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EntityLog | 122 | 48.4% | 52.5% | 53.3% | 52.5% | 53.3% | 50.0% | 48.4% |
| EventRecall | 122 | 46.7% | 49.2% | 52.5% | 52.5% | 52.5% | 48.4% | 49.2% |
| RelationMap | 121 | 48.8% | 50.4% | 52.9% | 47.1% | 47.9% | 48.8% | 49.6% |
| HabitInsight | 61 | 50.8% | 49.2% | 49.2% | 52.5% | 49.2% | 50.8% | 49.2% |
| TaskMaster | 61 | 45.9% | 44.3% | 42.6% | 47.5% | 45.9% | 44.3% | 44.3% |

## Reproducibility: new `full` vs previous `oracle_full` (single target clip)

Questions: 449. Same frame ids: 449. Same response: 449. Same correctness: 449.
Different responses on identical frames measure run-to-run noise (image order, batching, kernels).

