# Full / gaze / query routing / oracle results

QA common denominator: 500/500; excluded: 0.
Hit@3 denominators: tol=0 **487**, tol=60s **500**. Hit requires at least one target clip in the common candidate pool.

Oracle is computed separately for QA and Hit@3 as A OR B; it is an offline upper bound.

### correct

| Condition | Successes | N | Rate |
|---|---:|---:|---:|
| full | 220 | 500 | 44.0% |
| gaze | 225 | 500 | 45.0% |
| routing | 219 | 500 | 43.8% |
| oracle | 247 | 500 | 49.4% |

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| gaze − full | +1.0 | 27 | 22 | [-1.8, +3.6] |
| routing − full | -0.2 | 3 | 4 | [-1.2, +0.8] |
| routing − gaze | -1.2 | 18 | 24 | [-3.8, +1.4] |
| oracle − routing | +5.6 | 28 | 0 | [+3.6, +7.6] |

### hit0

| Condition | Successes | N | Rate |
|---|---:|---:|---:|
| full | 21 | 487 | 4.3% |
| gaze | 35 | 487 | 7.2% |
| routing | 27 | 487 | 5.5% |
| oracle | 44 | 487 | 9.0% |

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| gaze − full | +2.9 | 23 | 9 | [+0.6, +5.1] |
| routing − full | +1.2 | 6 | 0 | [+0.4, +2.3] |
| routing − gaze | -1.6 | 9 | 17 | [-3.7, +0.4] |
| oracle − routing | +3.5 | 17 | 0 | [+2.1, +5.3] |

### hit60

| Condition | Successes | N | Rate |
|---|---:|---:|---:|
| full | 54 | 500 | 10.8% |
| gaze | 79 | 500 | 15.8% |
| routing | 65 | 500 | 13.0% |
| oracle | 97 | 500 | 19.4% |

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| gaze − full | +5.0 | 43 | 18 | [+2.2, +8.0] |
| routing − full | +2.2 | 12 | 1 | [+0.8, +3.6] |
| routing − gaze | -2.8 | 17 | 31 | [-5.6, -0.2] |
| oracle − routing | +6.4 | 32 | 0 | [+4.4, +8.6] |

### correct (dual-view conditions)

D `dual_gaze`: gaze Top-3 clips shown as full frame + gaze crop. E `dual_retrieve`: full Top-3 as full frames + gaze Top-3 as gaze crops. Every condition gets the same text memory and at most 64 images.

| Condition | Successes | N | Rate |
|---|---:|---:|---:|
| full | 220 | 500 | 44.0% |
| gaze | 225 | 500 | 45.0% |
| routing | 219 | 500 | 43.8% |
| oracle | 247 | 500 | 49.4% |
| dual_gaze | 225 | 500 | 45.0% |
| dual_retrieve | 218 | 500 | 43.6% |

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| dual_gaze − gaze | +0.0 | 20 | 20 | [-2.4, +2.6] |
| dual_gaze − full | +1.0 | 27 | 22 | [-1.8, +3.8] |
| dual_retrieve − full | -0.4 | 18 | 20 | [-2.8, +2.0] |
| dual_retrieve − gaze | -1.4 | 16 | 23 | [-3.8, +1.0] |
| dual_retrieve − dual_gaze | -1.4 | 14 | 21 | [-3.6, +0.8] |
| dual_retrieve − routing | -0.2 | 18 | 19 | [-2.6, +2.2] |

### correct (gaze prompting on the full Top-3)

EgoGazeVQA-Plus prompting. `gp_text`: normalized gaze coordinates per frame. `gp_mark`: filled red dot (r=20 px) on the gaze point. `gp_saliency`: per-clip grayscale gaze-trajectory map (later gaze brighter) before the clip's frames. Same retrieval, frames and text memory as `full`; `gaze` (gaze index, gaze-crop frames) is shown for reference. `gaze_ret_full`: gaze-index Top-3 shown as full frames, no prompting; `*_gr`: the same three prompts on that Top-3, so retrieval (full / gaze index) x prompting is a 2 x 3 design.

| Condition | Successes | N | Rate |
|---|---:|---:|---:|
| full | 220 | 500 | 44.0% |
| gaze | 225 | 500 | 45.0% |
| gp_text | 218 | 500 | 43.6% |
| gp_mark | 213 | 500 | 42.6% |
| gp_saliency | 215 | 500 | 43.0% |
| gaze_ret_full | 217 | 500 | 43.4% |
| gp_text_gr | 218 | 500 | 43.6% |
| gp_mark_gr | 221 | 500 | 44.2% |
| gp_saliency_gr | 218 | 500 | 43.6% |

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| gp_text − full | -0.4 | 18 | 20 | [-2.8, +2.0] |
| gp_mark − full | -1.4 | 17 | 24 | [-4.0, +1.0] |
| gp_saliency − full | -1.0 | 13 | 18 | [-3.2, +1.0] |
| gp_mark − gp_text | -1.0 | 16 | 21 | [-3.4, +1.4] |
| gp_saliency − gp_mark | +0.4 | 16 | 14 | [-1.6, +2.6] |
| gaze_ret_full − full | -0.6 | 23 | 26 | [-3.4, +2.0] |
| gaze_ret_full − gaze | -1.6 | 21 | 29 | [-4.4, +1.2] |
| gp_text_gr − gaze_ret_full | +0.2 | 19 | 18 | [-2.2, +2.6] |
| gp_mark_gr − gaze_ret_full | +0.8 | 22 | 18 | [-1.6, +3.2] |
| gp_saliency_gr − gaze_ret_full | +0.2 | 15 | 14 | [-1.8, +2.4] |
| gp_mark_gr − gp_text_gr | +0.6 | 16 | 13 | [-1.4, +2.6] |
| gp_saliency_gr − gp_mark_gr | -0.6 | 13 | 16 | [-2.6, +1.4] |
| gp_text_gr − gp_text | +0.0 | 27 | 27 | [-2.8, +2.8] |
| gp_mark_gr − gp_mark | +1.6 | 30 | 22 | [-1.2, +4.4] |
| gp_saliency_gr − gp_saliency | +0.6 | 28 | 25 | [-2.2, +3.4] |

## Routing and complementarity

Qwen decisions: FULL 409, GAZE 91.
Router agreed across both set orders on 324/500 questions (disagreements fall back to FULL).

- correct: A/B disagree on 49/500; router picks the successful arm on 21/49 discordant questions.
- hit0: A/B disagree on 32/487; router picks the successful arm on 15/32 discordant questions.
- hit60: A/B disagree on 61/500; router picks the successful arm on 29/61 discordant questions.
