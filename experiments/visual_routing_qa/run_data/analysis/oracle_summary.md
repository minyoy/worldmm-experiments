# Oracle-frame gaze prompting

Questions with all four conditions finished: 487/487; excluded: 0.

Frames: the question's target clip(s) at tol=0, 1 fps, at most 64 frames (proportional per clip). Same text memory, QA model and greedy decoding in every condition; only the way the gaze is conveyed differs. Both analyses below (all questions; need_audio=False only) were fixed before the results were seen, and the primary one is the first.

## Analysis 1 (primary): all questions with a target clip  (N=487)

| Condition | Correct | Rate |
|---|---:|---:|
| oracle_full | 235 | 48.3% |
| oracle_text | 238 | 48.9% |
| oracle_mark | 236 | 48.5% |
| oracle_saliency | 244 | 50.1% |
| full | 215 | 44.1% |
| gaze | 218 | 44.8% |

**Primary (gaze prompting vs no gaze information, same oracle frames)**

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| oracle_text − oracle_full | +0.6 | 22 | 19 | [-1.8, +3.3] |
| oracle_mark − oracle_full | +0.2 | 21 | 20 | [-2.3, +2.7] |
| oracle_saliency − oracle_full | +1.8 | 27 | 18 | [-0.8, +4.5] |

**Between prompting methods**

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| oracle_mark − oracle_text | -0.4 | 14 | 16 | [-2.7, +1.8] |
| oracle_saliency − oracle_text | +1.2 | 19 | 13 | [-1.0, +3.5] |
| oracle_saliency − oracle_mark | +1.6 | 18 | 10 | [-0.4, +3.9] |

**Oracle frames vs retrieved frames (run.py `full` / `gaze`)**

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| oracle_full − full | +4.1 | 43 | 23 | [+0.8, +7.4] |
| oracle_full − gaze | +3.5 | 41 | 24 | [+0.2, +6.8] |

## Analysis 2: need_audio=False (visual-evidence) questions  (N=287)

| Condition | Correct | Rate |
|---|---:|---:|
| oracle_full | 139 | 48.4% |
| oracle_text | 145 | 50.5% |
| oracle_mark | 140 | 48.8% |
| oracle_saliency | 145 | 50.5% |
| full | 124 | 43.2% |
| gaze | 126 | 43.9% |

**Primary (gaze prompting vs no gaze information, same oracle frames)**

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| oracle_text − oracle_full | +2.1 | 13 | 7 | [-1.0, +5.2] |
| oracle_mark − oracle_full | +0.3 | 12 | 11 | [-2.8, +3.5] |
| oracle_saliency − oracle_full | +2.1 | 15 | 9 | [-1.4, +5.6] |

**Between prompting methods**

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| oracle_mark − oracle_text | -1.7 | 5 | 10 | [-4.5, +0.7] |
| oracle_saliency − oracle_text | +0.0 | 7 | 7 | [-2.4, +2.8] |
| oracle_saliency − oracle_mark | +1.7 | 10 | 5 | [-1.0, +4.5] |

**Oracle frames vs retrieved frames (run.py `full` / `gaze`)**

| Pair | Δ percentage points | A only | B only | Paired bootstrap 95% CI |
|---|---:|---:|---:|---:|
| oracle_full − full | +5.2 | 26 | 11 | [+1.0, +9.4] |
| oracle_full − gaze | +4.5 | 24 | 11 | [+0.3, +8.4] |

## Pooled effect of the three prompting methods (post-hoc)

Per-question mean of (oracle_text, oracle_mark, oracle_saliency) minus oracle_full. The three methods are not independent confirmations: they share the questions and the baseline.

| Analysis | N | Mean gain (pp) | 95% CI | sign-flip p |
|---|---:|---:|---|---:|
| all questions with a target clip | 487 | +0.89 | [-1.37, +3.15] | 0.479 |
| need_audio=False | 287 | +1.51 | [-1.39, +4.41] | 0.346 |

Correlation of per-question changes: text-mark 0.63, text-saliency 0.63, mark-saliency 0.68.
Questions improved by at least one method: 37; worsened by at least one: 31 of 487.

## By question type (descriptive; made after the results were seen, not tested)

Per-type samples are small (61-122); intervals are wide and are not corrected for the number of type x method comparisons.

### Composition

| Type | N | need_audio=False | need_audio=True | >1 target clips |
|---|---:|---:|---:|---:|
| EntityLog | 122 | 103 | 19 | 2 |
| EventRecall | 122 | 62 | 60 | 4 |
| RelationMap | 121 | 90 | 31 | 7 |
| HabitInsight | 61 | 28 | 33 | 24 |
| TaskMaster | 61 | 4 | 57 | 1 |

### Accuracy

| Type | N | oracle_full | oracle_text | oracle_mark | oracle_saliency |
|---|---:|---:|---:|---:|---:|
| EntityLog | 122 | 48.4% | 50.8% | 48.4% | 50.0% |
| EventRecall | 122 | 46.7% | 49.2% | 50.8% | 51.6% |
| RelationMap | 121 | 48.8% | 52.1% | 51.2% | 52.9% |
| HabitInsight | 61 | 52.5% | 44.3% | 44.3% | 44.3% |
| TaskMaster | 61 | 45.9% | 42.6% | 42.6% | 47.5% |

### Difference from oracle_full (percentage points, [95% CI], gain/loss questions)

| Type | N | oracle_text | oracle_mark | oracle_saliency |
|---|---:|---|---|---|
| EntityLog | 122 | +2.5 [-2.5, +7.4] (6/3) | +0.0 [-5.7, +5.7] (6/6) | +1.6 [-4.1, +7.4] (7/5) |
| EventRecall | 122 | +2.5 [-2.5, +8.2] (7/4) | +4.1 [-0.8, +9.0] (7/2) | +4.9 [-0.8, +10.7] (9/3) |
| RelationMap | 121 | +3.3 [-0.8, +8.3] (6/2) | +2.5 [-0.8, +6.6] (4/1) | +4.1 [-0.8, +9.1] (7/2) |
| HabitInsight | 61 | -8.2 [-16.4, +0.0] (1/6) | -8.2 [-16.4, +0.0] (1/6) | -8.2 [-14.8, -1.6] (0/5) |
| TaskMaster | 61 | -3.3 [-11.5, +4.9] (2/4) | -3.3 [-11.5, +4.9] (3/5) | +1.6 [-6.6, +9.8] (4/3) |

### Type x evidence kind (accuracy)

| Type | need_audio | N | oracle_full | oracle_text | oracle_mark | oracle_saliency |
|---|---|---:|---:|---:|---:|---:|
| EntityLog | False | 103 | 49.5% | 50.5% | 48.5% | 50.5% |
| EntityLog | True | 19 | 42.1% | 52.6% | 47.4% | 47.4% |
| EventRecall | False | 62 | 48.4% | 50.0% | 48.4% | 51.6% |
| EventRecall | True | 60 | 45.0% | 48.3% | 53.3% | 51.7% |
| RelationMap | False | 90 | 44.4% | 47.8% | 47.8% | 48.9% |
| RelationMap | True | 31 | 61.3% | 64.5% | 61.3% | 64.5% |
| HabitInsight | False | 28 | 60.7% | 60.7% | 53.6% | 57.1% |
| HabitInsight | True | 33 | 45.5% | 30.3% | 36.4% | 33.3% |
| TaskMaster | False | 4 | 25.0% | 50.0% | 50.0% | 25.0% |
| TaskMaster | True | 57 | 47.4% | 42.1% | 42.1% | 49.1% |

### By number of target clips (accuracy, mean difference from oracle_full in percentage points)

| Group | N | oracle_full | oracle_text | oracle_mark | oracle_saliency | Δ text / mark / saliency |
|---|---:|---:|---:|---:|---:|---|
| 1 target clip | 449 | 48.3% | 49.4% | 49.0% | 50.6% | +1.1 / +0.7 / +2.2 |
| >1 target clips | 38 | 47.4% | 42.1% | 42.1% | 44.7% | -5.3 / -5.3 / -2.6 |
| HabitInsight, 1 clip | 37 | 54.1% | 43.2% | 45.9% | 43.2% | -10.8 / -8.1 / -10.8 |
| HabitInsight, >1 clips | 24 | 50.0% | 45.8% | 41.7% | 45.8% | -4.2 / -8.3 / -4.2 |

