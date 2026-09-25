# Visual bottleneck — results summary

## Accuracy (%)

| condition | label | n | acc |
|---|---|---|---|
| A | Question only | 120 | 29.2 |
| B | Text memory | 120 | 40.8 |
| C | Oracle visual | 120 | 36.7 |
| D | Text + Oracle visual | 120 | 50.0 |
| E | Text + Retrieved visual (WorldMM) | 120 | 41.7 |
| E_prime | B ctx + Retrieved visual | 120 | 45.8 |

### by need_audio

| subset | n | A | B | C | D | E | E_prime |
|---|---|---|---|---|---|---|---|
| False | 70 | 32.9 | 42.9 | 47.1 | 54.3 | 45.7 | 44.3 |
| True | 50 | 24.0 | 38.0 | 22.0 | 44.0 | 36.0 | 48.0 |

### by type

| subset | n | A | B | C | D | E | E_prime |
|---|---|---|---|---|---|---|---|
| EntityLog | 24 | 25.0 | 25.0 | 45.8 | 50.0 | 41.7 | 33.3 |
| RelationMap | 24 | 25.0 | 45.8 | 29.2 | 58.3 | 50.0 | 58.3 |
| EventRecall | 24 | 20.8 | 29.2 | 33.3 | 29.2 | 20.8 | 29.2 |
| TaskMaster | 24 | 29.2 | 50.0 | 20.8 | 50.0 | 50.0 | 54.2 |
| HabitInsight | 24 | 45.8 | 54.2 | 54.2 | 62.5 | 45.8 | 54.2 |

### by gap (query day - target day)

| subset | n | A | B | C | D | E | E_prime |
|---|---|---|---|---|---|---|---|
| 0 | 57 | 29.8 | 50.9 | 36.8 | 52.6 | 50.9 | 49.1 |
| 1 | 33 | 27.3 | 33.3 | 42.4 | 54.5 | 42.4 | 48.5 |
| 2+ | 30 | 30.0 | 30.0 | 30.0 | 40.0 | 23.3 | 36.7 |

## Paired comparisons

delta = acc(X) - acc(Y) on questions present in both. b = X right & Y wrong, c = X wrong & Y right. p = exact McNemar (two-sided).

| exp | X - Y | n | acc X | acc Y | delta | b | c | p | meaning |
|---|---|---|---|---|---|---|---|---|---|
| Exp.1 | C - A | 120 | 36.7 | 29.2 | +7.5 | 19 | 10 | 0.136 | visual evidence alone helps? |
|  | ↳ need_audio=False | 70 | 47.1 | 32.9 | +14.2 | | | | |
|  | ↳ need_audio=True | 50 | 22.0 | 24.0 | -2.0 | | | | |
| Exp.1 | D - B | 120 | 50.0 | 40.8 | +9.2 | 19 | 8 | 0.052 | oracle visual gain on top of text |
|  | ↳ need_audio=False | 70 | 54.3 | 42.9 | +11.4 | | | | |
|  | ↳ need_audio=True | 50 | 44.0 | 38.0 | +6.0 | | | | |
| Exp.1 | C - B | 120 | 36.7 | 40.8 | -4.1 | 22 | 27 | 0.568 | which modality is more usable (diagnostic) |
| Exp.2 | E - B | 120 | 41.7 | 40.8 | +0.9 | 10 | 9 | 1.000 | retrieved visual gain on top of text (paper E+S+V - E+S) |
|  | ↳ need_audio=False | 70 | 45.7 | 42.9 | +2.8 | | | | |
|  | ↳ need_audio=True | 50 | 36.0 | 38.0 | -2.0 | | | | |
| Exp.2 | D - E | 120 | 50.0 | 41.7 | +8.3 | 20 | 10 | 0.099 | retrieval bottleneck size (oracle - retrieved) |
|  | ↳ need_audio=False | 70 | 54.3 | 45.7 | +8.6 | | | | |
|  | ↳ need_audio=True | 50 | 44.0 | 36.0 | +8.0 | | | | |
| Exp.2 | D - E_prime | 120 | 50.0 | 45.8 | +4.2 | 11 | 6 | 0.332 | oracle - retrieved with identical text context |
| Exp.2 | E_prime - B | 120 | 45.8 | 40.8 | +5.0 | 11 | 5 | 0.210 | retrieved visual gain, text fixed |

## Retrieval diagnostics (E, E')

- **E**: visual search used in 2/120 questions (query kinds {'text': 2}); recall@k vs target clips 0/2 = 0.0%; acc when visual used 0.0 vs not used 42.4; mean rounds 3.70
- **E_prime**: visual search used in 120/120 questions (query kinds {'text': 120}); recall@k vs target clips 6/120 = 5.0%; acc when visual used 45.8 vs not used -; mean rounds 3.42

## Reading (PLAN.md section 1)

- oracle visual gain D-B = +9.2, retrieved visual gain E-B = +0.9, retrieval loss D-E = +8.3
- visual-only gain C-A = +7.5; if this is large while D-B is small, consider text dominance / modality fusion
- large / large: both fine · large / small: retrieval bottleneck · small / small: utilization bottleneck
