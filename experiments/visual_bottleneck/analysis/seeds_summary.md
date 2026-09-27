# Visual bottleneck: repeated sampled runs

Questions: 120. Generation is sampled (Qwen3-VL default: do_sample, temperature 0.7); run0 is the original run, seed1/seed2 are repeats from `run_seeds.sh`. B_replay's run0 is `results/B.json` (identical final-answer prompt).

## Accuracy (%)

| condition | label | runs | per run | mean | sd | answers flip across runs |
|---|---|---|---|---|---|---|
| A | Question only | 3 | 29.2 / 28.3 / 30.8 | **29.4** | 1.3 | 9/120 |
| B_replay | Text memory (B context, answered once) | 3 | 40.8 / 43.3 / 45.0 | **43.1** | 2.1 | 10/120 |
| C | Oracle visual | 3 | 36.7 / 36.7 / 39.2 | **37.5** | 1.4 | 9/120 |
| D | Text + Oracle visual | 3 | 50.0 / 50.8 / 50.0 | **50.3** | 0.5 | 4/120 |
| E_prime | B context + Retrieved visual | 3 | 45.8 / 44.2 / 44.2 | **44.7** | 1.0 | 9/120 |

## Paired comparisons (mean over runs)

delta = mean acc(X) - mean acc(Y). better/worse = questions whose run-averaged score is higher/lower. p = two-sided sign-flip permutation test on per-question score differences (50,000 permutations). Majority = correct in >=2 of 3 runs, exact McNemar.

| X - Y | meaning | acc X | acc Y | **delta** | per-run deltas | better / worse | p | majority b / c | majority p |
|---|---|---|---|---|---|---|---|---|---|
| C - A | visual evidence alone helps? | 37.5 | 29.4 | **+8.1** | +7.5 / +8.3 / +8.3 | 24 / 17 | 0.074 | 20 / 9 | 0.061 |
| ↳ need_audio=False | | 46.2 | 34.3 | +11.9 | | 16 / 11 | 0.046 | | |
| ↳ need_audio=True | | 25.3 | 22.7 | +2.7 | | 8 / 6 | 0.760 | | |
| D - B_replay | oracle visual gain on top of text | 50.3 | 43.1 | **+7.2** | +9.2 / +7.5 / +5.0 | 23 / 11 | 0.083 | 16 / 8 | 0.152 |
| ↳ need_audio=False | | 53.3 | 44.8 | +8.6 | | 13 / 5 | 0.123 | | |
| ↳ need_audio=True | | 46.0 | 40.7 | +5.3 | | 10 / 6 | 0.459 | | |
| C - B_replay | which modality is more usable (diagnostic) | 37.5 | 43.1 | **-5.6** | -4.2 / -6.7 / -5.8 | 26 / 32 | 0.345 | 23 / 28 | 0.576 |
| ↳ need_audio=False | | 46.2 | 44.8 | +1.4 | | 17 / 15 | 0.892 | | |
| ↳ need_audio=True | | 25.3 | 40.7 | -15.3 | | 9 / 17 | 0.113 | | |
| E_prime - B_replay | retrieved visual gain, text fixed | 44.7 | 43.1 | **+1.7** | +5.0 / +0.8 / -0.8 | 15 / 10 | 0.620 | 8 / 5 | 0.581 |
| ↳ need_audio=False | | 44.3 | 44.8 | -0.5 | | 7 / 5 | 1.000 | | |
| ↳ need_audio=True | | 45.3 | 40.7 | +4.7 | | 8 / 5 | 0.455 | | |
| D - E_prime | oracle - retrieved, text fixed | 50.3 | 44.7 | **+5.6** | +4.2 / +6.7 / +5.8 | 16 / 7 | 0.089 | 11 / 6 | 0.332 |
| ↳ need_audio=False | | 53.3 | 44.3 | +9.0 | | 11 / 4 | 0.056 | | |
| ↳ need_audio=True | | 46.0 | 45.3 | +0.7 | | 5 / 3 | 1.000 | | |

### Mean accuracy by need_audio

| subset | n | A | B_replay | C | D | E_prime |
|---|---|---|---|---|---|---|
| False | 70 | 34.3 | 44.8 | 46.2 | 53.3 | 44.3 |
| True | 50 | 22.7 | 40.7 | 25.3 | 46.0 | 45.3 |

### Mean accuracy by type

| subset | n | A | B_replay | C | D | E_prime |
|---|---|---|---|---|---|---|
| EntityLog | 24 | 27.8 | 27.8 | 43.1 | 50.0 | 33.3 |
| RelationMap | 24 | 27.8 | 48.6 | 36.1 | 58.3 | 55.6 |
| EventRecall | 24 | 18.1 | 30.6 | 33.3 | 27.8 | 27.8 |
| TaskMaster | 24 | 27.8 | 54.2 | 20.8 | 51.4 | 52.8 |
| HabitInsight | 24 | 45.8 | 54.2 | 54.2 | 63.9 | 54.2 |

### Mean accuracy by gap

| subset | n | A | B_replay | C | D | E_prime |
|---|---|---|---|---|---|---|
| 0 | 57 | 29.8 | 52.0 | 35.7 | 52.6 | 48.0 |
| 1 | 33 | 29.3 | 38.4 | 44.4 | 55.6 | 47.5 |
| 2+ | 30 | 28.9 | 31.1 | 33.3 | 40.0 | 35.6 |
