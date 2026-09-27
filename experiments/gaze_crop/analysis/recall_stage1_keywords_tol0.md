# Visual retrieval recall by crop arm (`--query-source keywords`)

Pool: 623 clips (123 targets + 500 distractors, scope `same-day`), scored on 120 of 120 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `center_05` | 8.3% | 19.2% | 22.5% | 30.8% | 36.7% | 53.3% | 35 |
| `full` | 10.8% | 12.5% | 15.0% | 26.7% | 36.7% | 50.8% | 48 |
| `gaze_05` | 10.8% | 17.5% | 20.0% | 27.5% | 39.2% | 57.5% | 31 |
| `gazef_05` | 8.3% | 15.0% | 20.0% | 30.0% | 38.3% | 53.3% | 41 |
| `gazef_05_rand` | 7.5% | 11.7% | 17.5% | 28.3% | 34.2% | 49.2% | 53 |
| `pkl` | 10.0% | 16.7% | 18.3% | 28.3% | 36.7% | 51.7% | 47 |

Paired differences at k=3:

| arms | diff (pp) | 95% CI | a only | b only |
|---|---|---|---|---|
| `center_05` - `full` | +6.7 | [+1.7, +12.5] | 10 | 2 |
| `center_05` - `gaze_05` | +1.7 | [-3.3, +6.7] | 6 | 4 |
| `center_05` - `gazef_05` | +4.2 | [-0.8, +9.2] | 8 | 3 |
| `center_05` - `gazef_05_rand` | +7.5 | [+2.5, +13.3] | 11 | 2 |
| `center_05` - `pkl` | +2.5 | [-4.2, +8.3] | 9 | 6 |
| `full` - `gaze_05` | -5.0 | [-10.0, +0.0] | 2 | 8 |
| `full` - `gazef_05` | -2.5 | [-6.7, +1.7] | 2 | 5 |
| `full` - `gazef_05_rand` | +0.8 | [-4.2, +6.7] | 6 | 5 |
| `full` - `pkl` | -4.2 | [-8.3, -0.8] | 0 | 5 |
| `gaze_05` - `gazef_05` | +2.5 | [-2.5, +7.5] | 6 | 3 |
| `gaze_05` - `gazef_05_rand` | +5.8 | [+0.8, +11.7] | 9 | 2 |
| `gaze_05` - `pkl` | +0.8 | [-4.2, +5.8] | 6 | 5 |
| `gazef_05` - `gazef_05_rand` | +3.3 | [-2.5, +9.2] | 9 | 5 |
| `gazef_05` - `pkl` | -1.7 | [-6.7, +3.3] | 4 | 6 |
| `gazef_05_rand` - `pkl` | -5.0 | [-11.7, +0.8] | 5 | 11 |

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
