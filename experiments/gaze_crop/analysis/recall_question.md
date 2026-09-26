# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 623 clips (123 targets + 500 distractors, scope `same-day`), scored on 120 of 120 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `center_05` | 0.8% | 3.3% | 5.0% | 7.5% | 12.5% | 22.5% | 118 |
| `full` | 0.8% | 2.5% | 2.5% | 6.7% | 15.8% | 27.5% | 114 |
| `gaze_05` | 0.8% | 2.5% | 3.3% | 6.7% | 12.5% | 26.7% | 98 |
| `gazef_05` | 1.7% | 3.3% | 4.2% | 7.5% | 12.5% | 28.3% | 109 |
| `gazef_05_rand` | 0.8% | 2.5% | 2.5% | 5.8% | 7.5% | 24.2% | 120 |
| `pkl` | 1.7% | 1.7% | 4.2% | 8.3% | 14.2% | 30.8% | 111 |

Paired differences at k=3:

| arms | diff (pp) | 95% CI | a only | b only |
|---|---|---|---|---|
| `center_05` - `full` | +0.8 | [+0.0, +2.5] | 1 | 0 |
| `center_05` - `gaze_05` | +0.8 | [-1.7, +4.2] | 2 | 1 |
| `center_05` - `gazef_05` | +0.0 | [-2.5, +2.5] | 1 | 1 |
| `center_05` - `gazef_05_rand` | +0.8 | [-1.7, +3.3] | 2 | 1 |
| `center_05` - `pkl` | +1.7 | [+0.0, +4.2] | 2 | 0 |
| `full` - `gaze_05` | +0.0 | [-2.5, +2.5] | 1 | 1 |
| `full` - `gazef_05` | -0.8 | [-2.5, +0.0] | 0 | 1 |
| `full` - `gazef_05_rand` | +0.0 | [-2.5, +2.5] | 1 | 1 |
| `full` - `pkl` | +0.8 | [+0.0, +2.5] | 1 | 0 |
| `gaze_05` - `gazef_05` | -0.8 | [-3.3, +1.7] | 1 | 2 |
| `gaze_05` - `gazef_05_rand` | +0.0 | [-2.5, +2.5] | 1 | 1 |
| `gaze_05` - `pkl` | +0.8 | [+0.0, +2.5] | 1 | 0 |
| `gazef_05` - `gazef_05_rand` | +0.8 | [-1.7, +3.3] | 2 | 1 |
| `gazef_05` - `pkl` | +1.7 | [+0.0, +4.2] | 2 | 0 |
| `gazef_05_rand` - `pkl` | +0.8 | [+0.0, +2.5] | 1 | 0 |

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
