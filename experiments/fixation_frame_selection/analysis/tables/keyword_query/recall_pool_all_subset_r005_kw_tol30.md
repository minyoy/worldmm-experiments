# Visual retrieval recall by crop arm (`--query-source keywords`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 500 of 500 questions.

Target tolerance: 30s -- a clip counts as a hit when it lands within 30s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 5.2% | 8.8% | 11.0% | 15.8% | 22.2% | 33.6% | 133 |
| `center_05` | 5.8% | 8.6% | 11.2% | 18.2% | 26.4% | 38.6% | 113 |
| `gazef_05` | 3.6% | 9.8% | 12.0% | 18.6% | 29.6% | 41.4% | 97 |
| `fixsel` | 5.2% | 8.8% | 11.6% | 16.6% | 25.0% | 36.0% | 139 |
| `fixsel_ctl` | 4.6% | 8.0% | 9.8% | 15.6% | 22.4% | 34.8% | 125 |
| `fixsel_gaze_05` | 2.6% | 8.6% | 13.6% | 18.4% | 28.2% | 42.4% | 83 |
| `fixsel_gazef_05` | 3.6% | 9.8% | 13.8% | 19.2% | 27.6% | 42.4% | 87 |
| _chance_ | _0.33%_ | _0.97%_ | _1.60%_ | _3.07%_ | _5.72%_ | _12.30%_ | _642_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 3, visible pool median 2585)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | +0.2 | 17 | 16 | 33 | 1.000 |
| `full` - `gazef_05` | -1.0 | 20 | 25 | 45 | 0.551 |
| `full` - `fixsel` | +0.0 | 14 | 14 | 28 | 1.000 |
| `full` - `fixsel_ctl` | +0.8 | 16 | 12 | 28 | 0.572 |
| `full` - `fixsel_gaze_05` | +0.2 | 22 | 21 | 43 | 1.000 |
| `full` - `fixsel_gazef_05` | -1.0 | 20 | 25 | 45 | 0.551 |
| `center_05` - `gazef_05` | -1.2 | 15 | 21 | 36 | 0.405 |
| `center_05` - `fixsel` | -0.2 | 18 | 19 | 37 | 1.000 |
| `center_05` - `fixsel_ctl` | +0.6 | 19 | 16 | 35 | 0.736 |
| `center_05` - `fixsel_gaze_05` | +0.0 | 20 | 20 | 40 | 1.000 |
| `center_05` - `fixsel_gazef_05` | -1.2 | 18 | 24 | 42 | 0.441 |
| `gazef_05` - `fixsel` | +1.0 | 25 | 20 | 45 | 0.551 |
| `gazef_05` - `fixsel_ctl` | +1.8 | 27 | 18 | 45 | 0.233 |
| `gazef_05` - `fixsel_gaze_05` | +1.2 | 18 | 12 | 30 | 0.362 |
| `gazef_05` - `fixsel_gazef_05` | +0.0 | 15 | 15 | 30 | 1.000 |
| `fixsel` - `fixsel_ctl` | +0.8 | 17 | 13 | 30 | 0.585 |
| `fixsel` - `fixsel_gaze_05` | +0.2 | 19 | 18 | 37 | 1.000 |
| `fixsel` - `fixsel_gazef_05` | -1.0 | 21 | 26 | 47 | 0.560 |
| `fixsel_ctl` - `fixsel_gaze_05` | -0.6 | 18 | 21 | 39 | 0.749 |
| `fixsel_ctl` - `fixsel_gazef_05` | -1.8 | 22 | 31 | 53 | 0.272 |
| `fixsel_gaze_05` - `fixsel_gazef_05` | -1.2 | 16 | 22 | 38 | 0.418 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
