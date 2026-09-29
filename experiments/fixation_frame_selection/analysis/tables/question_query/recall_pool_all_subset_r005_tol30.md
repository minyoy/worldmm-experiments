# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 500 of 500 questions.

Target tolerance: 30s -- a clip counts as a hit when it lands within 30s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 3.4% | 6.6% | 9.0% | 16.4% | 22.0% | 34.4% | 166 |
| `center_05` | 4.2% | 8.6% | 12.0% | 18.6% | 25.8% | 38.2% | 108 |
| `gazef_05` | 5.0% | 9.8% | 13.2% | 20.6% | 28.2% | 39.6% | 113 |
| `fixsel` | 5.6% | 9.6% | 11.8% | 16.0% | 22.2% | 34.8% | 145 |
| `fixsel_gaze_05` | 4.6% | 9.0% | 12.6% | 20.0% | 26.6% | 36.8% | 122 |
| `fixsel_gazef_05` | 4.6% | 11.0% | 14.2% | 20.2% | 27.2% | 38.2% | 117 |
| `fixsel_ctl` | 3.4% | 7.2% | 10.4% | 14.6% | 23.4% | 33.4% | 137 |
| _chance_ | _0.33%_ | _0.97%_ | _1.60%_ | _3.07%_ | _5.72%_ | _12.30%_ | _642_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 3, visible pool median 2585)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -2.0 | 14 | 24 | 38 | 0.143 |
| `full` - `gazef_05` | -3.2 | 14 | 30 | 44 | 0.023 |
| `full` - `fixsel` | -3.0 | 7 | 22 | 29 | 0.008 |
| `full` - `fixsel_gaze_05` | -2.4 | 16 | 28 | 44 | 0.096 |
| `full` - `fixsel_gazef_05` | -4.4 | 16 | 38 | 54 | 0.004 |
| `full` - `fixsel_ctl` | -0.6 | 11 | 14 | 25 | 0.690 |
| `center_05` - `gazef_05` | -1.2 | 15 | 21 | 36 | 0.405 |
| `center_05` - `fixsel` | -1.0 | 20 | 25 | 45 | 0.551 |
| `center_05` - `fixsel_gaze_05` | -0.4 | 17 | 19 | 36 | 0.868 |
| `center_05` - `fixsel_gazef_05` | -2.4 | 20 | 32 | 52 | 0.126 |
| `center_05` - `fixsel_ctl` | +1.4 | 22 | 15 | 37 | 0.324 |
| `gazef_05` - `fixsel` | +0.2 | 26 | 25 | 51 | 1.000 |
| `gazef_05` - `fixsel_gaze_05` | +0.8 | 23 | 19 | 42 | 0.644 |
| `gazef_05` - `fixsel_gazef_05` | -1.2 | 14 | 20 | 34 | 0.392 |
| `gazef_05` - `fixsel_ctl` | +2.6 | 30 | 17 | 47 | 0.079 |
| `fixsel` - `fixsel_gaze_05` | +0.6 | 22 | 19 | 41 | 0.755 |
| `fixsel` - `fixsel_gazef_05` | -1.4 | 24 | 31 | 55 | 0.419 |
| `fixsel` - `fixsel_ctl` | +2.4 | 22 | 10 | 32 | 0.050 |
| `fixsel_gaze_05` - `fixsel_gazef_05` | -2.0 | 15 | 25 | 40 | 0.154 |
| `fixsel_gaze_05` - `fixsel_ctl` | +1.8 | 25 | 16 | 41 | 0.211 |
| `fixsel_gazef_05` - `fixsel_ctl` | +3.8 | 34 | 15 | 49 | 0.009 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
