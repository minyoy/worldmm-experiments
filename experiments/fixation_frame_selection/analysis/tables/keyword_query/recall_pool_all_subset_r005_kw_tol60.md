# Visual retrieval recall by crop arm (`--query-source keywords`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 500 of 500 questions.

Target tolerance: 60s -- a clip counts as a hit when it lands within 60s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 6.0% | 11.0% | 13.8% | 20.2% | 27.0% | 38.0% | 91 |
| `center_05` | 6.8% | 10.8% | 14.2% | 22.4% | 30.0% | 43.0% | 76 |
| `gazef_05` | 5.4% | 12.0% | 16.2% | 23.8% | 34.2% | 46.0% | 71 |
| `fixsel` | 6.8% | 12.6% | 15.4% | 19.8% | 28.6% | 41.8% | 91 |
| `fixsel_ctl` | 6.0% | 10.2% | 12.6% | 19.0% | 27.4% | 39.8% | 97 |
| `fixsel_gaze_05` | 4.0% | 11.4% | 17.0% | 22.8% | 32.4% | 46.4% | 67 |
| `fixsel_gazef_05` | 5.2% | 12.2% | 16.4% | 24.8% | 32.2% | 46.0% | 67 |
| _chance_ | _0.55%_ | _1.60%_ | _2.58%_ | _4.84%_ | _8.75%_ | _17.92%_ | _428_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 5, visible pool median 2585)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | +0.2 | 20 | 19 | 39 | 1.000 |
| `full` - `gazef_05` | -1.0 | 26 | 31 | 57 | 0.597 |
| `full` - `fixsel` | -1.6 | 14 | 22 | 36 | 0.243 |
| `full` - `fixsel_ctl` | +0.8 | 18 | 14 | 32 | 0.597 |
| `full` - `fixsel_gaze_05` | -0.4 | 24 | 26 | 50 | 0.888 |
| `full` - `fixsel_gazef_05` | -1.2 | 21 | 27 | 48 | 0.471 |
| `center_05` - `gazef_05` | -1.2 | 18 | 24 | 42 | 0.441 |
| `center_05` - `fixsel` | -1.8 | 21 | 30 | 51 | 0.262 |
| `center_05` - `fixsel_ctl` | +0.6 | 25 | 22 | 47 | 0.771 |
| `center_05` - `fixsel_gaze_05` | -0.6 | 20 | 23 | 43 | 0.761 |
| `center_05` - `fixsel_gazef_05` | -1.4 | 22 | 29 | 51 | 0.401 |
| `gazef_05` - `fixsel` | -0.6 | 30 | 33 | 63 | 0.801 |
| `gazef_05` - `fixsel_ctl` | +1.8 | 33 | 24 | 57 | 0.289 |
| `gazef_05` - `fixsel_gaze_05` | +0.6 | 21 | 18 | 39 | 0.749 |
| `gazef_05` - `fixsel_gazef_05` | -0.2 | 21 | 22 | 43 | 1.000 |
| `fixsel` - `fixsel_ctl` | +2.4 | 25 | 13 | 38 | 0.073 |
| `fixsel` - `fixsel_gaze_05` | +1.2 | 29 | 23 | 52 | 0.488 |
| `fixsel` - `fixsel_gazef_05` | +0.4 | 30 | 28 | 58 | 0.896 |
| `fixsel_ctl` - `fixsel_gaze_05` | -1.2 | 23 | 29 | 52 | 0.488 |
| `fixsel_ctl` - `fixsel_gazef_05` | -2.0 | 23 | 33 | 56 | 0.229 |
| `fixsel_gaze_05` - `fixsel_gazef_05` | -0.8 | 19 | 23 | 42 | 0.644 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
