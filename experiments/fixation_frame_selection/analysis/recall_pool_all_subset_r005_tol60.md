# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 500 of 500 questions.

Target tolerance: 60s -- a clip counts as a hit when it lands within 60s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 4.8% | 9.2% | 11.8% | 20.4% | 26.8% | 39.6% | 99 |
| `gazef_05` | 7.2% | 13.4% | 17.4% | 25.4% | 33.2% | 44.4% | 76 |
| `center_05` | 5.4% | 10.8% | 15.0% | 22.0% | 30.0% | 42.4% | 81 |
| `fixsel` | 6.6% | 12.2% | 15.0% | 22.0% | 28.6% | 41.2% | 91 |
| `fixsel_gaze_05` | 6.6% | 10.6% | 15.0% | 22.8% | 30.2% | 41.0% | 80 |
| `fixsel_ctl` | 4.0% | 8.0% | 12.4% | 17.4% | 27.4% | 39.0% | 105 |
| _chance_ | _0.55%_ | _1.60%_ | _2.58%_ | _4.84%_ | _8.75%_ | _17.92%_ | _428_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 5, visible pool median 2585)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | -4.2 | 17 | 38 | 55 | 0.006 |
| `full` - `center_05` | -1.6 | 20 | 28 | 48 | 0.312 |
| `full` - `fixsel` | -3.0 | 9 | 24 | 33 | 0.014 |
| `full` - `fixsel_gaze_05` | -1.4 | 21 | 28 | 49 | 0.392 |
| `full` - `fixsel_ctl` | +1.2 | 17 | 11 | 28 | 0.345 |
| `gazef_05` - `center_05` | +2.6 | 30 | 17 | 47 | 0.079 |
| `gazef_05` - `fixsel` | +1.2 | 33 | 27 | 60 | 0.519 |
| `gazef_05` - `fixsel_gaze_05` | +2.8 | 32 | 18 | 50 | 0.065 |
| `gazef_05` - `fixsel_ctl` | +5.4 | 44 | 17 | 61 | 0.001 |
| `center_05` - `fixsel` | -1.4 | 22 | 29 | 51 | 0.401 |
| `center_05` - `fixsel_gaze_05` | +0.2 | 19 | 18 | 37 | 1.000 |
| `center_05` - `fixsel_ctl` | +2.8 | 30 | 16 | 46 | 0.054 |
| `fixsel` - `fixsel_gaze_05` | +1.6 | 28 | 20 | 48 | 0.312 |
| `fixsel` - `fixsel_ctl` | +4.2 | 30 | 9 | 39 | 0.001 |
| `fixsel_gaze_05` - `fixsel_ctl` | +2.6 | 30 | 17 | 47 | 0.079 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
