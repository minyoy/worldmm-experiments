# Visual retrieval recall by crop arm (`--query-source keywords`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 500 of 500 questions.

Target tolerance: 60s -- a clip counts as a hit when it lands within 60s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 6.0% | 11.0% | 13.8% | 20.2% | 27.0% | 38.0% | 91 |
| `center_05` | 6.8% | 10.8% | 14.2% | 22.4% | 30.0% | 43.0% | 76 |
| `gazef_05` | 5.4% | 12.0% | 16.2% | 23.8% | 34.2% | 46.0% | 71 |
| `gazef_05_randf` | 6.4% | 12.0% | 14.4% | 23.0% | 30.6% | 41.6% | 95 |
| _chance_ | _0.55%_ | _1.60%_ | _2.58%_ | _4.84%_ | _8.75%_ | _17.92%_ | _428_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 5, visible pool median 2585)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | +0.2 | 20 | 19 | 39 | 1.000 |
| `full` - `gazef_05` | -1.0 | 26 | 31 | 57 | 0.597 |
| `full` - `gazef_05_randf` | -1.0 | 21 | 26 | 47 | 0.560 |
| `center_05` - `gazef_05` | -1.2 | 18 | 24 | 42 | 0.441 |
| `center_05` - `gazef_05_randf` | -1.2 | 17 | 23 | 40 | 0.430 |
| `gazef_05` - `gazef_05_randf` | +0.0 | 23 | 23 | 46 | 1.000 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
