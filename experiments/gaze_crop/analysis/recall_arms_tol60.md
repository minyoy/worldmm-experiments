# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 500 of 500 questions.

Target tolerance: 60s -- a clip counts as a hit when it lands within 60s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 4.8% | 9.2% | 11.8% | 20.4% | 26.8% | 39.6% | 99 |
| `center_05` | 5.4% | 10.8% | 15.0% | 22.0% | 30.0% | 42.4% | 81 |
| `gazef_05` | 7.2% | 13.4% | 17.4% | 25.4% | 33.2% | 44.4% | 76 |
| `gazef_05_randf` | 4.4% | 11.2% | 14.2% | 19.6% | 28.2% | 40.4% | 87 |
| _chance_ | _0.55%_ | _1.60%_ | _2.58%_ | _4.84%_ | _8.75%_ | _17.92%_ | _428_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 5, visible pool median 2585)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -1.6 | 20 | 28 | 48 | 0.312 |
| `full` - `gazef_05` | -4.2 | 17 | 38 | 55 | 0.006 |
| `full` - `gazef_05_randf` | -2.0 | 16 | 26 | 42 | 0.164 |
| `center_05` - `gazef_05` | -2.6 | 17 | 30 | 47 | 0.079 |
| `center_05` - `gazef_05_randf` | -0.4 | 24 | 26 | 50 | 0.888 |
| `gazef_05` - `gazef_05_randf` | +2.2 | 32 | 21 | 53 | 0.169 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
