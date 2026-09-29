# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 88 of 88 questions.

Target tolerance: 30s -- a clip counts as a hit when it lands within 30s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 2.3% | 4.5% | 6.8% | 13.6% | 23.9% | 31.8% | 152 |
| `center_05` | 3.4% | 8.0% | 12.5% | 17.0% | 21.6% | 37.5% | 126 |
| `gazef_05_randf` | 4.5% | 6.8% | 9.1% | 11.4% | 20.5% | 30.7% | 121 |
| `gazef_05` | 2.3% | 6.8% | 11.4% | 22.7% | 26.1% | 38.6% | 110 |
| _chance_ | _0.23%_ | _0.68%_ | _1.13%_ | _2.24%_ | _4.40%_ | _10.45%_ | _546_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 3, visible pool median 2208)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -3.4 | 2 | 5 | 7 | 0.453 |
| `full` - `gazef_05_randf` | -2.3 | 1 | 3 | 4 | 0.625 |
| `full` - `gazef_05` | -2.3 | 2 | 4 | 6 | 0.688 |
| `center_05` - `gazef_05_randf` | +1.1 | 4 | 3 | 7 | 1.000 |
| `center_05` - `gazef_05` | +1.1 | 3 | 2 | 5 | 1.000 |
| `gazef_05_randf` - `gazef_05` | +0.0 | 4 | 4 | 8 | 1.000 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
