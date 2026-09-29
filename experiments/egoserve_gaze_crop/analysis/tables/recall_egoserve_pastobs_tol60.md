# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 88 of 88 questions.

Target tolerance: 60s -- a clip counts as a hit when it lands within 60s of the annotated moment, so recall here is NOT comparable to a strict table. The chance row is what an uninformed ranker scores at this tolerance.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 3.4% | 6.8% | 10.2% | 15.9% | 26.1% | 35.2% | 130 |
| `center_05` | 3.4% | 10.2% | 15.9% | 22.7% | 26.1% | 40.9% | 115 |
| `gazef_05_randf` | 4.5% | 6.8% | 10.2% | 12.5% | 23.9% | 33.0% | 110 |
| `gazef_05` | 2.3% | 10.2% | 15.9% | 26.1% | 30.7% | 43.2% | 84 |
| _chance_ | _0.36%_ | _1.07%_ | _1.77%_ | _3.49%_ | _6.79%_ | _15.69%_ | _364_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 5, visible pool median 2208)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -3.4 | 3 | 6 | 9 | 0.508 |
| `full` - `gazef_05_randf` | +0.0 | 3 | 3 | 6 | 1.000 |
| `full` - `gazef_05` | -3.4 | 3 | 6 | 9 | 0.508 |
| `center_05` - `gazef_05_randf` | +3.4 | 6 | 3 | 9 | 0.508 |
| `center_05` - `gazef_05` | +0.0 | 5 | 5 | 10 | 1.000 |
| `gazef_05_randf` - `gazef_05` | -3.4 | 3 | 6 | 9 | 0.508 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
