# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 86 of 88 questions.

Target tolerance: 0s (strict: only the 30-sec clip EgoLife tagged).

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 2.3% | 3.5% | 4.7% | 10.5% | 17.4% | 24.4% | 267 |
| `center_05` | 3.5% | 4.7% | 8.1% | 11.6% | 15.1% | 24.4% | 262 |
| `gazef_05_randf` | 2.3% | 5.8% | 9.3% | 10.5% | 17.4% | 22.1% | 288 |
| `gazef_05` | 1.2% | 3.5% | 8.1% | 14.0% | 19.8% | 31.4% | 232 |
| _chance_ | _0.09%_ | _0.26%_ | _0.43%_ | _0.86%_ | _1.72%_ | _4.23%_ | _1092_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 1, visible pool median 2208)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -1.2 | 1 | 2 | 3 | 1.000 |
| `full` - `gazef_05_randf` | -2.3 | 1 | 3 | 4 | 0.625 |
| `full` - `gazef_05` | +0.0 | 2 | 2 | 4 | 1.000 |
| `center_05` - `gazef_05_randf` | -1.2 | 2 | 3 | 5 | 1.000 |
| `center_05` - `gazef_05` | +1.2 | 2 | 1 | 3 | 1.000 |
| `gazef_05_randf` - `gazef_05` | +2.3 | 4 | 2 | 6 | 0.688 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
