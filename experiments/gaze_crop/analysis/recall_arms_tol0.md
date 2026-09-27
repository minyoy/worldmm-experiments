# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 487 of 500 questions.

Target tolerance: 0s (strict: only the 30-sec clip EgoLife tagged).

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 1.2% | 3.1% | 4.5% | 8.4% | 12.7% | 23.4% | 289 |
| `center_05` | 1.6% | 4.7% | 6.8% | 11.3% | 16.6% | 25.9% | 245 |
| `gazef_05` | 2.5% | 5.1% | 6.4% | 12.7% | 17.5% | 28.1% | 249 |
| `gazef_05_randf` | 1.0% | 4.1% | 5.5% | 8.6% | 12.3% | 22.8% | 292 |
| _chance_ | _0.11%_ | _0.34%_ | _0.57%_ | _1.14%_ | _2.27%_ | _5.39%_ | _1273_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 1, visible pool median 2621)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -1.6 | 6 | 14 | 20 | 0.115 |
| `full` - `gazef_05` | -2.1 | 9 | 19 | 28 | 0.087 |
| `full` - `gazef_05_randf` | -1.0 | 8 | 13 | 21 | 0.383 |
| `center_05` - `gazef_05` | -0.4 | 14 | 16 | 30 | 0.856 |
| `center_05` - `gazef_05_randf` | +0.6 | 15 | 12 | 27 | 0.701 |
| `gazef_05` - `gazef_05_randf` | +1.0 | 17 | 12 | 29 | 0.458 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
