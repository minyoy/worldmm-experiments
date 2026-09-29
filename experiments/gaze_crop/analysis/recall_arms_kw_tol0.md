# Visual retrieval recall by crop arm (`--query-source keywords`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 487 of 500 questions.

Target tolerance: 0s (strict: only the 30-sec clip EgoLife tagged).

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 2.9% | 5.1% | 6.8% | 10.3% | 14.4% | 23.0% | 279 |
| `center_05` | 3.3% | 5.5% | 7.6% | 12.3% | 17.9% | 29.6% | 253 |
| `gazef_05` | 2.1% | 6.2% | 8.2% | 12.5% | 19.1% | 28.3% | 220 |
| `gazef_05_randf` | 1.6% | 3.9% | 5.3% | 9.9% | 13.6% | 23.4% | 301 |
| _chance_ | _0.11%_ | _0.34%_ | _0.57%_ | _1.14%_ | _2.27%_ | _5.39%_ | _1273_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 1, visible pool median 2621)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -0.4 | 11 | 13 | 24 | 0.839 |
| `full` - `gazef_05` | -1.0 | 13 | 18 | 31 | 0.473 |
| `full` - `gazef_05_randf` | +1.2 | 14 | 8 | 22 | 0.286 |
| `center_05` - `gazef_05` | -0.6 | 11 | 14 | 25 | 0.690 |
| `center_05` - `gazef_05_randf` | +1.6 | 16 | 8 | 24 | 0.152 |
| `gazef_05` - `gazef_05_randf` | +2.3 | 20 | 9 | 29 | 0.061 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
