# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 487 of 500 questions.

Target tolerance: 0s (strict: only the 30-sec clip EgoLife tagged).

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 1.2% | 3.1% | 4.5% | 8.4% | 12.7% | 23.4% | 289 |
| `gazef_05` | 2.5% | 5.1% | 6.4% | 12.7% | 17.5% | 28.1% | 249 |
| `fixsel` | 2.5% | 4.3% | 6.0% | 10.5% | 13.3% | 23.8% | 292 |
| `fixsel_gaze_05` | 2.5% | 5.3% | 7.6% | 12.5% | 17.5% | 26.1% | 261 |
| `fixsel_ctl` | 0.8% | 2.7% | 5.5% | 8.0% | 13.8% | 20.7% | 268 |
| _chance_ | _0.11%_ | _0.34%_ | _0.57%_ | _1.14%_ | _2.27%_ | _5.39%_ | _1273_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 1, visible pool median 2621)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | -2.1 | 9 | 19 | 28 | 0.087 |
| `full` - `fixsel` | -1.2 | 9 | 15 | 24 | 0.307 |
| `full` - `fixsel_gaze_05` | -2.3 | 9 | 20 | 29 | 0.061 |
| `full` - `fixsel_ctl` | +0.4 | 10 | 8 | 18 | 0.815 |
| `gazef_05` - `fixsel` | +0.8 | 16 | 12 | 28 | 0.572 |
| `gazef_05` - `fixsel_gaze_05` | -0.2 | 13 | 14 | 27 | 1.000 |
| `gazef_05` - `fixsel_ctl` | +2.5 | 20 | 8 | 28 | 0.036 |
| `fixsel` - `fixsel_gaze_05` | -1.0 | 12 | 17 | 29 | 0.458 |
| `fixsel` - `fixsel_ctl` | +1.6 | 17 | 9 | 26 | 0.169 |
| `fixsel_gaze_05` - `fixsel_ctl` | +2.7 | 21 | 8 | 29 | 0.024 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
