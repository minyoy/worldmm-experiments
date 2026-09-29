# Visual retrieval recall by crop arm (`--query-source keywords`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 487 of 500 questions.

Target tolerance: 0s (strict: only the 30-sec clip EgoLife tagged).

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 2.9% | 5.1% | 6.8% | 10.3% | 14.4% | 23.0% | 279 |
| `center_05` | 3.3% | 5.5% | 7.6% | 12.3% | 17.9% | 29.6% | 253 |
| `gazef_05` | 2.1% | 6.2% | 8.2% | 12.5% | 19.1% | 28.3% | 220 |
| `fixsel` | 3.3% | 4.9% | 7.0% | 10.1% | 15.4% | 24.6% | 265 |
| `fixsel_ctl` | 1.6% | 3.5% | 4.7% | 9.4% | 14.4% | 22.8% | 279 |
| `fixsel_gaze_05` | 1.6% | 5.7% | 9.0% | 12.7% | 18.7% | 30.4% | 232 |
| `fixsel_gazef_05` | 2.3% | 6.2% | 8.2% | 11.7% | 17.7% | 30.6% | 232 |
| _chance_ | _0.11%_ | _0.34%_ | _0.57%_ | _1.14%_ | _2.27%_ | _5.39%_ | _1273_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 1, visible pool median 2621)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `center_05` | -0.4 | 11 | 13 | 24 | 0.839 |
| `full` - `gazef_05` | -1.0 | 13 | 18 | 31 | 0.473 |
| `full` - `fixsel` | +0.2 | 10 | 9 | 19 | 1.000 |
| `full` - `fixsel_ctl` | +1.6 | 11 | 3 | 14 | 0.057 |
| `full` - `fixsel_gaze_05` | -0.6 | 15 | 18 | 33 | 0.728 |
| `full` - `fixsel_gazef_05` | -1.0 | 15 | 20 | 35 | 0.500 |
| `center_05` - `gazef_05` | -0.6 | 11 | 14 | 25 | 0.690 |
| `center_05` - `fixsel` | +0.6 | 14 | 11 | 25 | 0.690 |
| `center_05` - `fixsel_ctl` | +2.1 | 15 | 5 | 20 | 0.041 |
| `center_05` - `fixsel_gaze_05` | -0.2 | 14 | 15 | 29 | 1.000 |
| `center_05` - `fixsel_gazef_05` | -0.6 | 12 | 15 | 27 | 0.701 |
| `gazef_05` - `fixsel` | +1.2 | 16 | 10 | 26 | 0.327 |
| `gazef_05` - `fixsel_ctl` | +2.7 | 19 | 6 | 25 | 0.015 |
| `gazef_05` - `fixsel_gaze_05` | +0.4 | 13 | 11 | 24 | 0.839 |
| `gazef_05` - `fixsel_gazef_05` | +0.0 | 10 | 10 | 20 | 1.000 |
| `fixsel` - `fixsel_ctl` | +1.4 | 11 | 4 | 15 | 0.118 |
| `fixsel` - `fixsel_gaze_05` | -0.8 | 12 | 16 | 28 | 0.572 |
| `fixsel` - `fixsel_gazef_05` | -1.2 | 11 | 17 | 28 | 0.345 |
| `fixsel_ctl` - `fixsel_gaze_05` | -2.3 | 9 | 20 | 29 | 0.061 |
| `fixsel_ctl` - `fixsel_gazef_05` | -2.7 | 8 | 21 | 29 | 0.024 |
| `fixsel_gaze_05` - `fixsel_gazef_05` | -0.4 | 12 | 14 | 26 | 0.845 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
