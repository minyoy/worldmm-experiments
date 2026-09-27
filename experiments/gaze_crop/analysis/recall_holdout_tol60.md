# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3132 targets + 3091 distractors, scope `all-clips`), scored on 380 of 380 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 6.3% | 11.8% | 15.3% | 25.8% | 33.2% | 48.4% | 57 |
| `gazef_05` | 9.2% | 16.3% | 21.6% | 31.8% | 41.3% | 54.5% | 35 |

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | -4.5 | 17 | 34 | 51 | 0.024 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
