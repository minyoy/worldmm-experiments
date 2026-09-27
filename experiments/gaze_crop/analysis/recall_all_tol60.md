# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 500 of 500 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 4.8% | 9.2% | 11.8% | 20.4% | 26.8% | 39.6% | 99 |
| `gazef_05` | 7.2% | 13.4% | 17.4% | 25.4% | 33.2% | 44.4% | 76 |

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | -4.2 | 17 | 38 | 55 | 0.006 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
