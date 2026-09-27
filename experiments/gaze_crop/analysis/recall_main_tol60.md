# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (123 targets + 6100 distractors, scope `all-clips`), scored on 120 of 120 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 0.0% | 0.8% | 0.8% | 3.3% | 6.7% | 11.7% | 415 |
| `gazef_05` | 0.8% | 4.2% | 4.2% | 5.0% | 7.5% | 12.5% | 368 |

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | -3.3 | 0 | 4 | 4 | 0.125 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
