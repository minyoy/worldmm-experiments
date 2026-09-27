# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (3132 targets + 3091 distractors, scope `all-clips`), scored on 367 of 380 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 1.6% | 3.8% | 5.7% | 10.6% | 16.1% | 30.0% | 204 |
| `gazef_05` | 3.0% | 6.5% | 7.9% | 16.1% | 22.3% | 35.4% | 142 |

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | -2.7 | 8 | 18 | 26 | 0.076 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
