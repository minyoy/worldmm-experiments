# Visual retrieval recall by crop arm (`--query-source question`)

> **--self-test run: query embeddings are random. Numbers mean nothing.**

Pool: 6223 clips (3152 targets + 3071 distractors, scope `all-clips`), scored on 86 of 88 questions.

Target tolerance: 0s (strict: only the 30-sec clip EgoLife tagged).

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 0.0% | 0.0% | 0.0% | 0.0% | 1.2% | 2.3% | 1035 |
| `gazef_05` | 0.0% | 0.0% | 0.0% | 0.0% | 1.2% | 4.7% | 858 |
| _chance_ | _0.09%_ | _0.26%_ | _0.43%_ | _0.86%_ | _1.72%_ | _4.23%_ | _1092_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 1, visible pool median 2208)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | +0.0 | 0 | 0 | 0 | 1.000 |

Only the discordant questions carry information about which arm is better, and McNemar asks whether their split is further from even than a coin would give. Six discordant all one way is the first split reaching p < 0.05; below that, a clean-looking 4-0 is not evidence. A small discordant count means 'underpowered', not 'no difference'.

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
