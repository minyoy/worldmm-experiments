# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (123 targets + 6100 distractors, scope `all-clips`), scored on 120 of 120 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 0.0% | 0.8% | 0.8% | 3.3% | 6.7% | 11.7% | 415 |
| `gazef_05` | 0.8% | 4.2% | 4.2% | 5.0% | 7.5% | 12.5% | 368 |

Paired differences at k=5:

| arms | diff (pp) | 95% CI | a only | b only |
|---|---|---|---|---|
| `full` - `gazef_05` | -3.3 | [-6.7, -0.8] | 0 | 4 |

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
