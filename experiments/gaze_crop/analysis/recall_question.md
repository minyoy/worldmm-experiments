# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (123 targets + 6100 distractors, scope `all-clips`), scored on 120 of 120 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 2.5% | 4.2% | 5.0% | 8.3% | 15.8% | 25.8% | 142 |
| `gazef_05` | 1.7% | 6.7% | 8.3% | 10.0% | 16.7% | 28.3% | 142 |

Paired differences at k=20:

| arms | diff (pp) | 95% CI | a only | b only |
|---|---|---|---|---|
| `full` - `gazef_05` | -0.8 | [-7.5, +5.0] | 7 | 8 |

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
