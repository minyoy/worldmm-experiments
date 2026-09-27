# Visual retrieval recall by crop arm (`--query-source question`)

Pool: 6223 clips (123 targets + 6100 distractors, scope `all-clips`), scored on 120 of 120 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 0.0% | 0.8% | 0.8% | 1.7% | 2.5% | 3.3% | 878 |
| `gazef_05` | 0.8% | 0.8% | 1.7% | 2.5% | 2.5% | 5.8% | 909 |

Paired differences at k=3:

| arms | diff (pp) | 95% CI | a only | b only |
|---|---|---|---|---|
| `full` - `gazef_05` | +0.0 | [-2.5, +2.5] | 1 | 1 |

How to read it: `center@R` is the control. `gaze@R` beating `full` but not `center@R` means cropping helped and gaze did not.
