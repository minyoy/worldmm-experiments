# Visual retrieval recall, all 500 questions, strict targets (`--query-source question`, tol 0s)

Merged, not a separate run: `recall_main_tol0.json` (120q) + `recall_holdout_tol0.json` (367q). Same embeddings, pool, tolerance and query source, disjoint question sets, so this is what one run over all of them would have printed.

Target tolerance: 0s (not recorded in the inputs; supplied on the command line from the command that produced them), scored on 487 of 500 questions.

| arm | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | median rank |
|---|---|---|---|---|---|---|---|
| `full` | 1.2% | 3.1% | 4.5% | 8.4% | 12.7% | 23.4% | 289 |
| `gazef_05` | 2.5% | 5.1% | 6.4% | 12.7% | 17.5% | 28.1% | 249 |
| _chance_ | _0.11%_ | _0.34%_ | _0.57%_ | _1.14%_ | _2.27%_ | _5.39%_ | _1273_ |

(chance assumes a random ranking of each question's own visible pool: targets/question median 1, visible pool median 2621)

Paired differences at k=3:

| arms | diff (pp) | a only | b only | discordant | McNemar p |
|---|---|---|---|---|---|
| `full` - `gazef_05` | -2.1 | 9 | 19 | 28 | 0.087 |

By need_audio at k=3 (False = the visual-evidence questions):

| need_audio | n | `full` | `gazef_05` |
|---|---|---|---|
| False | 287 | 3.8% | 5.2% |
| True | 200 | 2.0% | 5.0% |
