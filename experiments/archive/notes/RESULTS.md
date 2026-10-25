# Initial run results

**Follow-up:** The original Python 3.10 environment was found in saved notebook
output. All three submissions now reproduce at tolerance 1e-10. The reproduction
uncertainty below describes the initial Python 3.13 run and is resolved; see
`SUBMISSION_FINDINGS.md` and `requirements-original.txt`. Supplied competition
documentation also explicitly permits using the supplied graph.

## Decision

Retain the existing V2 submission as the working benchmark. The frozen regularised
graph candidate lost on both evaluated windows; no replacement submission was
produced or sent to Kaggle. The locally reproduced V7 predictions are diagnostic
artifacts, not an authenticated replacement for the saved V2 file.

| Model | Development 8–21 ROC-AUC | Historical check 22–35 ROC-AUC |
|---|---:|---:|
| Original 165 features | 0.993416 | 0.892850 |
| V7 upstream graph | 0.993487 | 0.901627 |
| Regularised graph | 0.992687 | 0.896788 |

The new candidate trails V7 by 0.000800 and 0.004839 respectively. It also has lower
average precision on both windows. This single frozen configuration does not rule
out every form of regularisation, but does not justify a tuning sweep or a submission.
The historical-check baseline and graph AUCs exactly match `v7_results.csv`.
The development window is much easier despite its smaller training set, reinforcing
that forecast horizon alone does not capture period difficulty.

## V2 reproduction

Both four threads and the notebook's `n_jobs=-1` produce identical reproduction
results against saved `submission_v2.csv`:

- Mean absolute probability difference: 0.00046385352148397196.
- Maximum absolute probability difference: 0.04403641717703971.
- Spearman rank correlation: 0.9993427021673955.
- Exact reproduction at absolute tolerance 1e-10: **failed**.

Do not attribute this conclusively to package versions: the original versions are
unknown, the final filename is absent from notebook source, and the historical
fold results match exactly. Final-file provenance or execution differences remain
unresolved. Neither the original notebook nor successful submission files were altered.

## Verified data and submission checks

141,772 train rows; 31,235 labelled (3,644 illicit), 110,537 unknown; 15,329 test rows.
All existing `submission*.csv` files passed row count, exact index ordering, unique
indices, and finite [0,1] probability checks against test and sample submission.
These checks establish structure, not the training provenance of each file.

234,355 edges with no exact duplicate edges. 203,769 distinct graph nodes;
175,918 edges have feature-bearing endpoints, and all of these are within a step.
60.317% of test nodes have no available predecessor feature vector. This limits
upstream feature coverage; it is not evidence that the graph is useless.

Each run retains per-step results, predictions, trained model files, and source/data
hashes. Three focused safety tests pass. The original-file checksum manifest allows
independent verification of preservation.

## Limits

No untouched historical holdout remains. No confidence interval for private-test
performance is claimed. Rules and the feature-order semantic mapping are unavailable.
Public scores are taken from the Kaggle leaderboard and were not newly evaluated.
Only the first proposed candidate was implemented; see `AUDIT.md` for the next two.
