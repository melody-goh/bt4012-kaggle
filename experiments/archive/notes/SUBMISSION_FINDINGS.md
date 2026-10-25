# What the three scored submissions tell us

## Decision

Keep V2 (0.94785 public ROC-AUC) as the incumbent. V3's historical gains were
concentrated in a few periods, and did not generalise to the public sample. This
is a demonstrated failure of that particular selection decision, not proof that
all ensembles or feature-group models are bad. Three public scores cannot support
a reliable function predicting a candidate's public or private score.

All three original submission files now reproduce to floating-point precision
(maximum absolute difference 1.11e-16) using the original Python 3.10 environment.
The earlier V2 reproduction mismatch arose when running in the newer environment;
the individual package responsible was not isolated. Pin `requirements-original.txt`.
The original files remain unchanged. Submission generation and model lineage are
verified for these three artifacts; no hidden test labels were accessed.

## Observed model ranking

| Model | Validate 8–21, train 1–7 | Validate 15–28, train 1–14 | Validate 22–35, train 1–21 | Public ROC-AUC |
|---|---:|---:|---:|---:|
| V1 original features | 0.993416 | 0.965639 | 0.892850 | 0.94719 |
| V2 upstream graph | 0.993046 | 0.965982 | 0.901627 | **0.94785** |
| V3 group + weighted graph blend | 0.992073 | **0.973443** | **0.913116** | 0.94510 |

Public scores and filenames were recorded from the Kaggle leaderboard page, not retrieved through the API.
Historical results are freshly reconstructed with the environment that reproduces
the submitted files. The initial Python 3.13 development V7 score differed slightly;
use the table above for comparison to these submissions. All windows are 14 steps.
The middle window overlaps both other windows; these are not three independent tests.
Historical periods have been inspected before and are not a new holdout.

V2 improved over V1 by 0.00066 public ROC-AUC. V3 lost 0.00275 against V2, despite
improving the familiar folds by 0.007461 and 0.011489. Local ROC-AUC is neither a
public-score forecast nor on a consistently optimistic/pessimistic offset: public
V2 is lower than one local window and higher than another.

## Strongest diagnostic: gains concentrated in particular periods

- Train through 7: V3 improves 3 steps and worsens 11 relative to V2.
- Train through 14: 7 improve and 7 worsen. Step 24 improves from 0.757235 to
  0.902903. If step 24 is excluded from both models' evaluation, V3's pooled
  advantage flips from +0.007461 to **−0.004173**.
- Train through 21: only 3 steps improve, 10 worsen, and one ties. Improvements
  occur at steps 29, 31 and 32. Step 32 improves from 0.812898 to 0.902483;
  excluding it reduces the pooled advantage from +0.011489 to +0.003667.

These exclusion checks are sensitivity diagnostics, not grounds to discard hard
periods from validation. AUC is not additive over time steps. Removing a step
changes positive/negative pairs and the evaluation population; do not interpret
these differences as additive causal contributions.

A paired bootstrap resampling 14 entire time steps (1,000 draws, seed 42) yields
95% percentile intervals for the historical V3-minus-V2 AUC difference:

| Training cutoff | Observed difference | Historical bootstrap interval |
|---|---:|---:|
| 7 | −0.000973 | [−0.002110, +0.000405] |
| 14 | +0.007461 | [−0.007903, +0.036037] |
| 21 | +0.011489 | [−0.004414, +0.032431] |

Every interval includes zero. These intervals describe sensitivity to historical
step composition. Only 14 clusters are available, adjacent steps may be dependent,
and future distribution shifts are not modeled. These are **not** confidence
intervals for Kaggle scores and do not prove equivalent models.

## V3 substantially changed the test ranking

Spearman correlations over all 15,329 test transactions:

- V1 vs V2: **0.982561**.
- V2 vs V3: **0.723310**.
- V1 vs V3: 0.696541.

The V2/V3 correlation varies by test step (about 0.503–0.872). A ranking change
is not automatically harmful; without test labels it cannot identify the wrong
model or explain the exact 0.00275 leaderboard loss. Overall correlations also
include many low-risk transactions. Their top 5% predicted-risk sets overlap by
90.74%, while their top 1% sets overlap by 61.69% (ceil-based set sizes).

The blend simultaneously changes feature coverage and applies recency weighting.
With no scored component ablations, the public decline cannot be attributed
specifically to the first-94 model, last-71 model or recency weights.

## Why checking only per-step AUC is also insufficient

Across the three historical windows, only 6.84–8.63% of positive/negative pairs
come from the same time step. The other roughly 91–93% compare different steps.
Pooled AUC therefore depends strongly on cross-period ordering as well as
within-period ranking. The exact pair decomposition is saved in
`auc_pair_decomposition.csv`. These pair proportions have not been measured on
hidden test labels. Per-step percentile transforms could damage cross-step
ordering and are not justified by this diagnostic alone.

## One new controlled candidate evaluated

A fixed candidate retains 50% exact V2 predictions and uses 50% of a three-seed
V7-feature ensemble with 90% row and feature sampling. Seeds 17, 42 and 2026;
300 trees, 31 leaves and learning rate .03. No feature-group split or recency
weights. Configuration and advancement criteria were saved before execution.

| Training cutoff | V2 AUC | Candidate AUC | Difference |
|---|---:|---:|---:|
| 7 | 0.993046 | 0.993778 | +0.000732 |
| 14 | 0.965982 | 0.967241 | +0.001258 |
| 21 | 0.901627 | 0.900093 | −0.001534 |

It fails the predeclared requirement to improve both familiar folds. It was not
trained for final submission and no Kaggle submission was made. This avoids
spending one of the limited submissions on another inconsistent local candidate.

## What changes in the search strategy

1. Reproduce the incumbent in its pinned environment and compare every candidate
   to exactly that reference. Keep a ledger of filenames, hashes and actual scores.
2. Keep pooled AUC as the target, but review per-step changes, leave-one-step-out
   sensitivity, and historical uncertainty before promoting a local winner.
3. Avoid tuning a large blend to the two already reused windows. Prefer one
   predeclared hypothesis at a time, then freeze it before further evaluation.
4. The next untested direction is training-only feature stability selection:
   identify features whose contributions persist across inner temporal splits,
   without using outer evaluation or test labels. The short early history makes
   this uncertain; it should first be a controlled ablation, not a large search.
5. A genuine 0.96+ result would require a measured Kaggle gain. Neither these
   diagnostics nor an unsubmitted model establish it. Protect the private test
   objective rather than fitting assumptions to three public scores.

The newly supplied description permits graph features and identifies broad local
and aggregate groups, but does not verify the 94/71 ordering. No external labels
or recovered original labels were used.

## Reproducibility

`diagnose_submissions.py` reconstructs models and persists predictions, metrics,
input/source hashes and environment versions under `runs/submission_diagnosis/`.
`summarize_submissions.py` generates sensitivity results, paired bootstrap intervals
and figures from those predictions. `Submission_Diagnostics.ipynb` reviews them.
`configs/graph_bagging.json` and `runs/graph_bagging_initial/` record the fixed new
candidate. Original root files are protected by `original_checksums.json`.
