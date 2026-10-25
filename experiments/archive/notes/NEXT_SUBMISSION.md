# Scored experiment: 75% V2 + 25% time-balanced graph LightGBM

**Actual Kaggle public score: 0.94770**, versus V2 0.94785
(difference −0.00015). This candidate did not improve the incumbent. The
user renamed the delivered copy to `submissions/submission_v4.csv`; its hash
matches the generated file exactly. The text below records the original
pre-submission rationale; do not resubmit the same file.

Upload `submissions/submission_next_step_balanced_graph.csv` as the next controlled
Kaggle experiment. This is the best-supported **new candidate from this round**,
not a proven improvement on the public or private leaderboard. No Kaggle upload
has been performed and no new leaderboard score is claimed.

## What changed

Retain the exact publicly tested V2 prediction with weight 0.75. Train the same
183-feature LightGBM (165 originals plus 18 V7 graph features, 300 trees, learning
rate .03, 31 leaves, seed 42) with equal total sample weight per historical time
step. Each labelled transaction receives weight `1 / labelled_count_in_its_step`,
then weights are normalized to mean one. Blend this model at weight 0.25.

Unknown-label rows remain excluded from supervised fitting. The graph uses supplied
features without neighbour labels. No feature-group semantics, external data,
pseudo-labels or test targets are used. The final model fits all 31,235 eligible
labelled training transactions through step 35. The fixed 25% blend was declared
before results; no blend-weight optimization was performed.

## Validation

All windows have 14-step horizons. Train begins at step 1 in each case.

| Training cutoff | Validation steps | V2 ROC-AUC | Candidate ROC-AUC | Difference |
|---|---|---:|---:|---:|
| 7 | 8–21 | 0.993046 | 0.993614 | +0.000568 |
| 14 | 15–28 | 0.965982 | 0.967976 | +0.001993 |
| 21 | 22–35 | 0.901627 | 0.901540 | −0.000087 |
| 10 | 11–24 | 0.972417 | 0.973238 | +0.000821 |
| 17 | 18–31 | 0.889732 | 0.898707 | +0.008976 |

The last two cutoffs were evaluated after freezing this candidate. They are
additional-cutoff checks, **not independent untouched holdouts**: their periods
overlap familiar windows. Improvement in four of five windows is encouraging,
but these five results must not be treated as five independent experiments.
The previously reused 22–35 window is essentially tied, not improved.

The screening rule required a positive mean development difference, no development
loss worse than 0.0005, a positive mean additional-check difference, and no
additional-check loss worse than 0.001. The candidate passed. These are practical
screening tolerances, not statistical significance thresholds or score guarantees.
In each of the four improving windows, the gain remains positive after removing
any single validation time step. Minimum leave-one-step-out gains are +0.000417,
+0.000945, +0.000019 and +0.004736 for training cutoffs 7, 14, 10 and 17 respectively.
This is a useful robustness check against the earlier V3 failure pattern, though
it does not remove all selection bias or forecast future performance. Detailed
per-step and sensitivity metrics are retained in the run directory.

Test rank correlation with V2: **0.998729**. This is much closer to the incumbent
than V3 was (0.723310), but proximity does not prove improved accuracy.

## Other candidates rejected

Three new candidates were first tested with the same 75/25 V2 blend:

| Candidate | Mean AUC difference across initial windows | Worst difference |
|---|---:|---:|
| Training-prefix feature stability selection | −0.001065 | −0.002374 |
| Extra Trees | −0.026014 | −0.073264 |
| DART dropout boosting | −0.008814 | −0.018020 |

None passed screening. Their configurations, models, raw component scores and
blend results are retained in `runs/next_submission_01`. Probability blending can
interact poorly with different model score scales; these results do not rule out
all calibrated variants or every configuration of these model families.

Time-step balancing was then revisited because the pre-existing V16 results had
shown gains in both familiar windows and it had not been submitted. The fixed
new blend was screened under the same tolerances; it was not selected by changing
the failed candidates' parameters after seeing their scores. This is a sequential
search with four candidate families, so selection bias remains possible.

## Reproduction and checks

Use Python 3.10 and `requirements-original.txt`. From the project root:

```sh
python3.10 experiments/prepare_next_submission.py \
  --config experiments/configs/next_submission_step_balance.json \
  --run-id YOUR_NEW_RUN_ID
```

The script preserves existing run directories and records configuration, package
versions, input hashes and snapshots of the implementation. It saves models,
feature lists, predictions, metrics, and the selection decision before the
additional checks. It produces a new candidate only if the frozen criteria pass.

`verify_next_submission.py` replays the saved final model for this delivered run,
validates the CSV against both test ordering and sample submission, and confirms
preservation of all 25 original files. Its receipt is
`runs/next_submission_step_balance/verification.json`. The final model is
`final_step_balanced_graph__model.joblib`; the other blend component is the original
`submission_v2.csv`, identified by hash in the manifest.

Keep V2 available as the incumbent. Record the actual Kaggle result for this exact
CSV before treating it as a better model. Do not infer a 0.96+ score from these
historical gains, and do not spend another submission on the unchanged V2 copies
inside rejected experiment directories.
