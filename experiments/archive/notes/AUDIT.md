# Project audit and experiment plan

**Follow-up:** The original Python 3.10 installation was located from saved
notebook output. V1, V2 and V3 now reproduce to floating-point precision. The
user-supplied competition description explicitly permits the provided graph.
Earlier provenance/rules gaps below record the initial audit state; see
`SUBMISSION_FINDINGS.md` and `DATA_CONTEXT.md` for the updated conclusions.

## Evidence inspected

The project notes, the 33-cell structure of `BT4012_Bitcoin_Baseline_V1.ipynb`
(selected modeling cells inspected in detail), its saved metadata,
root CSV artifacts, and the two supplied workspace directories. The second
workspace directory is empty. No README or competition rules were present.
Notebook metadata records Python 3.10.11; the original dependency versions are
not recorded. The new isolated environment is pinned separately.

## Confirmed reproducibility issues

1. Cell 13 (zero-based) builds a submission from generic `pred`; cell 12 leaves
   `pred` containing validation predictions for steps 29–35. Running these cells
   consecutively does not establish test predictions. This is a hidden-state bug;
   it does not prove that the existing V1 CSV was incorrect when produced.
2. Cell 15 defines the 18 V7 upstream features. Cell 20 fits the full-labelled-data
   V7 model and writes `submission_graph_all.csv`, whereas the saved benchmark is
   called `submission_v2.csv`. Numerical comparison is required to connect them.
3. The final cell writes `submission_v20_group_graph.csv`; the preserved artifact
   is `submission_v3.csv`. Filenames alone do not prove model lineage.
4. The notebook reuses many mutable globals, performs an in-notebook package
   installation, and contains many modeling stages. Exact top-to-bottom execution
   is not established. The original notebook is intentionally not executed because
   it contains writes to successful submission files.
5. The inspected V7 joins validate cardinality and preserve ordering. Unknown
   labels are excluded from supervised training. Graph aggregation uses feature
   values, not neighbour labels. No target leakage was identified in that code.

## Why local improvements may not transfer

The old 15–28 and 22–35 validation windows share seven time steps and were reused
for many choices. That supports a concern about adaptive model selection, not a
proven causal explanation for the V3 leaderboard decline. Historical period
heterogeneity, limited public sampling and differences in model ranking between
future periods are plausible alternatives. Public labels are unavailable, so the
cause cannot be isolated here. Public scores are taken from the Kaggle leaderboard,
not newly verified Kaggle results.

## Three proposed experiments, in priority order

1. **Regularised V7 graph model (implemented).** Freeze 15 leaves, minimum 100
   observations per leaf, L2=10 and 80% feature sampling while retaining 300 trees
   and learning rate .03. Test whether constraining splits reduces sensitivity to
   period-specific patterns. The old V6 regularisation experiment used original
   features and a different configuration; this tests the publicly best feature
   representation under the new evaluation protocol. This is one hypothesis,
   not a parameter sweep. Accept only consistent development and later historical
   evidence, with per-step results reviewed before considering deployment.
2. **Historical stability selection (planned).** Estimate feature importance
   stability using inner chronological splits entirely inside each outer training
   period; select/prune only from those inner results. Compare with V7 using the
   same outer windows. Avoid pruning based on test distribution or evaluation labels.
3. **A structurally different tabular model (planned).** Evaluate a regularised
   linear model with training-fitted transformations and carefully controlled
   nonlinear feature expansions. Preserve unknown-label exclusion. Study prediction
   complementarity only if its performance is credible; do not tune blend weights
   on the later check. Lower implementation cost than a GNN, with no unverified
   assumptions about feature order or external pretraining.

The second and third experiments are proposed, not claimed as completed. GNNs,
pseudo-labels and broader transductive representations are deferred until the
competition rules and a stronger model-selection protocol justify their complexity.
