# BT4012 experimental pipeline

The notebooks and CSVs in the project root are preserved. `original_checksums.json`
records their starting SHA-256 hashes. All new artifacts live here.

## Run

For exact reproduction, use Python 3.10 and `requirements-original.txt`. The initial Python 3.13 environment below runs the pipeline but changes some graph-derived predictions; do not treat it as an exact reproduction environment.

```sh
python3.10 -m venv experiments/.venv-original
experiments/.venv-original/bin/python -m pip install -r experiments/requirements-original.txt
experiments/.venv-original/bin/python experiments/run.py reproduce-v2
```

Initial investigation environment (Python 3.13):

```sh
python3 -m venv experiments/.venv
experiments/.venv/bin/python -m pip install -r experiments/requirements.txt
experiments/.venv/bin/python -m unittest discover -s experiments/tests -v
experiments/.venv/bin/python experiments/run.py reproduce-v2
experiments/.venv/bin/python experiments/run.py evaluate --role development
# Freeze the candidate before reading this diagnostic evaluation:
experiments/.venv/bin/python experiments/run.py evaluate --role historical_check
# Optional historical fold reconciliation, not additional independent evidence:
experiments/.venv/bin/python experiments/run.py evaluate --role legacy
# Only after reviewing evidence; does not submit to Kaggle:
experiments/.venv/bin/python experiments/run.py submit --model regularized_graph
```

CLI paths resolve relative to the script, so it can run from another directory.
`--data-dir` supports another input directory. `--config` supports a copied JSON
configuration. `--run-id` is optional; existing run directories are never overwritten.
An incomplete run has no `COMPLETE` marker. Never infer success from a CSV alone.

## Structure

- `pipeline.py`: input validation, V7 graph features, temporal splitting, model fitting.
- `run.py`: CLI and immutable run artifacts, with source/data/version provenance.
- `configs/initial.json`: frozen baselines, candidate, folds and parameters.
- `runs/`: experiment tracker; one directory per invocation, including failures.
- `Experimental_Pipeline.ipynb`: small companion for running the CLI and reviewing results.
- `AUDIT.md`: evidence, limitations and experiment rationale.
- `tests/test_pipeline.py`: graph direction/coverage, label exclusion and submission safeguards.
- `requirements.txt`: exact environment used for this investigation.

To add an experiment, copy a configuration, add a named model, and run it. Keep
model logic in Python modules. Do not append another training workflow to the old notebook.
Training excludes unknown labels and IDs/time step as predictors. The seed is fixed;
thread count is four to keep the local machine usable. Library versions and platform
can affect exact numerical reproduction.

## Validation contract

Primary windows are 14-step rolling origins: train 1–7 / validate 8–21 for development,
and train 1–21 / evaluate 22–35 for a later historical check. These validation windows
are disjoint. The early training sample is smaller than final training, so this is
an informative stress test, not a perfect estimate of final performance. The legacy
train 1–14 / validate 15–28 window is optional and overlaps both primary windows;
do not average it as independent evidence.

All known historical periods have already informed previous analysis. No honest,
untouched historical holdout remains. Freeze configurations before the later check;
do not repeatedly tune against it. Do not early-stop on the reported evaluation set.
Retain per-step class counts, ROC-AUC and AP alongside pooled metrics. Undefined
single-class AUCs remain missing. Per-step variation is descriptive uncertainty,
not a confidence interval for private leaderboard performance. No automated winner
selection or tuned ensemble weights are included.

## Graph assumptions and rules

Exact V7 reproduction assumes the full provided edge list and same-step source
covariates are available. It uses no neighbour labels. Degree includes all incoming
edges, including sources absent from feature tables; mean/std use available sources.
Sample std uses ddof=1; singleton/no-neighbour aggregates remain NaN. Counts fill zero.
A runtime guard rejects inspectable cross-step edges, requiring an explicit redesign
before any different graph can be used. Unavailable source time steps remain unknown.

The subsequently supplied competition description explicitly allows engineered
features and graph models using the provided edge list. See `DATA_CONTEXT.md`.
This does not establish permission for external datasets or external labels.
External datasets, external labels and pretrained representations are not used.
The 94/71 feature semantics remain unverified and are not assumed by this candidate.

## Notebook verification

The companion executed successfully top-to-bottom in the isolated environment,
reviewing saved completed runs (`RERUN=False`). Its schema, execution counts, table
outputs and original-file preservation check were inspected programmatically.
Rendered appearance was not inspected in a notebook viewer. To inspect locally,
open `Experimental_Pipeline.ipynb` in Jupyter or VS Code and select the interpreter
`experiments/.venv/bin/python`. The CLI model fits were executed separately; the
notebook's opt-in `RERUN=True` branch invokes those same tested commands.

## Submission diagnostics

`Submission_Diagnostics.ipynb` reviews all three scored submissions and the new
fixed graph-only candidate. `SUBMISSION_FINDINGS.md` records the decision and evidence.
The original Python 3.10 environment reproduces all three submissions exactly.
`diagnose_submissions.py` reconstructs them and saves historical predictions;
`summarize_submissions.py` computes period diagnostics, paired time-step bootstrap
intervals and figures. Use `--run-id` for a new diagnostic run; the summary script
currently reads the fixed `submission_diagnosis` run. The graph ensemble test is
a single predeclared experiment recorded in `configs/graph_bagging.json`; it failed
its advancement rule and produced no candidate submission.

## Ready-to-upload next candidate

See `NEXT_SUBMISSION.md` and `submissions/submission_next_step_balanced_graph.csv`.
The candidate is 75% exact V2 plus 25% an equal-time-step-weighted V7 graph model.
`prepare_next_submission.py` and its configuration retain the full training and
selection procedure. Four candidate families were screened; the delivered blend
passed the frozen criteria. Actual Kaggle performance remains unmeasured.
