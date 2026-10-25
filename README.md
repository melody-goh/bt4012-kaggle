# BT4012 Elliptic Bitcoin fraud detection (Kaggle)

Predicts illicit Bitcoin transactions in future time steps (36-49). Metric: ROC-AUC.
Final selected submissions: `submission_v20.csv` (public 0.95400) and `submission_v22.csv` (public 0.95403).

## 1. Data

The dataset is not included. Download the competition files and place them in `data/` at the repo root:

```
data/
  train.csv
  test.csv
  txs_edgelist.csv
  sample_submission.csv
```

## 2. Setup

The final submissions were built in two stages with two Python versions (the library versions change LightGBM's results slightly, so each stage must use its original environment):

```sh
python3.10 -m venv .venv310 && .venv310/bin/pip install -r requirements-py310.txt
python3.13 -m venv .venv313 && .venv313/bin/pip install -r requirements.txt
```

## 3. Reproduce the two final submissions

```sh
./reproduce.sh
```

(Or point to other interpreters: `PY310=/path/to/python3.10 PY313=/path/to/python3.13 ./reproduce.sh`.)

- Stage 1 (`experiments/reproduce_stage1.py`, Python 3.10) trains V2, V3, D and X2 and writes them to `outputs/components/`.
- Stage 2 (`experiments/reproduce_stage2.py`, Python 3.13) builds v7, trains R15, blends, and writes `outputs/submission_v20.csv` and `outputs/submission_v22.csv`. It then compares them with the submitted files in `submissions/submitted/`.

Options for both scripts: `--data-dir` (default `data/`), `--out-dir` (default `outputs/`). Runtime is a few minutes on a laptop CPU.

## 4. What the models are

All models are LightGBM (300 trees, learning rate 0.03, 31 leaves, seed 42), trained on labelled rows of steps 1-35 only. "Graph features" are the mean and standard deviation of 8 key features over each transaction's incoming neighbours, plus in-degree and the number of visible sources (`experiments/pipeline.py`, `graph_features`).

| Part | Definition |
|---|---|
| V2 | all 165 features + graph features |
| V3 | 0.25 first-94-features model + 0.25 last-71-features model + 0.5 recency-weighted V2 |
| v7 | logit average of V2 and V3 |
| D (v12) | logit average of recency-weighted V2 (half-life 28 steps) and the last-71-features model |
| R20, R15 | the D recipe trained only on steps >= 20 and >= 15 |
| **v20** | logit average of D, R20, R15 and v7 |
| **v22** | v20 with the logit raised by 0.3 for test steps 43-49 |

## 5. Note on exact reproduction

The regenerated files have the same ranking as the submitted files, so they give the same AUC. Their values match to about 15 decimal places: 3 of 15,329 rows differ by about 1e-16, the floating-point rounding limit. The original build scripts show the same last-digit differences when they are rerun in their original environment, so this is floating-point noise, not a change in the model.

## 6. Repository layout

- `reproduce.sh`, `experiments/reproduce_stage1.py`, `experiments/reproduce_stage2.py`: end-to-end training and inference for the final submissions.
- `experiments/pipeline.py`: data loading, graph features, submission checks.
- `experiments/SUBMISSION_LOG.md`: every submission, its rationale, the pre-declared validation rules and public scores.
- `experiments/*_gate.py`, `*_build.py`: the experiments described in the log.
- `submissions/submitted/`: every file submitted to Kaggle (`BEST_0.95044.csv` is an identical copy of `submission_v7.csv`). The two final selections are `submission_v20.csv` and `submission_v22.csv`.
- `submissions/not_submitted/`: candidates that were built but not submitted (see the log for why).

The experiment scripts were written when all submission files sat in the repo root. To re-run them, copy the files back first: `cp submissions/*/*.csv .`
