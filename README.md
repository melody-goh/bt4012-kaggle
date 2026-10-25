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

Python 3.13:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Reproduce the two final submissions

```sh
python experiments/reproduce_final.py
```

This trains every model from `data/` and writes:

- `outputs/submission_v20.csv`
- `outputs/submission_v22.csv`

Options: `--data-dir` (default `data/`), `--out-dir` (default `outputs/`), `--check` (compare with the submitted files in the repo root). Runtime is a few minutes on a laptop CPU.

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

The script is deterministic: repeated runs give identical files. Its output matches the submitted files' ranking with Spearman correlation 0.9991 but not bit-for-bit, because some components were originally generated in a different computing environment and LightGBM's floating-point results differ slightly across platforms. The submitted files are kept in the repo root for reference.

## 6. Repository layout

- `experiments/reproduce_final.py`: end-to-end training and inference for the final submissions.
- `experiments/pipeline.py`: data loading, graph features, submission checks.
- `experiments/SUBMISSION_LOG.md`: every submission, its rationale, the pre-declared validation rules and public scores.
- `experiments/*_gate.py`, `*_build.py`: the experiments described in the log.
- `submission_*.csv`: all submitted and candidate files.
