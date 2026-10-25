"""Rebuild the two final-selected submissions from the raw competition data.

    python experiments/reproduce_final.py                  # reads data/, writes outputs/
    python experiments/reproduce_final.py --data-dir data --out-dir outputs --check

Writes outputs/submission_v20.csv and outputs/submission_v22.csv.
--check compares them with the submitted files in the repo root.

Model: LightGBM(300 trees, lr .03, 31 leaves, seed 42), labelled training rows only.
  V2  = BASE + 18 upstream graph features, unweighted
  V3  = .25 first94 + .25 last71 + .5 recency-weighted V2 (half-life 28), probability average
  v7  = logit avg(V2, V3)
  D   = logit avg(recency-weighted V2, last71)                      (= submission_v12)
  Rk  = D recipe trained only on steps >= k (R20 = submission_v16_new, R15)
  v20 = logit avg(D, R20, R15, v7)
  v22 = v20 with logit + 0.3 for test steps 43-49
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from pipeline import BASE, load_data, graph_features, validate_submission

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=ROOT / "data", type=Path)
ap.add_argument("--out-dir", default=ROOT / "outputs", type=Path)
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); F94, L71 = BASE[:94], BASE[94:]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(args.data_dir); train, test, g, _ = graph_features(train, test, edges)
lab = train[train.label.notna()]
def prob(X, cols, w=None): return LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w).predict_proba(test[cols])[:, 1]
def W(X, hl): w = 2. ** (-(35 - X.time_step.to_numpy()) / hl); return w / w.mean()
def D_recipe(X): return expit((L(prob(X, BASE + g, W(X, 28))) + L(prob(X, L71))) / 2)

v2 = prob(lab, BASE + g)
wg = prob(lab, BASE + g, W(lab, 28))
v3 = .25 * prob(lab, F94) + .25 * prob(lab, L71) + .5 * wg
v7 = expit((L(v2) + L(v3)) / 2)
d = expit((L(wg) + L(prob(lab, L71))) / 2)
r20, r15 = (D_recipe(lab[lab.time_step >= k]) for k in (20, 15))
z20 = (L(d) + L(r20) + L(r15) + L(v7)) / 4
late = test.time_step.to_numpy() >= 43
outs = {"submission_v20.csv": expit(z20), "submission_v22.csv": expit(z20 + .3 * late)}

args.out_dir.mkdir(parents=True, exist_ok=True)
for name, p in outs.items():
    out = pd.DataFrame({"index": test["index"], "target": p}); validate_submission(out, test, sample)
    out.to_csv(args.out_dir / name, index=False); print("saved", args.out_dir / name)
    if args.check and (ROOT / name).exists():
        ref = pd.read_csv(ROOT / name).target
        print(f"  vs submitted {name}: Spearman {out.target.corr(ref, method='spearman'):.6f}, max |diff| {np.abs(out.target - ref).max():.2e}")
