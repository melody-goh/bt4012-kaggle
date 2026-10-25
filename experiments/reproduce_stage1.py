"""Stage 1 of 2 (run with Python 3.10, requirements-py310.txt): models originally built in that environment.

Writes to <out-dir>/components/: v2.csv, v3.csv, v12.csv (D), v16_new.csv (X2 = D recipe on steps >= 20).
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
args = ap.parse_args()

L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); F94, L71 = BASE[:94], BASE[94:]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(args.data_dir); train, test, g, _ = graph_features(train, test, edges)
lab = train[train.label.notna()]
def prob(X, cols, w=None): return LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w).predict_proba(test[cols])[:, 1]
def W(X, hl): w = 2. ** (-(35 - X.time_step.to_numpy()) / hl); return w / w.mean()
def D_recipe(X): return expit((L(prob(X, BASE + g, W(X, 28))) + L(prob(X, L71))) / 2)

wg, l71 = prob(lab, BASE + g, W(lab, 28)), prob(lab, L71)
outs = {"v2": prob(lab, BASE + g),                                      # V2: unweighted graph model
        "v3": .25 * prob(lab, F94) + .25 * l71 + .5 * wg,               # V3: group specialists + weighted graph
        "v12": expit((L(wg) + L(l71)) / 2),                             # D
        "v16_new": D_recipe(lab[lab.time_step >= 20])}                  # X2

comp = args.out_dir / "components"; comp.mkdir(parents=True, exist_ok=True)
for name, p in outs.items():
    out = pd.DataFrame({"index": test["index"], "target": p}); validate_submission(out, test, sample)
    out.to_csv(comp / f"{name}.csv", index=False); print("saved", comp / f"{name}.csv")
