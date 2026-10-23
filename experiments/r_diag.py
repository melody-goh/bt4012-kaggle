from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features
ROOT = Path(".").resolve().parent; L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]; RK = [c + "_rk" for c in L71]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
lab[RK] = lab.groupby("time_step")[L71].rank(pct=True).to_numpy()
# test: per-step mean logit of v12 vs R2
v12 = pd.read_csv(ROOT / "submission_v12.csv"); r2 = pd.read_csv(ROOT / "submission_R2_rank.csv"); t = test[["index", "time_step"]].copy()
t["v12"] = L(v12.target.to_numpy()); t["r2"] = L(r2.target.to_numpy())
s = t.groupby("time_step").agg(n=("v12", "size"), v12=("v12", "mean"), r2=("r2", "mean")); print("TEST per-step mean logit\n", s.round(2).to_string())
print("std of step means  v12 %.2f  R2 %.2f" % (s.v12.std(), s.r2.std()))
# offline cutoff 21: pooled vs within-step AUC and step-mean spread
c = 21; tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, 35)].copy(); y = va.label.astype(int).to_numpy()
w = 2. ** (-(c - tr.time_step.to_numpy()) / 28); w /= w.mean()
f = lambda cols, ww=None: L(LGBMClassifier(**P).fit(tr[cols], tr.label.astype(int), sample_weight=ww).predict_proba(va[cols])[:, 1])
va["D"] = (f(BASE + g, w) + f(L71)) / 2; va["rk"] = f(RK)
va["R2"] = (f(BASE + g, w) + f(L71) + va["rk"]) * 0 + (va["D"] * 2 + va["rk"]) / 3
for k in ["D", "rk", "R2"]:
    within = np.mean([roc_auc_score(d.label.astype(int), d[k]) for _, d in va.groupby("time_step") if d.label.nunique() == 2])
    print(f"{k}: pooled {roc_auc_score(y, va[k]):.4f} | mean within-step {within:.4f}")
print("val illicit rate by step:", va.groupby("time_step").label.apply(lambda x: round(x.astype(int).mean(), 3)).to_dict())
