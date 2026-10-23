from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
ROOT = Path(".").resolve().parent; L71 = BASE[94:]; L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return L(m.predict_proba(V[cols])[:, 1])
def nov(fitrows, V):
    X = fitrows[L71].fillna(0); f = IsolationForest(n_estimators=300, max_samples=512, random_state=0, n_jobs=4).fit(X)
    s_tr = -f.score_samples(X); return (-f.score_samples(V[L71].fillna(0)) - s_tr.mean()) / s_tr.std()
rows = []
for c in [7, 10, 14, 17, 21, 24]:
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    w = 2. ** (-(c - tr.time_step.to_numpy()) / 28); w /= w.mean(); D = (fp(tr, va, BASE + g, w) + fp(tr, va, L71)) / 2
    z = nov(train[train.time_step <= c], va)
    rows.append({"cutoff": c, "D": roc_auc_score(y, D), "N3": roc_auc_score(y, D - .25 * z), "N4": roc_auc_score(y, D - .5 * z), "nov_alone": roc_auc_score(y, z)}); print("done", c, flush=True)
R = pd.DataFrame(rows).set_index("cutoff"); print(R.round(4).to_string()); passers = {}
v12 = pd.read_csv(ROOT / "submission_v12.csv"); zt = nov(train, test); st = test.time_step.to_numpy()
top_share = lambda s: float((st[s >= np.quantile(s, .95)] >= 43).mean()); base_share = top_share(v12.target.to_numpy())
for k, a in [("N3", -.25), ("N4", -.5)]:
    d = R[k] - R.D; g1 = d.mean() >= .001 and (d > 0).sum() >= 4 and d[17] > 0 and d[21] > 0
    s = L(v12.target.to_numpy()) + a * zt; sh = top_share(s); g2 = sh <= base_share * 1.25 + 1e-9
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d[17]:+.4f} 21:{d[21]:+.4f} | GATE1 {g1} | top5% share in steps 43-49: {sh:.3f} vs v12 {base_share:.3f} | GATE2 {g2}")
    if g1 and g2:
        out = pd.DataFrame({"index": test["index"], "target": expit(s)}); validate_submission(out, test, sample)
        out.to_csv(ROOT / f"submission_{k}_novelty.csv", index=False); print("saved", k, "| Spearman vs v12:", round(out.target.corr(v12.target, method="spearman"), 4))
print("test novelty z by step (mean):", pd.Series(zt).groupby(st).mean().round(2).to_dict())
