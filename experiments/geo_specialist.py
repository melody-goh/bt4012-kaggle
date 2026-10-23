"""Specialist on label-free geometry features only (f1,f2,f3 + in_degree), blended with ref and K1.
Candidates (declared before running): B1=(v2+l71+geo)/3 ; B2=(v2k+l71k+geo)/3 ; B3=(v2+l71+geo*2)/4 (geo weight .5)
RULE: mean gain over ref >= +0.002 AND better on >= 4 of 6 cutoffs.
"""
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features
HERE = Path(".").resolve(); ROOT = HERE.parent
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
PG = dict(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=50, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
lab = train[train.label.notna()]; FC = ["f1_dens_step", "f2_d_unk", "f3_d_labpast"]
def fp(X, y, V, cols, params=P):
    m = LGBMClassifier(**params).fit(X[cols], y); return L(m.predict_proba(V[cols])[:, list(m.classes_).index(1)])
rows = []
for c in [7, 10, 14, 17, 21, 24]:
    F = pd.read_pickle(HERE / f"knnfeat_c{c}.pkl"); d = lab.loc[F.index].join(F); tr = d[d.time_step <= c]; va = d[d.time_step > c]
    y = tr.label.astype(int); yv = va.label.astype(int).to_numpy()
    v2, l71, v2k, l71k = fp(tr, y, va, BASE + g), fp(tr, y, va, L71), fp(tr, y, va, BASE + g + FC), fp(tr, y, va, L71 + FC)
    geo = fp(tr, y, va, FC + ["in_degree"], PG)
    fs = va.time_step.isin([24, 29, 31, 32]).to_numpy()
    r = dict(cutoff=c, geo_alone=roc_auc_score(yv, geo), ref=roc_auc_score(yv, (v2 + l71) / 2), B1=roc_auc_score(yv, (v2 + l71 + geo) / 3),
             B2=roc_auc_score(yv, (v2k + l71k + geo) / 3), B3=roc_auc_score(yv, (v2 + l71 + .5 * geo) / 2.5), K1=roc_auc_score(yv, (v2k + l71k) / 2))
    rows.append(r)
R = pd.DataFrame(rows).set_index("cutoff"); print(R.round(4).to_string())
for k in ["K1", "B1", "B2", "B3"]:
    dd = R[k] - R.ref; print(f"{k}: mean gain {dd.mean():+.4f} | better on {(dd > 0).sum()}/6 | per-cutoff {dd.round(4).tolist()}")
