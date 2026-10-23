from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features
ROOT = Path(".").resolve().parent; L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()]
def fp(X, V, cols, w=None): return L(LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w).predict_proba(V[cols])[:, 1])
def W(X, c, hl): w = 2. ** (-(c - X.time_step.to_numpy()) / hl); return w / w.mean()
rows = []
for c in [14, 17, 21, 24]:
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int); l71 = fp(tr, va, L71); rc = tr[tr.time_step >= c - 15]
    rows.append({"cutoff": c, "D": roc_auc_score(y, (fp(tr, va, BASE + g, W(tr, c, 28)) + l71) / 2), "X1": roc_auc_score(y, (fp(tr, va, BASE + g, W(tr, c, 7)) + l71) / 2),
                 "X2": roc_auc_score(y, (fp(rc, va, BASE + g, W(rc, c, 28)) + fp(rc, va, L71)) / 2), "X3": roc_auc_score(y, (fp(tr, va, L71 + g, W(tr, c, 28)) + l71) / 2)})
print(pd.DataFrame(rows).set_index("cutoff").round(4).to_string())
