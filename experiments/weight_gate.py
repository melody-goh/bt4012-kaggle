from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features
ROOT = Path(".").resolve().parent
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()]
def fp(X, V, cols):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int)); return L(m.predict_proba(V[cols])[:, 1])
rows = []
for c in [7, 10, 14, 17, 21, 24]:
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    v2, l71 = fp(tr, va, BASE + g), fp(tr, va, L71)
    r = {"cutoff": c, "w50_ref": roc_auc_score(y, .5 * v2 + .5 * l71)}
    for w in (.65, .8, 1.0): r[f"w{int(w*100)}"] = roc_auc_score(y, (1 - w) * v2 + w * l71)
    rows.append(r); print("done", c, flush=True)
R = pd.DataFrame(rows).set_index("cutoff"); print(R.round(4).to_string())
for k in ["w65", "w80", "w100"]:
    d = R[k] - R.w50_ref; print(f"{k}: mean gain {d.mean():+.4f} | better on {(d > 0).sum()}/6 | per-cutoff {d.round(4).tolist()}")
