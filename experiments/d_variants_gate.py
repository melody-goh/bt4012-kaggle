"""Variants of D=(weighted_hl28 + l71)/2 (public 0.95277). Cutoffs 7,10,14,17,21,24. Train labels only.
E1 (weighted_hl14 + l71)/2 | E2 (weighted_hl56 + l71)/2 | E3 (weighted_hl28 + l71w_hl28)/2 | E4 (weighted_hl28 + l71 + v2)/3
RULE: mean gain over D >= +0.001 AND better on >= 4 of 6 cutoffs. File only for the single best passer (trained at cutoff 35).
"""
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
ROOT = Path(".").resolve().parent
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()]
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return L(m.predict_proba(V[cols])[:, 1])
def W(tr, c, hl): w = 2. ** (-(c - tr.time_step.to_numpy()) / hl); return w / w.mean()
def cands(tr, V, c):
    w28 = fp(tr, V, BASE + g, W(tr, c, 28)); l71 = fp(tr, V, L71)
    return {"D": (w28 + l71) / 2, "E1": (fp(tr, V, BASE + g, W(tr, c, 14)) + l71) / 2, "E2": (fp(tr, V, BASE + g, W(tr, c, 56)) + l71) / 2,
            "E3": (w28 + fp(tr, V, L71, W(tr, c, 28))) / 2, "E4": (w28 + l71 + fp(tr, V, BASE + g)) / 3}
rows = []
for c in [7, 10, 14, 17, 21, 24]:
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    rows.append({"cutoff": c, **{k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}}); print("done", c, flush=True)
R = pd.DataFrame(rows).set_index("cutoff"); print(R.round(4).to_string()); passers = {}
for k in ["E1", "E2", "E3", "E4"]:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4; print(f"{k}: mean gain {d.mean():+.4f} | better on {(d > 0).sum()}/6 | PASS {ok} | {d.round(4).tolist()}")
    if ok: passers[k] = d.mean()
if passers:
    b = max(passers, key=passers.get); out = pd.DataFrame({"index": test["index"], "target": expit(cands(lab, test, 35)[b])}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{b}_dvariant.csv", index=False); print("saved", b, "| Spearman vs v12 (D):", round(out.target.corr(pd.read_csv(ROOT / "submission_v12.csv").target, method="spearman"), 4))
else: print("No candidate passes: keep v12.")
