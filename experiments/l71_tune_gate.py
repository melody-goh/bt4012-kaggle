"""Time-proxy removal + last71 model settings, vs reference V2+last71 (public 0.95083). Cutoffs 7,10,14,17,21. Train labels only.
Candidates (6, declared before running):
  G1 V2 + l71_notime | G2 V2_notime + l71_notime | H1 V2 + l71(15 leaves, 600 trees, lr .02, colsample .6, min_child 40, lambda 5)
  H2 V2 + l71(is_unbalance) | H3 V2 + l71(63 leaves, min_child 10) | H4 V2 + l71(extra_trees)
FROZEN RULE (6 candidates): mean gain >= +0.002 AND better on >= 4 of 5 cutoffs. File only for the single best passer.
"""
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission

ROOT = Path(__file__).resolve().parent.parent
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
TIME = ["feat_101", "feat_103", "feat_136"]
L71 = BASE[94:]; L71nt = [f for f in L71 if f not in TIME]
print("time proxies inside last71:", [f for f in TIME if f in L71])

def fp(tr, V, cols, **kw):
    m = LGBMClassifier(**{**P, **kw}).fit(tr[cols], tr.label.astype(int))
    return L(m.predict_proba(V[cols])[:, list(m.classes_).index(1)])

train, test, edges, sample = load_data(ROOT)
train, test, g, _ = graph_features(train, test, edges)
lab = train.loc[train.label.notna()]
V2c = BASE + g; V2nt = [f for f in V2c if f not in TIME]

def run(tr, V):
    v2, l71 = fp(tr, V, V2c), fp(tr, V, L71)
    l71nt = fp(tr, V, L71nt)
    return {"ref": (v2 + l71) / 2, "G1": (v2 + l71nt) / 2, "G2": (fp(tr, V, V2nt) + l71nt) / 2,
            "H1": (v2 + fp(tr, V, L71, num_leaves=15, n_estimators=600, learning_rate=.02, colsample_bytree=.6, min_child_samples=40, reg_lambda=5.)) / 2,
            "H2": (v2 + fp(tr, V, L71, is_unbalance=True)) / 2,
            "H3": (v2 + fp(tr, V, L71, num_leaves=63, min_child_samples=10)) / 2,
            "H4": (v2 + fp(tr, V, L71, extra_trees=True, n_estimators=500)) / 2}

rows = []
for c in [7, 10, 14, 17, 21]:
    tr = lab.loc[lab.time_step <= c]; va = lab.loc[lab.time_step.between(c + 1, min(c + 14, 35))]
    y = va.label.astype(int).to_numpy()
    rows.append({"cutoff": c, **{k: roc_auc_score(y, v) for k, v in run(tr, va).items()}}); print("done", c, flush=True)
R = pd.DataFrame(rows).set_index("cutoff"); print("\n", R.round(4).to_string())
passers = {}
for k in ["G1", "G2", "H1", "H2", "H3", "H4"]:
    d = R[k] - R.ref; ok = d.mean() >= 0.002 and (d > 0).sum() >= 4
    print(f"{k}: mean gain {d.mean():+.4f} | better on {(d > 0).sum()}/5 | PASS {ok}")
    if ok: passers[k] = d.mean()
if passers:
    best = max(passers, key=passers.get); print("Best passer:", best, flush=True)
    out = pd.DataFrame({"index": test["index"], "target": expit(run(lab, test)[best])}); validate_submission(out, test, sample)
    p = ROOT / f"submission_{best}_l71tune.csv"; out.to_csv(p, index=False); print("saved", p)
    print("Spearman vs v10 (0.95083):", round(out.target.corr(pd.read_csv(ROOT / "submission_v10.csv").target, method="spearman"), 4))
else:
    print("No candidate passes: do not submit.")
