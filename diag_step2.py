"""Step 2: two untested robustness ideas, evaluated on 5 rolling cutoffs. Run next to train/test/edges CSVs.

Reference = V2-equivalent (165 feats + 18 full-graph V7 feats, 300 trees, lr .03, 31 leaves).

Idea 1  STEP-BAGGING: each model is trained with ~30% of training STEPS removed; average many.
        Forces patterns that hold across steps; models that rely on one step's family get outvoted.
Idea 2  CONSISTENT FEATURES: keep a feature only if its univariate illicit-vs-licit direction agrees
        across >=80% of training steps (computed inside the training window only).
Idea 3  both together.

Frozen rule, set before running: submit-worthy only if mean gain >= +0.005 AND better than reference
on >=4 of 5 cutoffs. Only train labels used.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
edges = pd.read_csv("txs_edgelist.csv")[["txId1", "txId2"]]
F = [f"feat_{i}" for i in range(1, 166)]
AGG = ["feat_2", "feat_53", "feat_58", "feat_16", "feat_90", "feat_138", "feat_142", "feat_163"]
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, verbosity=-1, n_jobs=4)
K_BAGS, DROP = 20, 0.30

lab = train[train.label.notna()].copy().reset_index(drop=True)
lab["label"] = lab.label.astype(int)

nodes = pd.concat([train[["txId"] + AGG], test[["txId"] + AGG]], ignore_index=True)
src = nodes.rename(columns={"txId": "txId1", **{c: "s_" + c for c in AGG}})
j = edges.merge(src, on="txId1", how="left")
sc = ["s_" + c for c in AGG]
grp = j.groupby("txId2")
agg = grp[sc].agg(["mean", "std"])
agg.columns = [f"in_{a}_{b}" for a, b in agg.columns]
agg["in_degree"] = grp.size()
agg["in_known_sources"] = grp[sc[0]].count()
GC = list(agg.columns)
D = lab.merge(agg, left_on="txId", right_index=True, how="left")
D[["in_degree", "in_known_sources"]] = D[["in_degree", "in_known_sources"]].fillna(0)
assert len(D) == len(lab)
ALL = F + GC

def consistent_features(tr, thr=0.8):
    signs = {f: [] for f in F}
    for s, g in tr.groupby("time_step"):
        if g.label.sum() < 5 or (1 - g.label).sum() < 20:
            continue
        for f in F:
            signs[f].append(np.sign(roc_auc_score(g.label, g[f]) - 0.5))
    keep = []
    for f in F:
        a = np.array(signs[f])
        if len(a) >= 3 and max((a > 0).mean(), (a < 0).mean()) >= thr:
            keep.append(f)
    return keep

def fit_ref(tr, cols, seed=42):
    return lgb.LGBMClassifier(random_state=seed, **P).fit(tr[cols], tr.label)

def predict_bag(tr, va, cols):
    steps = np.array(sorted(tr.time_step.unique()))
    out = np.zeros(len(va))
    for k in range(K_BAGS):
        rng = np.random.RandomState(k)
        drop = rng.choice(steps, int(len(steps) * DROP), replace=False)
        sub = tr[~tr.time_step.isin(drop)]
        if sub.label.nunique() < 2:
            sub = tr
        out += fit_ref(sub, cols, seed=k).predict_proba(va[cols])[:, 1]
    return out / K_BAGS

rows = []
for c in [7, 10, 14, 17, 21]:
    tr = D[D.time_step <= c]
    va = D[D.time_step.between(c + 1, min(c + 14, 35))]
    if va.label.nunique() < 2:
        continue
    keep = consistent_features(tr)
    cols2 = keep + GC
    ref = fit_ref(tr, ALL).predict_proba(va[ALL])[:, 1]
    r = dict(cutoff=c, n_consistent=len(keep),
             reference=roc_auc_score(va.label, ref),
             step_bagging=roc_auc_score(va.label, predict_bag(tr, va, ALL)),
             consistent_feats=roc_auc_score(va.label, fit_ref(tr, cols2).predict_proba(va[cols2])[:, 1]),
             both=roc_auc_score(va.label, predict_bag(tr, va, cols2)))
    rows.append(r)
    print(r, flush=True)

r = pd.DataFrame(rows).set_index("cutoff")
print("\n", r.round(4))
for m in ["step_bagging", "consistent_feats", "both"]:
    d = r[m] - r.reference
    ok = (d.mean() >= 0.005) and ((d > 0).sum() >= 4)
    print(f"{m}: mean gain {d.mean():+.4f} | better on {(d > 0).sum()}/{len(d)} | passes frozen rule: {ok}")