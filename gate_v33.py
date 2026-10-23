"""Offline gate before spending a scarce submission: does swapping V2 for V33 (or adding it) help the V3 blend?
Run next to train.csv, test.csv, txs_edgelist.csv. Needs the saved diagnosis predictions from your experiments folder:
  python gate_v33.py experiments/runs/submission_diagnosis
Trains V33 (800 trees, lr .015, 31 leaves, 165 + 18 graph feats, seed 42) at cutoffs 7, 14, 21 on labelled rows,
merges with saved V2/V3 predictions by txId (same validation rows: steps c+1..c+14), compares logit blends.

FROZEN RULE: submit a V33 blend only if it beats the V2+V3 blend on >= 2 of 3 cutoffs AND mean gain >= +0.001.
Otherwise do NOT submit a V33 blend; submit submission_blend_v2v3_60_40.csv or skip.
Only train labels used.
"""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit
from sklearn.metrics import roc_auc_score

diag = sys.argv[1] if len(sys.argv) > 1 else "experiments/runs/submission_diagnosis"
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
edges = pd.read_csv("txs_edgelist.csv")[["txId1", "txId2"]]
F = [f"feat_{i}" for i in range(1, 166)]
AGG = ["feat_2", "feat_53", "feat_58", "feat_16", "feat_90", "feat_138", "feat_142", "feat_163"]
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))

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

lab = train[train.label.notna()].copy().reset_index(drop=True)
lab["label"] = lab.label.astype(int)
D = lab.merge(agg, left_on="txId", right_index=True, how="left")
D[["in_degree", "in_known_sources"]] = D[["in_degree", "in_known_sources"]].fillna(0)
assert len(D) == len(lab)

rows = []
for c in [7, 14, 21]:
    tr = D[D.time_step <= c]
    va = D[D.time_step.between(c + 1, min(c + 14, 35))]
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.015, num_leaves=31, random_state=42,
                           verbosity=-1, n_jobs=4).fit(tr[F + GC], tr.label)
    va = va.assign(v33=m.predict_proba(va[F + GC])[:, 1])
    s = pd.read_csv(f"{diag}/predictions_cutoff_{c}.csv")[["txId", "v2", "v3"]]
    d = va[["txId", "label", "v33"]].merge(s, on="txId", how="inner")
    assert len(d) == len(va), f"row mismatch at cutoff {c}: {len(d)} vs {len(va)}"
    y = d.label.values
    z2, z3, z33 = L(d.v2.values), L(d.v3.values), L(d.v33.values)
    rows.append(dict(cutoff=c,
                     v33_alone=roc_auc_score(y, d.v33), v2_alone=roc_auc_score(y, d.v2),
                     v2_v3=roc_auc_score(y, (z2 + z3) / 2),
                     v33_v3=roc_auc_score(y, (z33 + z3) / 2),
                     v2_v3_v33=roc_auc_score(y, (z2 + z3 + z33) / 3)))
    print("done cutoff", c, flush=True)

R = pd.DataFrame(rows).set_index("cutoff")
print("\n", R.round(4).to_string())
print("\n=== frozen rule vs v2_v3 ===")
for k in ["v33_v3", "v2_v3_v33"]:
    g = R[k] - R.v2_v3
    ok = (g > 0).sum() >= 2 and g.mean() >= 0.001
    print(f"{k}: mean gain {g.mean():+.4f} | better on {(g > 0).sum()}/3 | PASS: {ok}")