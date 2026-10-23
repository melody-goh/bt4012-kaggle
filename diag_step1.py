"""Step 1 diagnostics. Run from the folder with train.csv, test.csv, txs_edgelist.csv.

C: is test = labelled-only nodes? (labelled train rows with labelled-only sources should
   show visible fraction ~0.54 and ~25% zero-visible, like test.)
E: train/eval graph-feature mismatch. Compare, on 5 rolling cutoffs:
   base165 | full-graph V7 | test-faithful V7 | trained-full-evaluated-faithful (the damage).
D: mini self-training simulation, cutoff 21, window 22-35, pool = labelled rows with labels hidden
   (mimics test, where unknown nodes are absent). Seeds come from the model's OWN confident
   predictions. Hidden labels used only to report seed precision and AUC. No selection on them.
Only train labels are used. Nothing touches test labels.
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
PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, verbosity=-1, n_jobs=4, random_state=42)

lab = train[train.label.notna()].copy().reset_index(drop=True)
lab["label"] = lab.label.astype(int)
lab_ids = set(lab.txId)
test_ids = set(test.txId)
all_ids = set(train.txId) | test_ids

# ---------- Check C ----------
print("=== Check C: visibility if only labelled nodes are present ===")
e = edges.copy()
e["src_lab"] = e.txId1.isin(lab_ids)
e["src_test"] = e.txId1.isin(test_ids)
g = e.groupby("txId2").agg(deg=("txId1", "size"), lab=("src_lab", "sum"), tst=("src_test", "sum"))

def summ(ids, col, name):
    m = pd.DataFrame({"txId": list(ids)}).merge(g, left_on="txId", right_index=True, how="left").fillna(0)
    has = m.deg > 0
    zero = has & (m[col] == 0)
    frac = (m[col][has] / m.deg[has]).mean()
    print(f"{name}: in_deg>0 {has.mean():.3f} | in_deg>0 & zero visible {zero.mean():.3f} | mean visible frac {frac:.3f}")

summ(lab_ids, "lab", "labelled train, labelled-only sources")
summ(test_ids, "tst", "test, test-only sources (should be 0.247 / 0.539)")

# ---------- graph features ----------
def build_v7(allowed):
    nodes = pd.concat([train[["txId"] + AGG], test[["txId"] + AGG]], ignore_index=True)
    src = nodes[nodes.txId.isin(allowed)].rename(columns={"txId": "txId1", **{c: "s_" + c for c in AGG}})
    j = edges.merge(src, on="txId1", how="left")
    sc = ["s_" + c for c in AGG]
    grp = j.groupby("txId2")
    agg = grp[sc].agg(["mean", "std"])
    agg.columns = [f"in_{a}_{b}" for a, b in agg.columns]
    agg["in_degree"] = grp.size()
    agg["in_known_sources"] = grp[sc[0]].count()
    return agg

def attach(df, agg):
    x = df.merge(agg, left_on="txId", right_index=True, how="left")
    x[["in_degree", "in_known_sources"]] = x[["in_degree", "in_known_sources"]].fillna(0)
    return x

agg_full = build_v7(all_ids)
agg_faith = build_v7(lab_ids | test_ids)
GC = list(agg_full.columns)
full = attach(lab, agg_full)
faith = attach(lab, agg_faith)
assert len(full) == len(lab) == len(faith)

def fit(X, y):
    return lgb.LGBMClassifier(**PARAMS).fit(X, y)

# ---------- Check E ----------
print("\n=== Check E: pooled AUC, train c, validate c+1..c+14 ===")
rows = []
for c in [7, 10, 14, 17, 21]:
    tr = lab.time_step <= c
    va = lab.time_step.between(c + 1, min(c + 14, 35))
    y_tr, y_va = lab.label[tr], lab.label[va]
    if y_va.nunique() < 2:
        continue
    m_base = fit(lab.loc[tr, F], y_tr)
    m_full = fit(full.loc[tr, F + GC], y_tr)
    m_faith = fit(faith.loc[tr, F + GC], y_tr)
    rows.append(dict(
        cutoff=c,
        base165=roc_auc_score(y_va, m_base.predict_proba(lab.loc[va, F])[:, 1]),
        full_train_full_eval=roc_auc_score(y_va, m_full.predict_proba(full.loc[va, F + GC])[:, 1]),
        full_train_FAITH_eval=roc_auc_score(y_va, m_full.predict_proba(faith.loc[va, F + GC])[:, 1]),
        faith_train_faith_eval=roc_auc_score(y_va, m_faith.predict_proba(faith.loc[va, F + GC])[:, 1]),
    ))
r = pd.DataFrame(rows).set_index("cutoff")
print(r.round(4))
print("mean:\n", r.mean().round(4))
print("READ: compare base165 vs faith_train_faith_eval (honest graph value on test-like data),")
print("      and full_train_full_eval vs full_train_FAITH_eval (offline optimism).")

# ---------- Check D ----------
print("\n=== Check D: mini self-training, cutoff 21, pool = steps 22-35 labelled rows, labels hidden ===")
tr = lab[lab.time_step <= 21]
pool = lab[lab.time_step.between(22, 35)].reset_index(drop=True)
POS_T, NEG_T = 0.8, 0.01  # frozen before running
FAIL = [24, 29, 31, 32]
p0 = fit(tr[F], tr.label).predict_proba(pool[F])[:, 1]
p = p0.copy()
for rnd in (1, 2):
    pos = np.where(p >= POS_T)[0]
    neg = np.where(p <= NEG_T)[0]
    neg = neg[np.argsort(p[neg])][: 3 * max(len(pos), 1)]
    prec_pos = pool.label.values[pos].mean() if len(pos) else float("nan")
    prec_neg = (1 - pool.label.values[neg]).mean() if len(neg) else float("nan")
    seed_idx = np.concatenate([pos, neg])
    X = pd.concat([tr[F], pool.loc[seed_idx, F]])
    y = np.concatenate([tr.label.values, np.ones(len(pos), int), np.zeros(len(neg), int)])
    p = fit(X, y).predict_proba(pool[F])[:, 1]
    print(f"round {rnd}: seeds pos {len(pos)} (precision {prec_pos:.3f}), neg {len(neg)} (precision {prec_neg:.3f})")
blend = 0.5 * p0 + 0.5 * p
y_all = pool.label.values
fail_m = pool.time_step.isin(FAIL).values
print(f"pooled AUC  base {roc_auc_score(y_all, p0):.4f} | self-trained {roc_auc_score(y_all, p):.4f} | blend {roc_auc_score(y_all, blend):.4f}")
print(f"failing-steps pooled AUC  base {roc_auc_score(y_all[fail_m], p0[fail_m]):.4f} | self-trained {roc_auc_score(y_all[fail_m], p[fail_m]):.4f} | blend {roc_auc_score(y_all[fail_m], blend[fail_m]):.4f}")
for s in FAIL:
    mm = (pool.time_step == s).values
    print(f"  step {s}: base {roc_auc_score(y_all[mm], p0[mm]):.3f} -> blend {roc_auc_score(y_all[mm], blend[mm]):.3f}")
print("READ: blend gain >= +0.02 pooled AND failing steps clearly up AND seed precision >= 0.9 -> build full version.")
print("      flat -> seeds from the model's own predictions can't reach the missed family; drop self-training.")