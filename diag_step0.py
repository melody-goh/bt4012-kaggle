"""Step 0 diagnostics. Run from the folder with train.csv, test.csv, txs_edgelist.csv.

Check A: is neighbour visibility different in train vs test?
Check B: oracle go/no-go. If a few labelled rows from a failing step fix that step,
         the missed illicit family is learnable and target-period self-training has headroom.
Uses offline validation labels only. Nothing here touches test labels.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
edges = pd.read_csv("txs_edgelist.csv")
F = [f"feat_{i}" for i in range(1, 166)]

# ---------- Check A: visibility ----------
present = set(train.txId) | set(test.txId)
e = edges.copy()
e["src_present"] = e.txId1.isin(present)
g = e.groupby("txId2").agg(in_deg=("txId1", "size"), known=("src_present", "sum"))

def vis(df, name):
    m = df[["txId"]].merge(g, left_on="txId", right_index=True, how="left").fillna(0)
    has_in = m.in_deg > 0
    hidden_all = has_in & (m.known == 0)
    print(f"{name}: n={len(df)} | in_degree>0: {has_in.mean():.3f} | "
          f"in_degree>0 but ZERO visible sources: {hidden_all.mean():.3f} | "
          f"mean visible/in_degree (where in>0): {(m.known[has_in] / m.in_deg[has_in]).mean():.3f}")

print("=== Check A ===")
vis(train, "train(all)")
vis(train[train.label.notna()], "train(labelled only)")
vis(test, "test")

# ---------- Check B: oracle ----------
print("\n=== Check B (oracle, cutoff 21, original 165 feats) ===")
params = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, verbosity=-1, n_jobs=4)
lab = train[train.label.notna()].copy()
lab["label"] = lab.label.astype(int)
base_tr = lab[lab.time_step <= 21]
STEPS = [24, 29, 31, 32, 22, 25, 35]  # first four are failing, rest are controls
rows = []
for seed in [0, 1, 2]:
    base = lgb.LGBMClassifier(random_state=seed, **params).fit(base_tr[F], base_tr.label)
    for s in STEPS:
        st = lab[lab.time_step == s]
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(st))
        k = int(0.2 * len(st))
        seen, held = st.iloc[idx[:k]], st.iloc[idx[k:]]
        if held.label.nunique() < 2 or seen.label.nunique() < 1:
            continue
        a0 = roc_auc_score(held.label, base.predict_proba(held[F])[:, 1])
        aug = pd.concat([base_tr, seen])
        m2 = lgb.LGBMClassifier(random_state=seed, **params).fit(aug[F], aug.label)
        a1 = roc_auc_score(held.label, m2.predict_proba(held[F])[:, 1])
        rows.append(dict(seed=seed, step=s, base_auc=a0, with_20pct_step_labels=a1, gain=a1 - a0))
r = pd.DataFrame(rows).groupby("step")[["base_auc", "with_20pct_step_labels", "gain"]].mean().round(4)
print(r)
print("\nREAD: failing steps (24,29,31,32) jump to >0.95 -> go (self-training worth building).")
print("      little change -> kill self-training, move to learned neighbour-score stacker.")