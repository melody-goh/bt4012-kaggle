"""Step 3: does a THIRD, low-correlation model improve the V2+V3-style blend?
Run next to train.csv, test.csv, txs_edgelist.csv. Needs lightgbm, sklearn, scipy; catboost optional.

Reference blend = equal logit average of
  A: LGBM on 165 feats + 18 graph feats (V2-equivalent)
  B: LGBM on feat_95..feat_165 + 18 graph feats (V3/last71 family)
Candidate C is added with equal weight (A,B,C). Candidates:
  logreg   : logistic regression on quantile-normal features (no time proxies)
  mlp      : 3-seed MLP on same features
  lgb_extra: LGBM extra_trees on features without time proxies and feat_2 (+graph)
  cat_l71  : CatBoost on feat_95..feat_165 + graph (if catboost is installed)

FROZEN RULE (set before running): a candidate is usable only if adding it improves the mean AUC
over the A+B reference by >= +0.003 AND it is better on >= 4 of 5 cutoffs.
Writes cand_<name>_test.csv (index,target) for every candidate; use ONLY passers.
Only train labels are used.
"""
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
edges = pd.read_csv("txs_edgelist.csv")[["txId1", "txId2"]]
F = [f"feat_{i}" for i in range(1, 166)]
L71 = [f"feat_{i}" for i in range(95, 166)]
TIME = ["feat_101", "feat_103", "feat_136"]
NT = [f for f in F if f not in TIME]
NT2 = [f for f in NT if f != "feat_2"]
AGG = ["feat_2", "feat_53", "feat_58", "feat_16", "feat_90", "feat_138", "feat_142", "feat_163"]
P = dict(n_estimators=300, learning_rate=0.03, num_leaves=31, verbosity=-1, n_jobs=4, random_state=42)
logit_ = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))

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

def attach(df):
    x = df.merge(agg, left_on="txId", right_index=True, how="left")
    x[["in_degree", "in_known_sources"]] = x[["in_degree", "in_known_sources"]].fillna(0)
    return x

lab = train[train.label.notna()].copy().reset_index(drop=True)
lab["label"] = lab.label.astype(int)
D = attach(lab)
T = attach(test)
assert len(D) == len(lab) and len(T) == len(test)

def lgbm(tr, ev, cols, **kw):
    return lgb.LGBMClassifier(**{**P, **kw}).fit(tr[cols], tr.label).predict_proba(ev[cols])[:, 1]

def prep(tr, ev, cols):
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0).fit(tr[cols])
    return qt.transform(tr[cols]), qt.transform(ev[cols])

def c_logreg(tr, ev):
    a, b = prep(tr, ev, NT)
    return LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced").fit(a, tr.label).predict_proba(b)[:, 1]

def c_mlp(tr, ev):
    a, b = prep(tr, ev, NT)
    ps = []
    for s in range(3):
        m = MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, early_stopping=True,
                          n_iter_no_change=10, max_iter=200, random_state=s).fit(a, tr.label)
        ps.append(m.predict_proba(b)[:, 1])
    return np.mean(ps, axis=0)

def c_lgb_extra(tr, ev):
    return lgbm(tr, ev, NT2 + GC, extra_trees=True, n_estimators=500)

CANDS = {"logreg": c_logreg, "mlp": c_mlp, "lgb_extra": c_lgb_extra}
try:
    from catboost import CatBoostClassifier
    def c_cat(tr, ev):
        cols = L71 + GC
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=0, thread_count=4)
        return m.fit(tr[cols], tr.label).predict_proba(ev[cols])[:, 1]
    CANDS["cat_l71"] = c_cat
except ImportError:
    print("catboost not installed: skipping cat_l71")

rows = []
for c in [7, 10, 14, 17, 21]:
    tr = D[D.time_step <= c]
    va = D[D.time_step.between(c + 1, min(c + 14, 35))]
    if va.label.nunique() < 2:
        continue
    pA = lgbm(tr, va, F + GC)
    pB = lgbm(tr, va, L71 + GC)
    ref = (logit_(pA) + logit_(pB)) / 2
    r0 = roc_auc_score(va.label, ref)
    for name, fn in CANDS.items():
        pc = fn(tr, va)
        blend = (logit_(pA) + logit_(pB) + logit_(pc)) / 3
        rows.append(dict(cutoff=c, cand=name, ref_AB=r0, cand_alone=roc_auc_score(va.label, pc),
                         spearman_vs_ref=pd.Series(pc).corr(pd.Series(ref), method="spearman"),
                         blend_ABC=roc_auc_score(va.label, blend), gain=roc_auc_score(va.label, blend) - r0))
    print("done cutoff", c, flush=True)

R = pd.DataFrame(rows)
print("\n", R.round(4).to_string(index=False))
print("\n=== frozen rule ===")
passers = []
for name, g in R.groupby("cand"):
    ok = (g.gain.mean() >= 0.003) and ((g.gain > 0).sum() >= 4)
    print(f"{name}: mean gain {g.gain.mean():+.4f} | better on {(g.gain > 0).sum()}/{len(g)} | PASS: {ok}")
    if ok:
        passers.append(name)

full = D
for name, fn in CANDS.items():
    p = fn(full, T)
    pd.DataFrame({"index": test["index"], "target": p}).to_csv(f"cand_{name}_test.csv", index=False)
print("\nsaved cand_*_test.csv for all candidates. USE ONLY PASSERS:", passers or "none")
print("Next: logit-average submission_v2.csv, submission_v3.csv and the passing cand file(s) with equal weights.")