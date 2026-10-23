from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features
HERE = Path(".").resolve(); ROOT = HERE.parent
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
PG = dict(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=50, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
lab = train[train.label.notna()]; FC = ["f1_dens_step", "f2_d_unk", "f3_d_labpast"]
def within(y, s, st):
    num = den = 0
    for t in np.unique(st):
        m = st == t; a, b = y[m].sum(), (1 - y[m]).sum()
        if a and b: num += roc_auc_score(y[m], s[m]) * a * b; den += a * b
    return num / den
def fp(X, y, V, cols, params=P):
    m = LGBMClassifier(**params).fit(X[cols], y); return L(m.predict_proba(V[cols])[:, 1])
for c in [17, 21]:
    F = pd.read_pickle(HERE / f"knnfeat_c{c}.pkl"); d = lab.loc[F.index].join(F); tr = d[d.time_step <= c]; va = d[d.time_step > c].copy(); y = tr.label.astype(int); yv = va.label.astype(int).to_numpy(); st = va.time_step.to_numpy()
    print(f"\n===== cutoff {c}: single-feature AUC, pooled vs within-step vs step-demeaned (diagnostic) =====")
    size = va.groupby("time_step").size(); prev = va.groupby("time_step").label.mean()
    for f in FC:
        x = va[f].to_numpy(); dm = x - va.groupby("time_step")[f].transform("mean").to_numpy()
        pooled = 1 - roc_auc_score(yv, x) if roc_auc_score(yv, x) < .5 else roc_auc_score(yv, x)
        sgn = -1 if roc_auc_score(yv, x) < .5 else 1
        stepmean = va.groupby("time_step")[f].mean()
        print(f"  {f:14s} pooled {pooled:.3f} | within-step {1 - within(yv, x, st) if sgn < 0 else within(yv, x, st):.3f} | step-demeaned pooled {1 - roc_auc_score(yv, dm) if sgn < 0 else roc_auc_score(yv, dm):.3f} | corr(step mean, step illicit rate) {spearmanr(stepmean, prev.loc[stepmean.index])[0]:+.2f} | corr(step mean, step size) {spearmanr(stepmean, size.loc[stepmean.index])[0]:+.2f}")
    v2, l71 = fp(tr, y, va, BASE + g), fp(tr, y, va, L71)
    sets = {"geo(f1,f2,f3,deg)": FC + ["in_degree"], "geo_noF1(f2,f3,deg)": ["f2_d_unk", "f3_d_labpast", "in_degree"], "geo_F2_only": ["f2_d_unk"]}
    ref = (v2 + l71) / 2; print(f"  ref pooled {roc_auc_score(yv, ref):.4f} within {within(yv, ref, st):.4f}")
    for name, cols in sets.items():
        geo = fp(tr, y, va, cols, PG); b = (v2 + l71 + geo) / 3
        print(f"  {name:20s} alone pooled {roc_auc_score(yv, geo):.4f} within {within(yv, geo, st):.4f} || B blend pooled {roc_auc_score(yv, b):.4f} within {within(yv, b, st):.4f}  (gain pooled {roc_auc_score(yv, b) - roc_auc_score(yv, ref):+.4f}, within {within(yv, b, st) - within(yv, ref, st):+.4f})")
# training-step relation: do step-level feature means track step size / prevalence in TRAIN as well?
F = pd.read_pickle(HERE / "knnfeat_c21.pkl"); d = lab.loc[F.index].join(F); d = d[d.time_step <= 21]
size = d.groupby("time_step").size(); prev = d.groupby("time_step").label.mean()
print("\nTRAIN steps<=21 corr of step-mean feature with step prevalence / step size:")
for f in FC: sm = d.groupby("time_step")[f].mean(); print(f"  {f:14s} prev {spearmanr(sm, prev)[0]:+.2f} | size {spearmanr(sm, size)[0]:+.2f}")
print("labelled rows per step: train median", int(size.median()), "| test rows per step (median)", int(test.groupby('time_step').size().median()), "| test per-step:", test.groupby('time_step').size().to_dict())
