"""Label-free kNN features (no label VALUES used; 'labelled population' membership only, which also holds for all test rows).
f1 dens_step : mean dist to 5 nearest labelled-population rows in the SAME step (self excluded)
f2 d_unk     : mean dist to 5 nearest UNKNOWN rows, steps <= c, other steps
f3 d_labpast : mean dist to 5 nearest labelled-population rows, steps <= c, other steps
Geometry: time-proxy-free quantile-normal -> PCA40, fit on labelled+unknown <= c. Resumable per cutoff. Then evaluates vs ref V2+last71.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.special import logit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features
HERE = Path(".").resolve(); ROOT = HERE.parent
TIME = ["feat_101", "feat_103", "feat_136"]; NT = [f for f in BASE if f not in TIME]; L71 = BASE[94:]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
lab = train[train.label.notna()]; unk = train[train.label.isna()]

def knn_mean(ref, q, k=5): return NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=4).fit(ref).kneighbors(q)[0].mean(axis=1)

def build(c, qrows, final=False):
    """features for qrows (DataFrame with txId,time_step,NT cols); pop = labelled-pop rows used as reference (lab<=c plus qrows' own steps for f1)"""
    fit_rows = pd.concat([lab[lab.time_step <= c][NT], unk[unk.time_step <= c][NT]])
    qt = QuantileTransformer(output_distribution="normal", n_quantiles=300, random_state=0).fit(fit_rows)
    pca = PCA(40, random_state=0).fit(qt.transform(fit_rows)); Z = lambda d: pca.transform(qt.transform(d[NT])).astype(np.float32)
    Zq = Z(qrows); st = qrows.time_step.to_numpy()
    U = unk[unk.time_step <= c]; ZU = Z(U); su = U.time_step.to_numpy()
    LP = lab[lab.time_step <= c]; ZL = Z(LP); sl = LP.time_step.to_numpy()
    out = np.zeros((len(qrows), 3), np.float32)
    for s in np.unique(st):
        m = st == s
        # f1: same-step density among the labelled-population rows of that step (the qrows of that step ARE that population)
        zz = Zq[m]; d = NearestNeighbors(n_neighbors=6, algorithm="brute", n_jobs=4).fit(zz).kneighbors(zz)[0][:, 1:]; out[m, 0] = d.mean(axis=1)
        out[m, 1] = knn_mean(ZU[su != s], zz); out[m, 2] = knn_mean(ZL[sl != s], zz)
    return pd.DataFrame(out, columns=["f1_dens_step", "f2_d_unk", "f3_d_labpast"], index=qrows.index)

def feats_for(c):
    f = HERE / f"knnfeat_c{c}.pkl"
    if f.exists(): return pd.read_pickle(f)
    q = lab[lab.time_step <= min(c + 14, 35)]; df = build(c, q); df.to_pickle(f); print("features built for cutoff", c, flush=True); return df

def fp(X, y, V, cols):
    m = LGBMClassifier(**P).fit(X[cols], y); return L(m.predict_proba(V[cols])[:, list(m.classes_).index(1)])

cut = [int(x) for x in sys.argv[1:]] or [10, 14, 17, 21]
res = HERE / "knn_results.csv"; done = pd.read_csv(res) if res.exists() else pd.DataFrame()
for c in cut:
    if len(done) and c in set(done.cutoff): continue
    F = feats_for(c); d = lab.loc[F.index].join(F); tr = d[d.time_step <= c]; va = d[d.time_step > c]; y = tr.label.astype(int); yv = va.label.astype(int).to_numpy()
    FC = list(F.columns); r = {"cutoff": c}
    for f in FC:
        r[f"auc_{f}"] = max(roc_auc_score(yv, va[f]), 1 - roc_auc_score(yv, va[f])); r[f"dir_{f}"] = "+" if roc_auc_score(yv, va[f]) > .5 else "-"
    v2, l71 = fp(tr, y, va, BASE + g), fp(tr, y, va, L71); v2k, l71k = fp(tr, y, va, BASE + g + FC), fp(tr, y, va, L71 + FC)
    r.update(v2=roc_auc_score(yv, v2), v2k=roc_auc_score(yv, v2k), ref=roc_auc_score(yv, (v2 + l71) / 2), K1=roc_auc_score(yv, (v2k + l71k) / 2), K2=roc_auc_score(yv, (v2 + l71k) / 2))
    done = pd.concat([done, pd.DataFrame([r])]); done.to_csv(res, index=False); print("evaluated", c, flush=True)
R = done.set_index("cutoff").sort_index(); pd.set_option("display.width", 250)
print(R[[c for c in R.columns if c.startswith("auc_") or c.startswith("dir_")]].to_string()); print(R[["v2", "v2k", "ref", "K1", "K2"]].round(4).to_string())
for k in ["K1", "K2"]:
    dd = R[k] - R.ref; print(f"{k}: mean gain vs ref {dd.mean():+.4f} | better on {(dd > 0).sum()}/{len(dd)}")
print(f"v2k vs v2: mean gain {(R.v2k - R.v2).mean():+.4f}")
