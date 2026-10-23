"""Final B1 = (V2 + last71 + geo specialist)/3 at cutoff 35. Stages: python final_geo.py train | test | fit. Resumable per step.
Geometry: time-proxy-free quantile-normal -> PCA40 fit on labelled+unknown <= 35 (features only). No label values used for features.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from scipy.special import logit, expit
from pipeline import BASE, load_data, graph_features, validate_submission
HERE = Path(".").resolve(); ROOT = HERE.parent
TIME = ["feat_101", "feat_103", "feat_136"]; NT = [f for f in BASE if f not in TIME]; L71 = BASE[94:]
FC = ["f1_dens_step", "f2_d_unk", "f3_d_labpast"]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
PG = dict(n_estimators=200, learning_rate=.03, num_leaves=7, min_child_samples=50, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
lab = train[train.label.notna()]; unk = train[train.label.isna()]; c = 35
stage = sys.argv[1]
fit_rows = pd.concat([lab[NT], unk[NT]])
qt = QuantileTransformer(output_distribution="normal", n_quantiles=300, random_state=0).fit(fit_rows)
pca = PCA(40, random_state=0).fit(qt.transform(fit_rows)); Z = lambda d: pca.transform(qt.transform(d[NT])).astype(np.float32)
knn = lambda ref, q, k=5: NearestNeighbors(n_neighbors=k, algorithm="brute", n_jobs=4).fit(ref).kneighbors(q)[0].mean(axis=1)
ZU, su = Z(unk), unk.time_step.to_numpy(); ZL, sl = Z(lab), lab.time_step.to_numpy()
def feats(q, name):
    Zq = Z(q); st = q.time_step.to_numpy(); out = np.zeros((len(q), 3), np.float32)
    for s in np.unique(st):
        f = HERE / f"final_{name}_step{s}.npy"
        if not f.exists():
            m = st == s; zz = Zq[m]; r = np.zeros((m.sum(), 3), np.float32)
            r[:, 0] = NearestNeighbors(n_neighbors=6, algorithm="brute", n_jobs=4).fit(zz).kneighbors(zz)[0][:, 1:].mean(axis=1)
            r[:, 1] = knn(ZU[su != s], zz); r[:, 2] = knn(ZL[sl != s], zz); np.save(f, r)
        out[st == s] = np.load(f)
    return pd.DataFrame(out, columns=FC, index=q.index)
if stage == "train": print("train feats done", feats(lab, "train").shape)
elif stage == "test": print("test feats done", feats(test, "test").shape)
else:
    Ftr, Fte = feats(lab, "train"), feats(test, "test"); d = lab.join(Ftr); T = test.join(Fte); y = d.label.astype(int)
    print("f2_d_unk median: train rows", float(Ftr.f2_d_unk.median()), "| test rows", float(Fte.f2_d_unk.median()), "| by test step:", Fte.groupby(test.time_step).f2_d_unk.median().round(1).to_dict())
    fp = lambda cols, params=P: L(LGBMClassifier(**params).fit(d[cols], y).predict_proba(T[cols])[:, 1])
    v2, l71, geo = fp(BASE + g), fp(L71), fp(FC + ["in_degree"], PG)
    out = pd.DataFrame({"index": test["index"], "target": expit((v2 + l71 + geo) / 3)}); validate_submission(out, test, sample)
    out.to_csv(ROOT / "submission_B1_geo.csv", index=False)
    v10 = pd.read_csv(ROOT / "submission_v10.csv")
    print("saved submission_B1_geo.csv | Spearman vs v10 (0.95083):", round(out.target.corr(v10.target, method="spearman"), 4),
          "| geo vs v2:", round(pd.Series(geo).corr(pd.Series(v2), method="spearman"), 3))
    print("mean predicted risk by test step (B1 vs v10):"); print(pd.DataFrame({"B1": out.target.values, "v10": v10.target.values, "step": test.time_step.values}).groupby("step").mean().round(3).T.to_string())
