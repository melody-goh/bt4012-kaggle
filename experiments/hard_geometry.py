from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from pipeline import BASE, load_data, graph_features
ROOT = Path(".").resolve().parent
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
TIME = ["feat_101", "feat_103", "feat_136"]; NT = [f for f in BASE if f not in TIME]
lab = train[train.label.notna()]; unk = train[train.label.isna()]
c = 21; tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(22, 35)].copy(); un = unk[unk.time_step <= c]
m = LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4).fit(tr[BASE + g], tr.label.astype(int))
va["p"] = m.predict_proba(va[BASE + g])[:, 1]
miss = va[(va.label == 1) & (va.p < .05)]; caught = va[(va.label == 1) & (va.p > .5)]; lic = va[va.label == 0]
print(f"window 22-35: licit {len(lic)} | missed illicit {len(miss)} ({len(miss)/ (va.label==1).sum():.0%} of illicit) | caught {len(caught)}")
print("missed illicit by step:", miss.time_step.value_counts().sort_index().to_dict())
qt = QuantileTransformer(output_distribution="normal", n_quantiles=300, random_state=0).fit(pd.concat([tr[NT], un[NT]]))
pca = PCA(40, random_state=0).fit(qt.transform(pd.concat([tr[NT], un[NT]])))
Z = lambda d: pca.transform(qt.transform(d[NT])).astype(np.float32)
Ztr, Zun, Zva = Z(tr), Z(un), Z(va); va = va.reset_index(drop=True)
ytr = tr.label.astype(int).to_numpy(); yva = va.label.astype(int).to_numpy()
# (a) novelty distances
def nn_d(ref, q, k=1): return NearestNeighbors(n_neighbors=k, n_jobs=4).fit(ref).kneighbors(q)[0].mean(axis=1)
sets = {"missed": (va.label == 1) & (va.p < .05), "caught": (va.label == 1) & (va.p > .5), "licit": va.label == 0}
d_ill = nn_d(Ztr[ytr == 1], Zva); d_lic = nn_d(Ztr[ytr == 0], Zva); d_unk = nn_d(Zun, Zva)
print("\n(a) median distance to nearest TRAINING illicit / licit / UNKNOWN row (PCA-40 of time-proxy-free quantile-normal features):")
for k, mk in sets.items(): print(f"  {k:7s} ill {np.median(d_ill[mk]):.2f} | licit {np.median(d_lic[mk]):.2f} | unknown {np.median(d_unk[mk]):.2f} | ratio ill/licit {np.median(d_ill[mk]/d_lic[mk]):.2f}")
# (b) kNN classifier
for name, cols in [("all", slice(None))]:
    knn = KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=4).fit(Ztr, ytr); pk = knn.predict_proba(Zva)[:, 1]
    fs = va.time_step.isin([24, 29, 31, 32]).to_numpy()
    print(f"\n(b) kNN(15) trained on <=21: AUC {roc_auc_score(yva, pk):.4f} (LGBM V2 {roc_auc_score(yva, va.p):.4f}); failing steps kNN {roc_auc_score(yva[fs], pk[fs]):.4f} vs LGBM {roc_auc_score(yva[fs], va.p[fs]):.4f}")
    mm = sets["missed"].to_numpy() | (va.label == 0).to_numpy()
    print(f"    missed-vs-licit AUC: kNN {roc_auc_score(sets['missed'].to_numpy()[mm], pk[mm]):.4f} | LGBM {roc_auc_score(sets['missed'].to_numpy()[mm], va.p[mm]):.4f}")
# (c) cluster structure of missed illicit + purity in the validation window (diagnostic: uses validation labels)
nbr = NearestNeighbors(n_neighbors=11, n_jobs=4).fit(Zva); idx = nbr.kneighbors(Zva)[1][:, 1:]
pur = (yva[idx]).mean(axis=1); samestep = (va.time_step.to_numpy()[idx] == va.time_step.to_numpy()[:, None]).mean(axis=1)
print("\n(c) 10-NN purity inside the 22-35 window (fraction of a row's neighbours that are illicit):")
for k, mk in sets.items(): print(f"  {k:7s} mean purity {pur[mk.to_numpy()].mean():.2f} | share of neighbours from same step {samestep[mk.to_numpy()].mean():.2f}")
Zm = Zva[sets["missed"].to_numpy()]
for k in range(2, 7):
    lab_k = KMeans(k, n_init=5, random_state=0).fit_predict(Zm); print(f"  KMeans k={k}: silhouette {silhouette_score(Zm, lab_k):.2f}, largest cluster {np.bincount(lab_k).max()/len(lab_k):.0%}")
# (d) regime persistence of flipping features across ALL steps
labs = pd.concat([lab, ]); rows = []
for t in range(1, 36):
    s = lab[lab.time_step == t]
    if s.label.sum() >= 8 and (s.label == 0).sum() >= 8:
        rows.append(dict(step=t, n_ill=int(s.label.sum()), **{f: roc_auc_score(s.label, s[f]) for f in ["feat_49", "feat_46", "feat_88", "feat_76", "feat_113"]}))
R = pd.DataFrame(rows).set_index("step"); print("\n(d) per-step AUC of illicit-vs-licit for features that flipped (0.5 = no signal):"); print(R.round(2).T.to_string())
sg = np.sign(R[["feat_49", "feat_46", "feat_88", "feat_76", "feat_113"]] - .5)
print("lag-1 sign agreement of consecutive steps:", {f: round(float((sg[f].values[1:] == sg[f].values[:-1]).mean()), 2) for f in sg.columns})
