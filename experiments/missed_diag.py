from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features
ROOT = Path(__file__).resolve().parent.parent
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
lab = train.loc[train.label.notna()]
tr = lab[lab.time_step <= 21]; va = lab[lab.time_step.between(22, 35)].copy()
m = LGBMClassifier(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4).fit(tr[BASE + g], tr.label.astype(int))
va["p"] = m.predict_proba(va[BASE + g])[:, 1]
F = [24, 29, 31, 32]
fa = va[va.time_step.isin(F)]
miss = fa[(fa.label == 1) & (fa.p < .05)]; caught = fa[(fa.label == 1) & (fa.p > .5)]; lic = fa[fa.label == 0]
print(f"failing steps: licit {len(lic)}, missed illicit {len(miss)}, caught illicit {len(caught)}")
rows = []
trf = tr
for f in BASE + g:
    x = pd.concat([miss[f], lic[f]]).fillna(-999); y = np.r_[np.ones(len(miss)), np.zeros(len(lic))]
    a_miss = roc_auc_score(y, x)
    a_tr = roc_auc_score(trf.label, trf[f].fillna(-999))
    rows.append((f, a_miss, a_tr))
R = pd.DataFrame(rows, columns=["feat", "auc_missed_vs_licit_failsteps", "auc_train_illicit_vs_licit"]).set_index("feat")
R["strength_miss"] = (R.iloc[:, 0] - .5).abs(); R["strength_tr"] = (R.iloc[:, 1] - .5).abs()
R["flip"] = np.sign(R.iloc[:, 0] - .5) != np.sign(R.iloc[:, 1] - .5)
R["in_last71"] = [f in BASE[94:] for f in R.index]
print("\nTop 25 features separating MISSED illicit from licit in failing steps:")
print(R.sort_values("strength_miss", ascending=False).head(25)[["auc_missed_vs_licit_failsteps", "auc_train_illicit_vs_licit", "flip", "in_last71"]].round(3).to_string())
top = R.sort_values("strength_miss", ascending=False).head(40)
print(f"\nof top-40: in_last71 = {int(top.in_last71.sum())}, direction flipped vs train = {int(top.flip.sum())}")
print("\nAUC of single best features alone (missed vs licit):", R.strength_miss.max().round(3) + .5)
print("\nmean predicted p: missed", miss.p.mean().round(4), "| licit", lic.p.mean().round(4))
for f in ["in_degree", "in_known_sources"]:
    print(f, "mean: missed", miss[f].mean().round(2), "caught", caught[f].mean().round(2), "licit", lic[f].mean().round(2))
# best simple model trained ONLY to separate missed-like: LGBM on last71 using failing-step labels, 5-fold, to see learnability by feature block
from sklearn.model_selection import StratifiedKFold
Xf = fa[BASE + g]; yf = fa.label.astype(int).to_numpy()
for name, cols in [("all", BASE + g), ("first94", BASE[:94]), ("last71", BASE[94:])]:
    s = np.zeros(len(fa))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Xf, yf):
        s[b] = LGBMClassifier(n_estimators=200, learning_rate=.05, verbosity=-1, n_jobs=4, random_state=0).fit(fa.iloc[a][cols], yf[a]).predict_proba(fa.iloc[b][cols])[:, 1]
    print(f"in-step CV AUC on failing steps using {name}: {roc_auc_score(yf, s):.4f}")
