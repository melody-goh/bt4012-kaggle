from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from pipeline import BASE, load_data, graph_features, validate_submission
ROOT = Path(".").resolve().parent; L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()]
def fp(X, cols, w=None): return L(LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w).predict_proba(test[cols])[:, 1])
def W(X, hl): w = 2. ** (-(35 - X.time_step.to_numpy()) / hl); return w / w.mean()
def R(k): rec = lab[lab.time_step >= k]; return (fp(rec, BASE + g, W(rec, 28)) + fp(rec, L71)) / 2
v12 = pd.read_csv(ROOT / "submission_v12.csv"); y1 = pd.read_csv(ROOT / "submission_v17_blend.csv")
x2 = pd.read_csv(ROOT / "submission_v16_new.csv"); lv = L(v12.target.to_numpy())
r = {k: R(k) for k in (15, 25)}; r[20] = L(x2.target.to_numpy())
C = {"Y2_v12_r15": (lv + r[15]) / 2, "Y3_v12_r25": (lv + r[25]) / 2, "Y4_v12_rbag": (lv + (r[15] + r[20] + r[25]) / 3) / 2}
step = test.time_step.to_numpy()
for k, z in C.items():
    out = pd.DataFrame({"index": test["index"], "target": expit(z)}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{k}.csv", index=False); top = out.target >= out.target.quantile(.95)
    print(f"saved submission_{k}.csv | Spearman vs v12 {out.target.corr(v12.target, method='spearman'):.4f}"
          f" vs Y1 {out.target.corr(y1.target, method='spearman'):.4f} | top5% share 43-49 {(top & (step >= 43)).sum() / top.sum():.3f}")
