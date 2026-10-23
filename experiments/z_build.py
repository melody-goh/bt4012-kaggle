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
F = lambda f: L(pd.read_csv(ROOT / f"submission_{f}.csv").target.to_numpy())
a, r20, v7, x1 = F("v12"), F("v16_new"), F("v7"), F("X1_hl7"); r15 = 2 * F("Y2_v12_r15") - a; r10 = R(10)
v20 = pd.read_csv(ROOT / "submission_v20.csv").target; step = test.time_step.to_numpy()
C = {"Z8_plus_r10": (a + r20 + r15 + r10 + v7) / 5, "Z9_plus_r10_x1": (a + r20 + r15 + r10 + v7 + x1) / 6}
for k, z in C.items():
    out = pd.DataFrame({"index": test["index"], "target": expit(z)}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{k}.csv", index=False); top = out.target >= out.target.quantile(.95)
    print(f"{k}: Spearman vs v20 {out.target.corr(v20, method='spearman'):.4f} | top5% share 43-49 {(top & (step >= 43)).sum() / top.sum():.3f}")
