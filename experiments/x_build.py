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
l71 = fp(lab, L71); rec = lab[lab.time_step >= 20]
C = {"X1_hl7": (fp(lab, BASE + g, W(lab, 7)) + l71) / 2,
     "X2_recent20": (fp(rec, BASE + g, W(rec, 28)) + fp(rec, L71)) / 2,
     "X3_no_first94": (fp(lab, L71 + g, W(lab, 28)) + l71) / 2}
v12 = pd.read_csv(ROOT / "submission_v12.csv"); outs = {}
for k, z in C.items():
    out = pd.DataFrame({"index": test["index"], "target": expit(z)}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{k}.csv", index=False); outs[k] = out.target
    print(f"saved submission_{k}.csv | Spearman vs v12: {out.target.corr(v12.target, method='spearman'):.4f}")
print("pairwise Spearman:", {f"{a}-{b}": round(outs[a].corr(outs[b], method='spearman'), 3) for i, a in enumerate(outs) for b in list(outs)[i + 1:]})
