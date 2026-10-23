from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from pipeline import BASE, load_data, graph_features, validate_submission
ROOT = Path(".").resolve().parent; L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()]
fp = lambda cols, w=None: L(LGBMClassifier(**P).fit(lab[cols], lab.label.astype(int), sample_weight=w).predict_proba(test[cols])[:, 1])
w = 2. ** (-(35 - lab.time_step.to_numpy()) / 28); w /= w.mean()
out = pd.DataFrame({"index": test["index"], "target": expit((fp(BASE + g, w) + fp(L71, w)) / 2)}); validate_submission(out, test, sample)
out.to_csv(ROOT / "submission_E3_draw.csv", index=False)
v12 = pd.read_csv(ROOT / "submission_v12.csv"); f2 = pd.read_csv(ROOT / "submission_F2_draw.csv")
print("saved E3 | Spearman vs v12:", round(out.target.corr(v12.target, method="spearman"), 4), "| vs F2:", round(out.target.corr(f2.target, method="spearman"), 4))
