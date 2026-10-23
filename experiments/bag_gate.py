import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
T0 = time.time(); ROOT = Path(".").resolve().parent; CACHE = Path("bag_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4); B = dict(P, subsample=.8, subsample_freq=1, colsample_bytree=.8)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
def fp(X, V, cols, w=None, prm=P):
    m = LGBMClassifier(**prm).fit(X[cols], X.label.astype(int), sample_weight=w); return L(m.predict_proba(V[cols])[:, 1])
def W(tr, c): w = 2. ** (-(c - tr.time_step.to_numpy()) / 28); return w / w.mean()
def cands(tr, V, c):
    w = W(tr, c); D = (fp(tr, V, BASE + g, w) + fp(tr, V, L71)) / 2
    bw = np.mean([fp(tr, V, BASE + g, w, dict(B, random_state=s)) for s in (1, 2, 3)], 0); bl = np.mean([fp(tr, V, L71, None, dict(B, random_state=s)) for s in (1, 2, 3)], 0)
    return {"D": D, "Dbag": (bw + bl) / 2}
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    if str(c) in cache: continue
    if time.time() - T0 > 95: print("time budget; rerun"); break
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    cache[str(c)] = {k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}; CACHE.write_text(json.dumps(cache)); print("done", c, flush=True)
if len(cache) < 6: raise SystemExit("incomplete")
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string())
d = R.Dbag - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
print(f"Dbag: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok} | {d.round(4).tolist()}")
