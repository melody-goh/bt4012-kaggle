import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
T0 = time.time(); ROOT = Path(".").resolve().parent; CACHE = Path("t_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
V = {"T1": dict(num_leaves=15), "T2": dict(num_leaves=63), "T3": dict(min_child_samples=100), "T4": dict(reg_lambda=5, min_child_samples=50), "T5": dict(n_estimators=600, learning_rate=.015)}
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
def fp(X, Vv, cols, w=None, **kw):
    m = LGBMClassifier(**{**P, **kw}).fit(X[cols], X.label.astype(int), sample_weight=w); return L(m.predict_proba(Vv[cols])[:, 1])
def W(tr, c): w = 2. ** (-(c - tr.time_step.to_numpy()) / 28); return w / w.mean()
def cands(tr, Vv, c, only=None):
    w = W(tr, c); l71 = fp(tr, Vv, L71); out = {"D": (fp(tr, Vv, BASE + g, w) + l71) / 2}
    for k, kw in V.items():
        if only is None or k == only: out[k] = (fp(tr, Vv, BASE + g, w, **kw) + l71) / 2
    return out
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    if str(c) in cache: continue
    if time.time() - T0 > 90: print("time budget; rerun"); break
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    cache[str(c)] = {k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}; CACHE.write_text(json.dumps(cache)); print("done", c, flush=True)
if len(cache) < 6: raise SystemExit("incomplete")
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string()); passers = {}
for k in V:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok}")
    if ok: passers[k] = d.mean()
if passers:
    b = max(passers, key=passers.get); C = cands(lab, test, 35, b)
    out = pd.DataFrame({"index": test["index"], "target": expit(C[b])}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{b}_tuned.csv", index=False); print("saved", b, "| Spearman vs v12:", round(out.target.corr(pd.read_csv(ROOT / "submission_v12.csv").target, method="spearman"), 4))
else: print("No candidate passes: keep v12.")
