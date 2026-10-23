import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, AGG, load_data, graph_features, validate_submission
T0 = time.time(); ROOT = Path(".").resolve().parent; CACHE = Path("k_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
groups = {f: [c for c in g if c.startswith(f"in_source_{f}_")] for f in AGG}; groups["deg"] = [c for c in g if "degree" in c or "known" in c]
assert sum(len(v) for v in groups.values()) == len(g), (len(g), {k: len(v) for k, v in groups.items()})
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return m.predict_proba(V[cols])[:, 1]
def W(tr, c, hl): w = 2. ** (-(c - tr.time_step.to_numpy()) / hl); return w / w.mean()
def cands(tr, V, c, only=None):
    w = W(tr, c, 28); l71 = L(fp(tr, V, L71)); out = {"D": (L(fp(tr, V, BASE + g, w)) + l71) / 2}
    for k, cols in groups.items():
        if only is None or k == only: out["drop_" + k] = (L(fp(tr, V, BASE + [x for x in g if x not in cols], w)) + l71) / 2
    return out
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    if str(c) in cache: continue
    if time.time() - T0 > 70: print("time budget; rerun"); break
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    cache[str(c)] = {k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}; CACHE.write_text(json.dumps(cache)); print("done", c, flush=True)
if len(cache) < 6: raise SystemExit("incomplete")
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string()); passers = {}
for k in [x for x in R if x.startswith("drop_")]:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok}")
    if ok: passers[k] = d.mean()
if passers:
    b = max(passers, key=passers.get); C = cands(lab, test, 35, b[5:])
    out = pd.DataFrame({"index": test["index"], "target": expit(C[b])}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{b}.csv", index=False); print("saved", b, "| Spearman vs v12:", round(out.target.corr(pd.read_csv(ROOT / "submission_v12.csv").target, method="spearman"), 4))
else: print("No candidate passes: keep v12.")
