import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
T0 = time.time(); ROOT = Path(".").resolve().parent; CACHE = Path("r_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]; RK = [c + "_rk" for c in L71]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
lab[RK] = lab.groupby("time_step")[L71].rank(pct=True).to_numpy(); test[RK] = test.groupby("time_step")[L71].rank(pct=True).to_numpy()
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return m.predict_proba(V[cols])[:, 1]
def W(tr, c, hl): w = 2. ** (-(c - tr.time_step.to_numpy()) / hl); return w / w.mean()
def cands(tr, V, c):
    w28 = L(fp(tr, V, BASE + g, W(tr, c, 28))); l71 = L(fp(tr, V, L71)); rk = L(fp(tr, V, RK))
    return {"D": (w28 + l71) / 2, "R1": (w28 + rk) / 2, "R2": (w28 + l71 + rk) / 3, "_rk": rk}
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    if str(c) in cache: continue
    if time.time() - T0 > 100: print("time budget; rerun"); break
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    cache[str(c)] = {k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}; CACHE.write_text(json.dumps(cache)); print("done", c, flush=True)
if len(cache) < 6: raise SystemExit("incomplete")
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string()); passers = {}
for k in ["R1", "R2"]:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok} | {d.round(4).tolist()}")
    if ok: passers[k] = d.mean()
if passers:
    b = max(passers, key=passers.get); C = cands(lab, test, 35)
    out = pd.DataFrame({"index": test["index"], "target": expit(C[b])}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{b}_rank.csv", index=False); print("saved", b, "| Spearman vs v12:", round(out.target.corr(pd.read_csv(ROOT / "submission_v12.csv").target, method="spearman"), 4))
else: print("No candidate passes: keep v12.")
