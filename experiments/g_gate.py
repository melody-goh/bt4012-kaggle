import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
T0 = time.time(); ROOT = Path(".").resolve().parent; CACHE = Path("g_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]; NP = [c for c in L71 if c not in ("feat_101", "feat_103", "feat_136")]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return m.predict_proba(V[cols])[:, 1]
def W(tr, c, hl): w = 2. ** (-(c - tr.time_step.to_numpy()) / hl); return w / w.mean()
def hardw(tr):
    steps = np.sort(tr.time_step.unique()); blocks = np.array_split(steps, 3); oof = np.zeros(len(tr)); y = tr.label.astype(int).to_numpy()
    for b in blocks:
        te = tr.time_step.isin(b).to_numpy(); oof[te] = fp(tr[~te], tr[te], L71)
    w = 1 + 2 * np.abs(y - oof); return w / w.mean()
def cands(tr, V, c):
    w28 = L(fp(tr, V, BASE + g, W(tr, c, 28))); h = hardw(tr)
    l71 = L(fp(tr, V, L71)); l71h = L(fp(tr, V, L71, h)); np_ = L(fp(tr, V, NP)); nph = L(fp(tr, V, NP, h))
    return {"D": (w28 + l71) / 2, "G1": (w28 + l71h) / 2, "G2": (w28 + np_) / 2, "G3": (w28 + nph) / 2}
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    if str(c) in cache: continue
    if time.time() - T0 > 100: print("time budget; rerun to continue"); break
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    cache[str(c)] = {k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}; CACHE.write_text(json.dumps(cache)); print("done", c, flush=True)
if len(cache) < 6: raise SystemExit("incomplete")
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string()); passers = {}
for k in ["G1", "G2", "G3"]:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok} | {d.round(4).tolist()}")
    if ok: passers[k] = d.mean()
if passers:
    b = max(passers, key=passers.get); C = cands(lab, test, 35)
    out = pd.DataFrame({"index": test["index"], "target": expit(C[b])}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{b}_hard.csv", index=False); print("saved", b, "| Spearman vs v12:", round(out.target.corr(pd.read_csv(ROOT / "submission_v12.csv").target, method="spearman"), 4))
else: print("No candidate passes: keep v12.")
