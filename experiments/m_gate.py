import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
T0 = time.time(); ROOT = Path(".").resolve().parent; CACHE = Path("m_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return m.predict_proba(V[cols])[:, 1]
def W(tr, c, hl): w = 2. ** (-(c - tr.time_step.to_numpy()) / hl); return w / w.mean()
def advw(tr, tgt):
    A = dict(n_estimators=150, learning_rate=.05, num_leaves=15, random_state=1, verbosity=-1, n_jobs=4); n = len(tr); idx = np.arange(n) % 2; p = np.zeros(n)
    tg = tgt[L71].sample(min(len(tgt), 20000), random_state=0).reset_index(drop=True); half = np.arange(len(tg)) % 2
    for f in (0, 1):
        Xtr = pd.concat([tr.loc[idx != f, L71], tg[half != f]]); y = np.r_[np.zeros((idx != f).sum()), np.ones((half != f).sum())]
        m = LGBMClassifier(**A).fit(Xtr, y, sample_weight=np.r_[np.ones((idx != f).sum()), np.full((half != f).sum(), (idx != f).sum() / (half != f).sum())])
        p[idx == f] = m.predict_proba(tr.loc[idx == f, L71])[:, 1]
    w = np.clip((p / (1 - p + 1e-9)) ** .5, .1, 10); return w / w.mean()
def cands(tr, V, c):
    a = advw(tr, V); r = W(tr, c, 28); ra = r * a; ra /= ra.mean(); l71 = L(fp(tr, V, L71))
    return {"D": (L(fp(tr, V, BASE + g, r)) + l71) / 2, "A1": (L(fp(tr, V, BASE + g, a)) + l71) / 2,
            "A2": (L(fp(tr, V, BASE + g, ra)) + l71) / 2, "A3": (L(fp(tr, V, BASE + g, ra)) + L(fp(tr, V, L71, a))) / 2}
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    if str(c) in cache: continue
    if time.time() - T0 > 90: print("time budget; rerun"); break
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    cache[str(c)] = {k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}; CACHE.write_text(json.dumps(cache)); print("done", c, flush=True)
if len(cache) < 6: raise SystemExit("incomplete")
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string()); passers = {}
for k in ["A1", "A2", "A3"]:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok} | {d.round(4).tolist()}")
    if ok: passers[k] = d.mean()
if passers:
    b = max(passers, key=passers.get); C = cands(lab, test, 35)
    out = pd.DataFrame({"index": test["index"], "target": expit(C[b])}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{b}_adv.csv", index=False); print("saved", b, "| Spearman vs v12:", round(out.target.corr(pd.read_csv(ROOT / "submission_v12.csv").target, method="spearman"), 4))
else: print("No candidate passes: keep v12.")
