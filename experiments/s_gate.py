import json
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission
ROOT = Path(".").resolve().parent; CACHE = Path("s_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges); lab = train[train.label.notna()].reset_index(drop=True)
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return L(m.predict_proba(V[cols])[:, 1])
def em(p, pt, it=200):
    pi = pt
    for _ in range(it):
        a = pi / pt * p; b = (1 - pi) / (1 - pt) * (1 - p); new = np.mean(a / (a + b))
        if abs(new - pi) < 1e-7: break
        pi = new
    return pi
def adjust(z, steps, pt, strength=1., true=None):
    out = z.copy(); est = {}
    for s in np.unique(steps):
        m = steps == s; pi = true[s] if true is not None else em(expit(z[m]), pt); pi = np.clip(pi, 1e-4, 1 - 1e-4)
        out[m] = z[m] + strength * (logit(pi) - logit(pt)); est[int(s)] = round(float(pi), 4)
    return out, est
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy(); st = va.time_step.to_numpy()
    w = 2. ** (-(c - tr.time_step.to_numpy()) / 28); w /= w.mean(); pt = float(np.average(tr.label.astype(int), weights=w) * .5 + tr.label.astype(int).mean() * .5)
    z = (fp(tr, va, BASE + g, w) + fp(tr, va, L71)) / 2; true = va.groupby("time_step").label.apply(lambda x: x.astype(int).mean()).to_dict()
    s1, est = adjust(z, st, pt); s2, _ = adjust(z, st, pt, .5); orc, _ = adjust(z, st, pt, 1., true)
    cache[str(c)] = {"D": roc_auc_score(y, z), "S1": roc_auc_score(y, s1), "S2": roc_auc_score(y, s2), "oracle": roc_auc_score(y, orc)}
    print(c, "est vs true prevalence:", {k: (est[k], round(true[k], 3)) for k in list(est)[:6]}, flush=True)
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string()); passers = {}
for k in ["S1", "S2"]:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok} | {d.round(4).tolist()}")
    if ok: passers[k] = d.mean()
print(f"oracle: mean {(R.oracle - R.D).mean():+.4f}")
# test: apply to v12 directly (no refit needed)
v12 = pd.read_csv(ROOT / "submission_v12.csv"); zt = L(v12.target.to_numpy()); w = 2. ** (-(35 - lab.time_step.to_numpy()) / 28); w /= w.mean()
pt = float(np.average(lab.label.astype(int), weights=w) * .5 + lab.label.astype(int).mean() * .5); _, est = adjust(zt, test.time_step.to_numpy(), pt)
print("training prior", round(pt, 4), "| TEST estimated illicit prevalence by step:", est)
for k in passers:
    a, _ = adjust(zt, test.time_step.to_numpy(), pt, 1. if k == "S1" else .5)
    out = pd.DataFrame({"index": test["index"], "target": expit(a)}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{k}_priorshift.csv", index=False); print("saved", k, "| Spearman vs v12:", round(out.target.corr(v12.target, method="spearman"), 4))
if not passers: print("No candidate passes: keep v12.")
