import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, AGG, load_data, graph_features, validate_submission
T0 = time.time(); ROOT = Path(".").resolve().parent; CACHE = Path("o_gate_cache.json")
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
def outfeats(labonly, tag):
    tr_nodes = train[train.label.notna()] if labonly else train
    nodes = pd.concat([tr_nodes[["txId"] + AGG], test[["txId"] + AGG]], ignore_index=True)
    e = edges[["txId1", "txId2"]].merge(nodes.rename(columns={"txId": "txId2", **{c: "d_" + c for c in AGG}}), on="txId2", how="left")
    gr = e.groupby("txId1"); a = gr[["d_" + c for c in AGG]].agg(["mean", "std"]); a.columns = [f"{tag}_{c}_{s}" for c, s in a.columns]
    a[f"{tag}_degree"] = gr.size(); a[f"{tag}_known"] = gr["d_" + AGG[0]].count(); cols = list(a.columns)
    return a, cols
A1, O1c = outfeats(False, "out"); A2, O2c = outfeats(True, "outl")
for A in (A1, A2):
    train = train.merge(A, left_on="txId", right_index=True, how="left"); test = test.merge(A, left_on="txId", right_index=True, how="left")
for f in (train, test): f[["out_degree", "out_known", "outl_degree", "outl_known"]] = f[["out_degree", "out_known", "outl_degree", "outl_known"]].fillna(0)
lab = train[train.label.notna()].reset_index(drop=True)
cov = lambda f, k: round(float((f[k] > 0).mean()), 3)
print("COVERAGE  share with >=1 successor edge | >=1 visible successor (all) | (labelled-type only)")
print(" train-lab:", cov(lab, "out_degree"), cov(lab, "out_known"), cov(lab, "outl_known"))
print(" test     :", cov(test, "out_degree"), cov(test, "out_known"), cov(test, "outl_known"), flush=True)
def fp(X, V, cols, w=None):
    m = LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w); return L(m.predict_proba(V[cols])[:, 1])
def W(tr, c): w = 2. ** (-(c - tr.time_step.to_numpy()) / 28); return w / w.mean()
def cands(tr, V, c, only=None):
    w = W(tr, c); l71 = fp(tr, V, L71); out = {"D": (fp(tr, V, BASE + g, w) + l71) / 2}
    if only in (None, "O1"): out["O1"] = (fp(tr, V, BASE + g + O1c, w) + l71) / 2
    if only in (None, "O2"): out["O2"] = (fp(tr, V, BASE + g + O2c, w) + l71) / 2
    return out
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
for c in [7, 10, 14, 17, 21, 24]:
    if str(c) in cache: continue
    if time.time() - T0 > 100: print("time budget; rerun"); break
    tr = lab[lab.time_step <= c]; va = lab[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    cache[str(c)] = {k: roc_auc_score(y, v) for k, v in cands(tr, va, c).items()}; CACHE.write_text(json.dumps(cache)); print("done", c, flush=True)
if len(cache) < 6: raise SystemExit("incomplete")
R = pd.DataFrame(cache).T.astype(float).sort_index(key=lambda s: s.astype(int)); print(R.round(4).to_string()); passers = {}
for k in ["O1", "O2"]:
    d = R[k] - R.D; ok = d.mean() >= .001 and (d > 0).sum() >= 4 and d["17"] > 0 and d["21"] > 0
    print(f"{k}: mean {d.mean():+.4f} | better {(d > 0).sum()}/6 | 17:{d['17']:+.4f} 21:{d['21']:+.4f} | PASS {ok} | {d.round(4).tolist()}")
    if ok: passers[k] = d.mean()
if passers:
    b = max(passers, key=passers.get); C = cands(lab, test, 35, b)
    out = pd.DataFrame({"index": test["index"], "target": expit(C[b])}); validate_submission(out, test, sample)
    out.to_csv(ROOT / f"submission_{b}_outedge.csv", index=False); print("saved", b, "| Spearman vs v12:", round(out.target.corr(pd.read_csv(ROOT / "submission_v12.csv").target, method="spearman"), 4))
else: print("No candidate passes: keep v12.")
