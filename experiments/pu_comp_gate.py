"""PU-weak-negatives + connected-component features. Resumable. Train labels only; unknown rows used as weak NEGATIVES (weight w).
Reference = V2+last71 (public 0.95083). Cutoffs 7,10,14,17,21. Candidates declared before running:
  E1 V2 + l71_pu(.05) | E2 V2_pu(.05)+l71_pu(.05) | E3 V2 + l71_pu(.2) | E4 V2_pu(.2)+l71_pu(.2) | F1 V2c + l71c (component feats)
FROZEN RULE (5 candidates): mean gain >= +0.002 AND better on >= 4 of 5 cutoffs. Writes a file for the single best passer only.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission

ROOT = Path(__file__).resolve().parent.parent
RES = Path(__file__).resolve().parent / "pu_comp_results.csv"
PARAMS = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
L71 = BASE[94:]

train, test, edges, sample = load_data(ROOT)
train, test, g, _ = graph_features(train, test, edges)
# component features (structure only, no labels)
ids = pd.concat([edges.txId1, edges.txId2]).unique()
idx = pd.Series(np.arange(len(ids)), index=ids)
a, b = edges.txId1.map(idx).to_numpy(), edges.txId2.map(idx).to_numpy()
n = len(ids)
_, lab_c = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
size = np.bincount(lab_c); ne = np.bincount(lab_c[a], minlength=len(size))
cs = pd.Series(np.log1p(size[lab_c]), index=ids); cd = pd.Series(ne[lab_c] / size[lab_c], index=ids)
for df in (train, test):
    df["comp_logsize"] = df.txId.map(cs).fillna(0.0); df["comp_density"] = df.txId.map(cd).fillna(0.0)
C = ["comp_logsize", "comp_density"]

def fp(X, y, w, V, cols):
    m = LGBMClassifier(**PARAMS).fit(X[cols], y, sample_weight=w)
    return L(m.predict_proba(V[cols])[:, list(m.classes_).index(1)])

def run(cutoff, labtr, unk, V):
    """returns dict of candidate logit scores on V; labtr labelled train rows, unk unknown rows (<= cutoff)."""
    y = labtr.label.astype(int).to_numpy(); one = np.ones(len(labtr))
    z = {"v2": fp(labtr, y, one, V, BASE + g), "l71": fp(labtr, y, one, V, L71)}
    pu = pd.concat([labtr, unk]); ypu = np.r_[y, np.zeros(len(unk), int)]
    for w in (.05, .2):
        wt = np.r_[one, np.full(len(unk), w)]
        z[f"l71_pu{w}"] = fp(pu, ypu, wt, V, L71); z[f"v2_pu{w}"] = fp(pu, ypu, wt, V, BASE + g)
    z["v2c"] = fp(labtr, y, one, V, BASE + g + C); z["l71c"] = fp(labtr, y, one, V, L71 + C)
    return {"ref": (z["v2"] + z["l71"]) / 2, "E1": (z["v2"] + z["l71_pu0.05"]) / 2,
            "E2": (z["v2_pu0.05"] + z["l71_pu0.05"]) / 2, "E3": (z["v2"] + z["l71_pu0.2"]) / 2,
            "E4": (z["v2_pu0.2"] + z["l71_pu0.2"]) / 2, "F1": (z["v2c"] + z["l71c"]) / 2}, z

lab = train.loc[train.label.notna()]; unkall = train.loc[train.label.isna()]
done = pd.read_csv(RES) if RES.exists() else pd.DataFrame()
for c in [7, 10, 14, 17, 21]:
    if len(done) and c in set(done.cutoff): continue
    tr = lab.loc[lab.time_step <= c]; va = lab.loc[lab.time_step.between(c + 1, min(c + 14, 35))]
    cand, _ = run(c, tr, unkall.loc[unkall.time_step <= c], va)
    y = va.label.astype(int).to_numpy()
    row = {"cutoff": c, **{k: roc_auc_score(y, v) for k, v in cand.items()}}
    done = pd.concat([done, pd.DataFrame([row])]); done.to_csv(RES, index=False); print("done", c, flush=True)
R = done.set_index("cutoff").sort_index()
print("\n", R.round(4).to_string())
passers = {}
for k in ["E1", "E2", "E3", "E4", "F1"]:
    d = R[k] - R["ref"]; ok = d.mean() >= 0.002 and (d > 0).sum() >= 4
    print(f"{k}: mean gain {d.mean():+.4f} | better on {(d > 0).sum()}/5 | PASS {ok}")
    if ok: passers[k] = d.mean()
if passers:
    best = max(passers, key=passers.get); print("Best passer:", best, flush=True)
    cand, _ = run(35, lab, unkall, test)
    out = pd.DataFrame({"index": test["index"], "target": expit(cand[best])}); validate_submission(out, test, sample)
    p = ROOT / f"submission_{best}_pu_comp.csv"; out.to_csv(p, index=False); print("saved", p)
    print("Spearman vs v10 (0.95083):", round(out.target.corr(pd.read_csv(ROOT / "submission_v10.csv").target, method="spearman"), 4))
else:
    print("No candidate passes: do not submit.")
