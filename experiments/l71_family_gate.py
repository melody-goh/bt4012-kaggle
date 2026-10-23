"""last71-family variants vs the new reference V2+last71 (public 0.95083). Cutoffs 7,10,14,17,21. Train labels only.
Candidates (declared before running, 4 of them):
  A v2_l71g    : V2 + LGBM(last71 + graph feats)
  B v2_l71w    : V2 + LGBM(last71, recency weights half-life 28)
  C v2_l71_l71g: V2 + last71 + last71g (3-way)
  D weighted_l71: recency-weighted graph model + last71
FROZEN RULE (stricter because 4 candidates): mean gain over V2+last71 >= +0.0015 AND better on >= 4 of 5 cutoffs.
Writes a file only for the single best passing candidate.
"""
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pathlib import Path
from pipeline import BASE, load_data, graph_features, validate_submission

ROOT = Path(__file__).resolve().parent.parent
PARAMS = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
L71 = BASE[94:]

def fp(tr, va, cols, w=None):
    m = LGBMClassifier(**PARAMS).fit(tr[cols], tr.label.astype(int), sample_weight=w)
    return m.predict_proba(va[cols])[:, list(m.classes_).index(1)]

def comps(tr, va, g, cutoff):
    w = 2. ** (-(cutoff - tr.time_step.to_numpy()) / 28); w /= w.mean()
    z = {"v2": fp(tr, va, BASE + g), "l71": fp(tr, va, L71), "l71g": fp(tr, va, L71 + g),
         "l71w": fp(tr, va, L71, w), "weighted": fp(tr, va, BASE + g, w)}
    z = {k: L(v) for k, v in z.items()}
    return {"ref_v2_l71": (z["v2"] + z["l71"]) / 2,
            "A_v2_l71g": (z["v2"] + z["l71g"]) / 2,
            "B_v2_l71w": (z["v2"] + z["l71w"]) / 2,
            "C_v2_l71_l71g": (z["v2"] + z["l71"] + z["l71g"]) / 3,
            "D_weighted_l71": (z["weighted"] + z["l71"]) / 2}

train, test, edges, sample = load_data(ROOT)
train, test, g, _ = graph_features(train, test, edges)
lab = train.loc[train.label.notna()]
rows = []
for c in [7, 10, 14, 17, 21]:
    tr = lab.loc[lab.time_step <= c]; va = lab.loc[lab.time_step.between(c + 1, min(c + 14, 35))]
    y = va.label.astype(int).to_numpy()
    rows.append({"cutoff": c, **{k: roc_auc_score(y, v) for k, v in comps(tr, va, g, c).items()}})
    print("done", c, flush=True)
R = pd.DataFrame(rows).set_index("cutoff")
print("\n", R.round(4).to_string())
passers = {}
for k in R.columns[1:]:
    d = R[k] - R.ref_v2_l71
    ok = d.mean() >= 0.0015 and (d > 0).sum() >= 4
    print(f"{k}: mean gain {d.mean():+.4f} | better on {(d > 0).sum()}/5 | PASS {ok}")
    if ok: passers[k] = d.mean()
if passers:
    best = max(passers, key=passers.get); print("\nBest passer:", best)
    final = comps(lab, test, g, 35)[best]
    out = pd.DataFrame({"index": test["index"], "target": expit(final)}); validate_submission(out, test, sample)
    p = ROOT / f"submission_{best}.csv"; out.to_csv(p, index=False); print("saved", p)
    ref = pd.read_csv(ROOT / "submission_blend_v2last71.csv")
    print("Spearman vs the 0.95083 file:", round(out.target.corr(ref.target, method="spearman"), 4))
else:
    print("\nNo candidate passes: do not submit any of them.")
