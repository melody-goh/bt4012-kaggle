"""Gate for a V2 + last71 logit blend vs the submitted V2 + V3 blend.
Cutoffs 7,10,14,17,21. FROZEN RULE: V2+last71 beats V2+V3 by mean >= +0.001 AND on >= 3 of 5 cutoffs.
Writes submission_blend_v2last71.csv (in the parent folder) only if the rule passes (or --force). Train labels only.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PARAMS = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))


def fit_pred(tr, va, cols, w=None):
    m = LGBMClassifier(**PARAMS).fit(tr[cols], tr.label.astype(int), sample_weight=w)
    return m.predict_proba(va[cols])[:, list(m.classes_).index(1)]


def comps(tr, va, gcols, cutoff):
    w = 2. ** (-(cutoff - tr.time_step.to_numpy()) / 28)
    w /= w.mean()
    p = {"v2": fit_pred(tr, va, BASE + gcols), "first94": fit_pred(tr, va, BASE[:94]),
         "last71": fit_pred(tr, va, BASE[94:]), "weighted": fit_pred(tr, va, BASE + gcols, w)}
    p["v3"] = .25 * p["first94"] + .25 * p["last71"] + .5 * p["weighted"]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    train, test, edges, sample = load_data(ROOT)
    train, test, gcols, _ = graph_features(train, test, edges)
    lab = train.loc[train.label.notna()]
    rows = []
    for c in [7, 10, 14, 17, 21]:
        tr = lab.loc[lab.time_step <= c]
        va = lab.loc[lab.time_step.between(c + 1, min(c + 14, 35))]
        p = comps(tr, va, gcols, c)
        y = va.label.astype(int).to_numpy()
        z = {k: L(v) for k, v in p.items()}
        rows.append(dict(cutoff=c,
                         v2_v3=roc_auc_score(y, (z["v2"] + z["v3"]) / 2),
                         v2_last71=roc_auc_score(y, (z["v2"] + z["last71"]) / 2),
                         v2_v3_last71=roc_auc_score(y, (z["v2"] + z["v3"] + z["last71"]) / 3)))
        print("done cutoff", c, flush=True)
    R = pd.DataFrame(rows).set_index("cutoff")
    g = R.v2_last71 - R.v2_v3
    ok = g.mean() >= 0.001 and (g > 0).sum() >= 3
    print("\n", R.round(4).to_string())
    print(f"\nV2+last71 vs V2+V3: mean gain {g.mean():+.4f} | better on {(g > 0).sum()}/5 | GATE PASS: {ok}", flush=True)
    if not (ok or args.force):
        print("Gate failed: no file written. Do not spend a submission on this.")
        return
    final = comps(lab, test, gcols, 35)
    out = pd.DataFrame({"index": test["index"], "target": expit((L(final["v2"]) + L(final["last71"])) / 2)})
    validate_submission(out, test, sample)
    path = ROOT / "submission_blend_v2last71.csv"
    out.to_csv(path, index=False)
    print("saved", path)
    best = ROOT / "BEST_0.95044.csv"
    if best.exists():
        print("Spearman vs the 0.95044 file:", round(out.target.corr(pd.read_csv(best).target, method="spearman"), 4))


if __name__ == "__main__":
    main()
