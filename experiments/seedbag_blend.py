"""Seed-bagged V2 + V3 blend, gated offline. Put this file in your experiments/ folder (next to pipeline.py); data CSVs in the parent folder.
Use the original Python 3.10 environment (requirements-original.txt) so the reference matches your saved files.

Why sampling: V2/V3 use no row/feature sampling, so seeds alone give identical models. Bagged models use
subsample=.9, subsample_freq=1, colsample_bytree=.9 (same as the earlier graph_bagging test), 5 seeds each.
V3 = .25*first94 + .25*last71 + .5*recency-weighted graph model (half-life 28), exactly as in diagnose_submissions.py.

Gate (frozen): bagged blend must beat the single-model V2+V3 logit blend on >= 2 of 3 cutoffs (7, 14, 21)
AND mean gain >= +0.001. The final file is written only if the gate passes (or with --force).
Only train labels are used.

  python seedbag_blend.py
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
SEEDS = [42, 7, 13, 99, 2026]
REF = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
BAG = dict(REF, subsample=.9, subsample_freq=1, colsample_bytree=.9)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))


def fit_pred(tr, va, cols, params, w=None):
    m = LGBMClassifier(**params).fit(tr[cols], tr.label.astype(int), sample_weight=w)
    return m.predict_proba(va[cols])[:, list(m.classes_).index(1)]


def components(tr, va, gcols, cutoff, params, seeds):
    w = 2. ** (-(cutoff - tr.time_step.to_numpy()) / 28)
    w /= w.mean()
    out = {k: [] for k in ["v2", "first94", "last71", "weighted"]}
    for s in seeds:
        p = dict(params, random_state=s)
        out["v2"].append(fit_pred(tr, va, BASE + gcols, p))
        out["first94"].append(fit_pred(tr, va, BASE[:94], p))
        out["last71"].append(fit_pred(tr, va, BASE[94:], p))
        out["weighted"].append(fit_pred(tr, va, BASE + gcols, p, w))
    m = {k: np.mean(v, axis=0) for k, v in out.items()}
    m["v3"] = .25 * m["first94"] + .25 * m["last71"] + .5 * m["weighted"]
    return m


def blend(m):
    return expit((L(m["v2"]) + L(m["v3"])) / 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="write the final file even if the gate fails")
    args = ap.parse_args()
    train, test, edges, sample = load_data(ROOT)
    train, test, gcols, _ = graph_features(train, test, edges)
    lab = train.loc[train.label.notna()]
    rows = []
    for c in [7, 14, 21]:
        tr = lab.loc[lab.time_step <= c]
        va = lab.loc[lab.time_step.between(c + 1, c + 14)]
        ref = components(tr, va, gcols, c, REF, [42])
        bag = components(tr, va, gcols, c, BAG, SEEDS)
        y = va.label.astype(int).to_numpy()
        rows.append(dict(cutoff=c, ref_v2v3=roc_auc_score(y, blend(ref)), bag_v2v3=roc_auc_score(y, blend(bag)),
                         ref_v2=roc_auc_score(y, ref["v2"]), bag_v2=roc_auc_score(y, bag["v2"]),
                         ref_v3=roc_auc_score(y, ref["v3"]), bag_v3=roc_auc_score(y, bag["v3"])))
        print("done cutoff", c, flush=True)
    R = pd.DataFrame(rows).set_index("cutoff")
    R["gain"] = R.bag_v2v3 - R.ref_v2v3
    print("\n", R.round(4).to_string())
    ok = (R.gain > 0).sum() >= 2 and R.gain.mean() >= 0.001
    print(f"\nmean gain {R.gain.mean():+.4f} | better on {(R.gain > 0).sum()}/3 | GATE PASS: {ok}")
    if not (ok or args.force):
        print("Gate failed: no file written. Do not spend a submission on this.")
        return
    final = components(lab, test, gcols, 35, BAG, SEEDS)
    out = pd.DataFrame({"index": test["index"], "target": blend(final)})
    validate_submission(out, test, sample)
    path = ROOT / "submission_bag_v2v3.csv"
    out.to_csv(path, index=False)
    print("saved", path)
    best = ROOT / "submission_blend_v2v3_logit.csv"
    if best.exists():
        prev = pd.read_csv(best)
        print("Spearman vs the 0.95044 file:", round(out.target.corr(prev.target, method="spearman"), 4))


if __name__ == "__main__":
    main()