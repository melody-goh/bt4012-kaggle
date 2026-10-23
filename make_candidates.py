"""Builds three candidate blends from files you already have. Run next to: submission_v2.csv, submission_v3.csv,
submission_v33_fine_grained.csv, submission_blend_v2v3_logit.csv (the 0.95044 file), test.csv.
No retraining, no labels. Equal-weight logit averages, fixed in advance.

Tonight's pick:  submission_blend_v33v3.csv   (V2 swapped for its sibling V33 fine_grained, which had the best
                 rolling-origin mean in your notebook: 0.9376 vs 0.9324 for V2).
Later options:   submission_blend_v2v3v33.csv  and  submission_blend_v2v3_60_40.csv
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

v2 = pd.read_csv("submission_v2.csv")
v3 = pd.read_csv("submission_v3.csv")
v33 = pd.read_csv("submission_v8.csv")
best = pd.read_csv("submission_v7.csv")
test = pd.read_csv("test.csv", usecols=["index"])
for name, d in [("v3", v3), ("v33", v33), ("best", best)]:
    assert v2["index"].equals(d["index"]), f"index mismatch: {name}"
assert v2["index"].equals(test["index"])

L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
z2, z3, z33 = L(v2.target.values), L(v3.target.values), L(v33.target.values)

cands = {
    "submission_blend_v33v3.csv": (z33 + z3) / 2,
    "submission_blend_v2v3v33.csv": (z2 + z3 + z33) / 3,
    "submission_blend_v2v3_60_40.csv": 0.6 * z2 + 0.4 * z3,
}
for fname, z in cands.items():
    out = pd.DataFrame({"index": v2["index"], "target": expit(z)})
    assert out.target.between(0, 1).all() and out.target.notna().all() and len(out) == len(test)
    out.to_csv(fname, index=False)
    print(f"{fname}: Spearman vs 0.95044 file = {out.target.corr(best.target, method='spearman'):.4f}")
print("Spearman V33 vs V2:", round(v33.target.corr(v2.target, method="spearman"), 4))
print("Sanity: each Spearman vs the 0.95044 file should be > 0.9.")