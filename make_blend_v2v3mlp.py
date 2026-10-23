"""Equal-weight logit blend: submission_v2 + submission_v3 + cand_mlp_test.
Run next to those files and test.csv. Writes submission_blend_v2v3mlp.csv. No retraining, no labels.
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

v2 = pd.read_csv("submission_v2.csv")
v3 = pd.read_csv("submission_v3.csv")
mlp = pd.read_csv("cand_mlp_test.csv")
prev = pd.read_csv("submission_v7.csv")  # the 0.95044 file
test = pd.read_csv("test.csv", usecols=["index"])
for name, d in [("v3", v3), ("mlp", mlp), ("prev", prev)]:
    assert v2["index"].equals(d["index"]), f"index mismatch: {name}"
assert v2["index"].equals(test["index"])

L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
out = pd.DataFrame({"index": v2["index"],
                    "target": expit((L(v2.target.values) + L(v3.target.values) + L(mlp.target.values)) / 3)})
assert out.target.between(0, 1).all() and out.target.notna().all() and len(out) == len(test)
out.to_csv("submission_blend_v2v3mlp.csv", index=False)
print("saved", len(out), "rows")
print("Spearman of MLP vs the 0.95044 blend:", round(mlp.target.corr(prev.target, method="spearman"), 4))
print("Spearman of new blend vs the 0.95044 blend:", round(out.target.corr(prev.target, method="spearman"), 4))