"""Equal-weight logit blend of V2 and V3 submissions. Run next to submission_v2.csv and submission_v3.csv.
No retraining, no labels. Writes submission_blend_v2v3_logit.csv.
"""
import numpy as np
import pandas as pd
from scipy.special import logit, expit

v2 = pd.read_csv("submission_v2.csv")
v3 = pd.read_csv("submission_v3.csv")
test = pd.read_csv("test.csv", usecols=["index"])
assert v2["index"].equals(v3["index"]) and v2["index"].equals(test["index"]), "index mismatch"

def L(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))

out = pd.DataFrame({"index": v2["index"], "target": expit((L(v2.target.values) + L(v3.target.values)) / 2)})
assert out.target.between(0, 1).all() and out.target.notna().all() and len(out) == len(test)
out.to_csv("submission_blend_v2v3_logit.csv", index=False)
print("saved", len(out), "rows")
print("Spearman vs V2:", round(out.target.corr(v2.target, method="spearman"), 4))
print("Spearman vs V3:", round(out.target.corr(v3.target, method="spearman"), 4))