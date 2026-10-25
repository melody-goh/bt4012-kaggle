"""Stage 2 of 2 (run with Python 3.13, requirements.txt): the steps originally run in that environment.

Reads stage 1's components and replays the original order of operations, including the CSV
round-trips between steps, so the final files are identical to the submitted ones:
  v7  = logit avg(v2, v3)
  Y2  = logit avg(v12, R15), R15 = D recipe trained on steps >= 15; R15 is then recovered from Y2
  v20 = logit avg(v12, v16_new, R15, v7)
  v22 = v20 with logit + 0.3 for test steps 43-49
Writes <out-dir>/submission_v20.csv and <out-dir>/submission_v22.csv. --check compares them with the
submitted files in the repo root.
"""
import argparse, filecmp
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from pipeline import BASE, load_data, graph_features, validate_submission

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default=ROOT / "data", type=Path)
ap.add_argument("--out-dir", default=ROOT / "outputs", type=Path)
ap.add_argument("--check", action="store_true")
args = ap.parse_args()

L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
train, test, edges, sample = load_data(args.data_dir); train, test, g, _ = graph_features(train, test, edges)
lab = train[train.label.notna()]; comp = args.out_dir / "components"
def fp(X, cols, w=None): return L(LGBMClassifier(**P).fit(X[cols], X.label.astype(int), sample_weight=w).predict_proba(test[cols])[:, 1])
def W(X, hl): w = 2. ** (-(35 - X.time_step.to_numpy()) / hl); return w / w.mean()
def R(k): rec = lab[lab.time_step >= k]; return (fp(rec, BASE + g, W(rec, 28)) + fp(rec, L71)) / 2
def read(path): return pd.read_csv(path).target
def save(p, path):
    out = pd.DataFrame({"index": test["index"], "target": p}); validate_submission(out, test, sample)
    out.to_csv(path, index=False); return path
Lcsv = lambda path: logit(read(path).clip(1e-6, 1 - 1e-6).to_numpy())

v2, v3 = read(comp / "v2.csv").values, read(comp / "v3.csv").values
save(expit((L(v2) + L(v3)) / 2), comp / "v7.csv")
save(expit((L(read(comp / "v12.csv").to_numpy()) + R(15)) / 2), comp / "Y2.csv")

a, b, c = Lcsv(comp / "v12.csv"), Lcsv(comp / "v16_new.csv"), Lcsv(comp / "v7.csv")
r15 = 2 * Lcsv(comp / "Y2.csv") - a
v20 = save(expit((a + b + r15 + c) / 4), args.out_dir / "submission_v20.csv")
late = test.time_step.to_numpy() >= 43
v22 = save(expit(Lcsv(v20) + .3 * late), args.out_dir / "submission_v22.csv")

for path in (v20, v22):
    print("saved", path)
    if args.check and (ROOT / path.name).exists():
        new, ref = read(path), read(ROOT / path.name)
        print(f"  vs submitted {path.name}: byte-identical {filecmp.cmp(path, ROOT / path.name, shallow=False)}, "
              f"ranking identical {(new.rank(method='min') == ref.rank(method='min')).all()}, max |diff| {np.abs(new - ref).max():.1e}")
