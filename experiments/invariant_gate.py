"""Invariant feature-BLOCK selection by step-group-held-out CV (training labels only), then outer chronological evaluation.
Blocks: 15 contiguous blocks of 11 base columns + 1 graph block. Backward elimination, objective = 0.5*mean + 0.5*min fold AUC
over K=5 contiguous step groups (train on other groups, test on held-out group). <=5 rounds, accept removal only if objective +0.002.
Outer cutoffs 10,14,17,21. Candidates (declared before running), reference = V2 + last71 (public 0.95083):
  J1 LGBM(S) alone | J2 (LGBM(S) + l71)/2 | J3 (V2 + l71 + LGBM(S))/3
FROZEN RULE: mean gain >= +0.002 AND better on >= 3 of 4 cutoffs. File only for the single best passer.  Resumable.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from lightgbm import LGBMClassifier
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
from pipeline import BASE, load_data, graph_features, validate_submission

ROOT = Path(__file__).resolve().parent.parent; HERE = Path(__file__).resolve().parent
SEL = HERE / "invariant_selection.json"; RES = HERE / "invariant_results.csv"
P = dict(n_estimators=300, learning_rate=.03, num_leaves=31, random_state=42, verbosity=-1, n_jobs=4)
SMALL = dict(n_estimators=80, learning_rate=.1, num_leaves=15, random_state=0, verbosity=-1, n_jobs=4, colsample_bytree=.8, max_bin=63)
L = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6)); L71 = BASE[94:]

train, test, edges, sample = load_data(ROOT); train, test, g, _ = graph_features(train, test, edges)
lab = train.loc[train.label.notna()]
blocks = [BASE[i:i + 11] for i in range(0, 165, 11)] + [g]
sel_cache = json.loads(SEL.read_text()) if SEL.exists() else {}
sel_cache = {k: v for k, v in sel_cache.items() if isinstance(v, dict)}

def objective(df, folds, cols):
    aucs = []
    for tr_i, te_i in folds:
        t, v = df.iloc[tr_i], df.iloc[te_i]
        m = LGBMClassifier(**SMALL).fit(t[cols], t.label.astype(int))
        aucs.append(roc_auc_score(v.label.astype(int), m.predict_proba(v[cols])[:, 1]))
    return 0.5 * np.mean(aucs) + 0.5 * np.min(aucs)

def select(c):
    st = sel_cache.get(str(c))
    if st and st["done"]: return st["keep"]
    df = lab.loc[lab.time_step <= c].reset_index(drop=True)
    steps = np.array_split(np.sort(df.time_step.unique()), 5); folds = []
    for sg in steps:
        te = np.where(df.time_step.isin(sg))[0]; tr = np.where(~df.time_step.isin(sg))[0]
        if df.label.iloc[te].nunique() == 2 and df.label.iloc[tr].nunique() == 2: folds.append((tr, te))
    if st: keep, cur, rnd0 = st["keep"], st["cur"], st["rnd"]
    else:
        keep = list(range(len(blocks))); cur = objective(df, folds, sum([blocks[i] for i in keep], [])); rnd0 = 0
    t0 = time.time(); print(f"cutoff {c}: resume at round {rnd0}, objective {cur:.4f}", flush=True)
    for rnd in range(rnd0, 5):
        best = (cur, None)
        for b in keep:
            sc = objective(df, folds, sum([blocks[i] for i in keep if i != b], []))
            if sc > best[0]: best = (sc, b)
        if best[1] is None or best[0] < cur + 0.002:
            sel_cache[str(c)] = {"keep": keep, "cur": cur, "rnd": rnd, "done": True}; SEL.write_text(json.dumps(sel_cache)); return keep
        keep = [i for i in keep if i != best[1]]; cur = best[0]
        sel_cache[str(c)] = {"keep": keep, "cur": cur, "rnd": rnd + 1, "done": rnd + 1 >= 5}; SEL.write_text(json.dumps(sel_cache))
        print(f"  round {rnd}: drop block {best[1]} -> {cur:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return keep

def fp(tr, V, cols):
    m = LGBMClassifier(**P).fit(tr[cols], tr.label.astype(int)); return L(m.predict_proba(V[cols])[:, list(m.classes_).index(1)])

def cands(tr, V, keep):
    S = sum([blocks[i] for i in keep], []); v2, l71, s = fp(tr, V, BASE + g), fp(tr, V, L71), fp(tr, V, S)
    return {"ref": (v2 + l71) / 2, "J1": s, "J2": (s + l71) / 2, "J3": (v2 + l71 + s) / 3}

done = pd.read_csv(RES) if RES.exists() else pd.DataFrame()
for c in [10, 14, 17, 21]:
    if len(done) and c in set(done.cutoff): continue
    keep = select(c)
    tr = lab.loc[lab.time_step <= c]; va = lab.loc[lab.time_step.between(c + 1, min(c + 14, 35))]; y = va.label.astype(int).to_numpy()
    row = {"cutoff": c, "dropped_blocks": str([i for i in range(len(blocks)) if i not in keep]), **{k: roc_auc_score(y, v) for k, v in cands(tr, va, keep).items()}}
    done = pd.concat([done, pd.DataFrame([row])]); done.to_csv(RES, index=False); print("outer done", c, flush=True)
R = done.set_index("cutoff").sort_index(); print("\n", R.drop(columns="dropped_blocks").round(4).to_string()); print(R.dropped_blocks.to_string())
passers = {}
for k in ["J1", "J2", "J3"]:
    d = R[k] - R.ref; ok = d.mean() >= 0.002 and (d > 0).sum() >= 3
    print(f"{k}: mean gain {d.mean():+.4f} | better on {(d > 0).sum()}/4 | PASS {ok}")
    if ok: passers[k] = d.mean()
if passers:
    best = max(passers, key=passers.get); print("Best passer:", best, flush=True)
    keep = select(35); out = pd.DataFrame({"index": test["index"], "target": expit(cands(lab, test, keep)[best])}); validate_submission(out, test, sample)
    p = ROOT / f"submission_{best}_invariant.csv"; out.to_csv(p, index=False); print("saved", p, "dropped blocks at 35:", [i for i in range(len(blocks)) if i not in keep])
    print("Spearman vs v10 (0.95083):", round(out.target.corr(pd.read_csv(ROOT / "submission_v10.csv").target, method="spearman"), 4))
else:
    print("No candidate passes: do not submit.")
