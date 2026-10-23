"""Compare the three scored submissions and their reconstructed historical models.

Run with the original Python 3.10 environment for exact saved-file provenance.
This is retrospective diagnosis, not untouched model evaluation.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline import BASE, sha256, load_data, graph_features, fit_predict, metrics, validate_submission

HERE = Path(__file__).resolve().parent
PARAMS = dict(n_estimators=300, learning_rate=.03, num_leaves=31,
              random_state=42, verbosity=-1, n_jobs=4)
SCORED = {'v1': ('submission_lgbm_v1.csv', .94719), 'v2': ('submission_v2.csv', .94785),
          'v3': ('submission_v3.csv', .94510)}


def train_components(tr, va, graph_cols, cutoff):
    predictions = {}
    for name, cols in [('v1', BASE), ('v2', BASE+graph_cols), ('first94', BASE[:94]), ('last71', BASE[94:])]:
        _, predictions[name] = fit_predict(tr, va, cols, PARAMS)
    from lightgbm import LGBMClassifier
    weights = 2.**(-(cutoff-tr.time_step.to_numpy())/28)
    weights /= weights.mean()
    model = LGBMClassifier(**PARAMS)
    model.fit(tr[BASE+graph_cols], tr.label.astype(int), sample_weight=weights)
    predictions['weighted_graph'] = model.predict_proba(va[BASE+graph_cols])[:, list(model.classes_).index(1)]
    predictions['v3'] = .25*predictions['first94']+.25*predictions['last71']+.5*predictions['weighted_graph']
    return predictions


def pair_decomposition(y, p, steps):
    """Exact pooled AUC decomposition into within-step and cross-step pairs."""
    total_pairs = int((y == 1).sum()) * int((y == 0).sum())
    total_wins = roc_auc_score(y, p)*total_pairs
    within_pairs = 0
    within_wins = 0.
    for step in np.unique(steps):
        mask = steps == step
        positives, negatives = int((y[mask] == 1).sum()), int((y[mask] == 0).sum())
        pairs = positives*negatives
        if pairs:
            within_pairs += pairs
            within_wins += roc_auc_score(y[mask], p[mask])*pairs
    return {'within_step_pair_share': within_pairs/total_pairs,
            'within_step_pair_auc': within_wins/within_pairs if within_pairs else np.nan,
            'cross_step_pair_auc': (total_wins-within_wins)/(total_pairs-within_pairs) if total_pairs != within_pairs else np.nan}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default=datetime.now(timezone.utc).strftime('diagnosis_%Y%m%dT%H%M%S%fZ'))
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {'.', '..'}:
        parser.error('run-id must be a single directory name')
    out = HERE/'runs'/args.run_id
    out.mkdir(parents=True, exist_ok=False)
    root = HERE.parent
    manifest = {'source': 'Scores supplied in handover and confirmed by user; no Kaggle API access.',
                'scored_files': {k: {'filename': f, 'public_auc': score, 'sha256': sha256(root/f)} for k,(f,score) in SCORED.items()},
                'parameters': PARAMS, 'origins': [7, 14, 21], 'horizon': 14,
                'source_hashes': {p.name: sha256(p) for p in [Path(__file__),HERE/'pipeline.py']},
                'input_hashes': {f: sha256(root/f) for f in ['train.csv','test.csv','txs_edgelist.csv','sample_submission.csv']},
                'python': sys.version, 'versions': {p: importlib.metadata.version(p) for p in ['numpy','pandas','lightgbm','scikit-learn']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    train,test,edges,sample = load_data(root)
    train,test,graph_cols,audit = graph_features(train,test,edges)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    submitted = test[['index','txId','time_step']].copy()
    for name,(file,score) in SCORED.items():
        frame = pd.read_csv(root/file)
        validate_submission(frame,test,sample)
        submitted[name] = frame.target.to_numpy()
    submitted[['v1','v2','v3']].corr(method='spearman').to_csv(out/'submission_rank_correlations.csv')
    deltas=[]
    for step, group in submitted.groupby('time_step'):
        ranks = group[['v1','v2','v3']].rank(pct=True)
        deltas.append({'time_step':step,'rows':len(group),
                       'v1_v2_rank_corr':group.v1.corr(group.v2,method='spearman'),
                       'v2_v3_rank_corr':group.v2.corr(group.v3,method='spearman'),
                       'mean_abs_v2_v3_probability_change':float((group.v3-group.v2).abs().mean()),
                       'mean_abs_v2_v3_percentile_change':float((ranks.v3-ranks.v2).abs().mean()),
                       **{f'{name}_mean_probability':group[name].mean() for name in ['v1','v2','v3']}})
    pd.DataFrame(deltas).to_csv(out/'submission_changes_by_step.csv',index=False)
    results, by_step, decompositions, comparisons = [], [], [], []
    for cutoff in [7,14,21,35]:
        tr = train.loc[train.label.notna() & (train.time_step <= cutoff)]
        va = test if cutoff == 35 else train.loc[train.label.notna() & train.time_step.between(cutoff+1,cutoff+14)]
        preds = train_components(tr,va,graph_cols,cutoff)
        frame = va[['txId','time_step']].copy()
        if cutoff != 35:
            frame['label'] = va.label
        for name,p in preds.items():
            frame[name] = p
            if cutoff == 35:
                if name in SCORED:
                    reference=submitted[name].to_numpy()
                    comparisons.append({'model':name,'max_absolute_difference':float(np.max(np.abs(p-reference))),
                                        'mean_absolute_difference':float(np.mean(np.abs(p-reference))),
                                        'matches_at_1e_10':bool(np.allclose(p,reference,rtol=0,atol=1e-10))})
            else:
                results.append({'train_end':cutoff,'model':name,**metrics(va.label,p)})
                decompositions.append({'train_end':cutoff,'model':name,**pair_decomposition(va.label.to_numpy(),p,va.time_step.to_numpy())})
                for step in sorted(va.time_step.unique()):
                    mask=va.time_step.to_numpy()==step
                    by_step.append({'train_end':cutoff,'time_step':step,'model':name,**metrics(va.loc[mask,'label'],p[mask])})
        frame.to_csv(out/f'predictions_cutoff_{cutoff}.csv',index=False)
        print('Completed cutoff',cutoff,flush=True)
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(by_step).to_csv(out/'metrics_by_step.csv',index=False)
    pd.DataFrame(decompositions).to_csv(out/'auc_pair_decomposition.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'submission_reproduction.csv',index=False)
    # A small leaderboard ledger: no inferred score for unsubmitted candidates.
    pd.DataFrame([{'model':name,'filename':file,'public_roc_auc':score,'score_source':'user-confirmed',
                   'sha256':sha256(root/file)} for name,(file,score) in SCORED.items()]).to_csv(out/'leaderboard.csv',index=False)
    (out/'COMPLETE').write_text('Completed successfully\n')
    print(pd.DataFrame(results).pivot(index='model',columns='train_end',values='roc_auc').round(6).to_string())
    print(pd.DataFrame(comparisons).to_string(index=False))
    print('Artifacts:',out,flush=True)


if __name__ == '__main__':
    main()
