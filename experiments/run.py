"""Run from any directory; outputs are immutable per run ID."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import numpy as np
import pandas as pd
from pipeline import BASE, load_data, graph_features, split, fit_predict, metrics, sha256, validate_submission

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['reproduce-v2', 'evaluate', 'submit'])
    parser.add_argument('--data-dir', type=Path, default=HERE.parent)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/initial.json')
    parser.add_argument('--role', choices=['development', 'historical_check', 'legacy'], default='development')
    parser.add_argument('--model', default='v7_graph')
    parser.add_argument('--run-id', default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if Path(run_id).name != run_id or run_id in {'.', '..'}:
        parser.error('run-id must be a single directory name')
    out = HERE / 'runs' / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'command': vars(args) | {'data_dir': str(args.data_dir.resolve()), 'config': str(args.config.resolve())},
                'config': config, 'python': sys.version, 'platform': platform.platform(),
                'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm']},
                'source_hashes': {p.name: sha256(p) for p in [HERE/'run.py', HERE/'pipeline.py', args.config]},
                'inputs': {name: sha256(args.data_dir/name) for name in
                           ['train.csv', 'test.csv', 'txs_edgelist.csv', 'sample_submission.csv']}}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    train, test, edges, sample = load_data(args.data_dir)
    train, test, graph_cols, audit = graph_features(train, test, edges)
    audit.update(train_rows=len(train), test_rows=len(test), labelled_rows=int(train.label.notna().sum()),
                 unknown_rows=int(train.label.isna().sum()), illicit_rows=int((train.label == 1).sum()))
    for path in args.data_dir.glob('submission*.csv'):
        validate_submission(pd.read_csv(path), test, sample)
    (out/'audit.json').write_text(json.dumps(audit, indent=2))
    train.groupby('time_step').label.agg(['size', 'count', 'sum', 'mean']).to_csv(out/'label_coverage.csv')
    if args.command == 'evaluate':
        results, by_step = [], []
        for fold in config['folds']:
            if fold['role'] != args.role:
                continue
            tr, va = split(train, fold)
            for name, model_config in config['models'].items():
                cols = BASE + (graph_cols if model_config['graph'] else [])
                params = config['common_params'] | model_config['params']
                model, pred = fit_predict(tr, va, cols, params)
                result = {'fold': fold['name'], 'role': fold['role'], 'model': name, **metrics(va.label, pred)}
                results.append(result)
                prediction = va[['txId', 'time_step', 'label']].copy()
                prediction['probability'] = pred
                prediction.to_csv(out/f"{fold['name']}__{name}__predictions.csv", index=False)
                model.booster_.save_model(str(out/f"{fold['name']}__{name}.txt"))
                for step, group in prediction.groupby('time_step'):
                    by_step.append({'fold': fold['name'], 'model': name, 'time_step': int(step),
                                    **metrics(group.label, group.probability)})
                print(json.dumps(result), flush=True)
        if not results:
            raise ValueError('No folds match the requested role')
        pd.DataFrame(results).to_csv(out/'metrics.csv', index=False)
        pd.DataFrame(by_step).to_csv(out/'metrics_by_step.csv', index=False)
    else:
        name = 'v7_graph' if args.command == 'reproduce-v2' else args.model
        spec = config['models'][name]
        cols = BASE + (graph_cols if spec['graph'] else [])
        params = config['common_params'] | spec['params']
        model, pred = fit_predict(train.dropna(subset=['label']), test, cols, params)
        submission = pd.DataFrame({'index': test['index'], 'target': pred})
        validate_submission(submission, test, sample)
        submission.to_csv(out/'submission.csv', index=False)
        validate_submission(pd.read_csv(out/'submission.csv'), test, sample)
        model.booster_.save_model(str(out/'model.txt'))
        (out/'features.json').write_text(json.dumps(cols, indent=2))
        if args.command == 'reproduce-v2':
            reference = pd.read_csv(args.data_dir/'submission_v2.csv')
            validate_submission(reference, test, sample)
            difference = np.abs(reference.target.to_numpy() - pred)
            comparison = {'reference_sha256': sha256(args.data_dir/'submission_v2.csv'),
                          'max_absolute_difference': float(difference.max()),
                          'mean_absolute_difference': float(difference.mean()),
                          'matches_at_1e_10': bool(np.allclose(reference.target, pred, atol=1e-10, rtol=0)),
                          'spearman_rank_correlation': float(reference.target.corr(pd.Series(pred), method='spearman'))}
            (out/'reproduction.json').write_text(json.dumps(comparison, indent=2))
            print(json.dumps(comparison), flush=True)
    (out/'COMPLETE').write_text('Completed successfully\n')
    print(f'Artifacts: {out}', flush=True)


if __name__ == '__main__':
    main()
