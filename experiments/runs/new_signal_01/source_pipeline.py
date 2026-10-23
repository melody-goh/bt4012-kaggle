"""Pure data/model helpers; never read or write the original notebooks."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = [f'feat_{i}' for i in range(1, 166)]
AGG = ['feat_2', 'feat_53', 'feat_58', 'feat_16', 'feat_90', 'feat_138', 'feat_142', 'feat_163']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_submission(frame, test, sample):
    require(list(frame.columns) == ['index', 'target'], 'Submission columns must be index,target')
    require(len(frame) == len(test) == len(sample), 'Submission row count mismatch')
    require(frame['index'].is_unique and test['index'].is_unique, 'Duplicate submission index')
    require(np.array_equal(frame['index'], test['index']), 'Test ordering mismatch')
    require(np.array_equal(frame['index'], sample['index']), 'Sample ordering mismatch')
    require(np.isfinite(frame.target).all() and frame.target.between(0, 1).all(), 'Invalid probabilities')


def load_data(root):
    train, test, edges, sample = [pd.read_csv(Path(root) / f) for f in
                                ['train.csv', 'test.csv', 'txs_edgelist.csv', 'sample_submission.csv']]
    for frame in [train, test]:
        require(frame.txId.notna().all() and frame.txId.is_unique, 'Missing/duplicate txId')
        require(set(BASE + ['time_step']).issubset(frame.columns), 'Missing features')
        require(np.isfinite(frame[BASE].to_numpy()).all(), 'Nonfinite original features')
        require(frame.time_step.notna().all(), 'Missing time step')
    require(not set(train.txId) & set(test.txId), 'Train/test ID overlap')
    require(set(train.label.dropna().unique()) == {0, 1}, 'Unexpected labels')
    require(train.time_step.max() < test.time_step.min(), 'Train/test time overlap')
    require(edges[['txId1', 'txId2']].notna().all().all(), 'Missing edge endpoint')
    validate_submission(pd.DataFrame({'index': test['index'], 'target': 0.5}), test, sample)
    return train, test, edges, sample


def graph_features(train, test, edges):
    """Exact V7: all-edge degree, available-source mean/sample std, no labels.

    Full released graph and same-step covariates are assumed available. Fail if
    inspectable edges cross time steps; this would require a cutoff-aware design.
    """
    nodes = pd.concat([train[['txId', 'time_step'] + AGG], test[['txId', 'time_step'] + AGG]], ignore_index=True)
    require(nodes.txId.is_unique, 'Duplicate nodes')
    steps = nodes.set_index('txId').time_step
    src, dst = edges.txId1.map(steps), edges.txId2.map(steps)
    known = src.notna() & dst.notna()
    require((src[known] == dst[known]).all(), 'Cross-step edges require cutoff-aware graph features')
    graph = edges[['txId1', 'txId2']].merge(nodes.rename(columns={
        'txId': 'txId1', 'time_step': 'source_step', **{c: f'source_{c}' for c in AGG}
    }), on='txId1', how='left', validate='many_to_one')
    grouped = graph.groupby('txId2')
    agg = grouped[[f'source_{c}' for c in AGG]].agg(['mean', 'std'])
    agg.columns = [f'in_{c}_{stat}' for c, stat in agg.columns]
    agg['in_degree'] = grouped.size()
    agg['in_known_sources'] = grouped.source_step.count()
    agg = agg.reset_index().rename(columns={'txId2': 'txId'})
    cols = list(agg.columns[1:])
    result = []
    for frame in [train, test]:
        merged = frame.merge(agg, on='txId', how='left', sort=False, validate='one_to_one')
        require(np.array_equal(merged.txId, frame.txId), 'Graph join reordered transactions')
        merged[['in_degree', 'in_known_sources']] = merged[['in_degree', 'in_known_sources']].fillna(0)
        result.append(merged)
    return *result, cols, {'edges': len(edges), 'duplicate_edges': int(edges.duplicated().sum()),
                         'edges_with_both_features': int(known.sum()), 'cross_step_known_edges': 0,
                         'graph_nodes': len(set(edges.txId1) | set(edges.txId2)),
                         'test_without_known_predecessor_fraction': float((result[1].in_known_sources == 0).mean())}


def split(data, fold):
    require(fold['train_end'] < fold['valid_start'] <= fold['valid_end'], 'Invalid chronological fold')
    labelled = data.loc[data.label.notna()]
    train = labelled.loc[labelled.time_step <= fold['train_end']]
    valid = labelled.loc[labelled.time_step.between(fold['valid_start'], fold['valid_end'])]
    require(train.label.nunique() == valid.label.nunique() == 2, 'Fold needs both classes')
    return train, valid


def fit_predict(train, predict, cols, params):
    require('label' not in cols and 'txId' not in cols and 'time_step' not in cols, 'Forbidden model column')
    require(train.label.notna().all(), 'Unknown labels cannot be used for supervised fit')
    model = LGBMClassifier(**params)
    model.fit(train[cols], train.label.astype(int))
    positive = list(model.classes_).index(1)
    return model, model.predict_proba(predict[cols])[:, positive]


def metrics(y, p):
    return {'roc_auc': float(roc_auc_score(y, p)) if y.nunique() == 2 else None,
            'average_precision': float(average_precision_score(y, p)) if (y == 1).any() else None,
            'rows': len(y), 'positives': int((y == 1).sum())}
