from pathlib import Path
import numpy as np
import pandas as pd

BASE = [f'feat_{i}' for i in range(1, 166)]
# features averaged over each transaction's incoming neighbours
AGG = ['feat_2', 'feat_53', 'feat_58', 'feat_16', 'feat_90', 'feat_138', 'feat_142', 'feat_163']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_submission(frame, test, sample):
    require(list(frame.columns) == ['index', 'target'], 'Submission columns must be index,target')
    require(np.array_equal(frame['index'], test['index']), 'Rows must follow test.csv order')
    require(np.array_equal(frame['index'], sample['index']), 'Rows must follow sample_submission.csv order')
    require(np.isfinite(frame.target).all() and frame.target.between(0, 1).all(), 'Invalid probabilities')


def load_data(root):
    train, test, edges, sample = [pd.read_csv(Path(root) / f) for f in
                                  ['train.csv', 'test.csv', 'txs_edgelist.csv', 'sample_submission.csv']]
    require(train.time_step.max() < test.time_step.min(), 'Train and test time steps overlap')
    require(len(test) == len(sample), 'test.csv and sample_submission.csv differ in length')
    return train, test, edges, sample


def graph_features(train, test, edges):
    """Mean and std of AGG over each transaction's incoming neighbours, plus in-degree
    (all edges) and the number of neighbours with features. Uses no labels."""
    nodes = pd.concat([train[['txId', 'time_step'] + AGG], test[['txId', 'time_step'] + AGG]], ignore_index=True)
    graph = edges[['txId1', 'txId2']].merge(nodes.rename(columns={
        'txId': 'txId1', 'time_step': 'source_step', **{c: f'source_{c}' for c in AGG}
    }), on='txId1', how='left')
    grouped = graph.groupby('txId2')
    agg = grouped[[f'source_{c}' for c in AGG]].agg(['mean', 'std'])
    agg.columns = [f'in_{c}_{stat}' for c, stat in agg.columns]
    agg['in_degree'] = grouped.size()
    agg['in_known_sources'] = grouped.source_step.count()
    agg = agg.reset_index().rename(columns={'txId2': 'txId'})
    cols = list(agg.columns[1:])
    result = []
    for frame in [train, test]:
        merged = frame.merge(agg, on='txId', how='left', sort=False)
        merged[['in_degree', 'in_known_sources']] = merged[['in_degree', 'in_known_sources']].fillna(0)
        result.append(merged)
    info = {'edges': len(edges), 'test_without_known_predecessor': float((result[1].in_known_sources == 0).mean())}
    return *result, cols, info
