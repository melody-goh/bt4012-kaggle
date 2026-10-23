import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import AGG, graph_features, split, validate_submission


class SafetyTests(unittest.TestCase):
    def test_graph_direction_missing_sources_and_sample_std(self):
        train = pd.DataFrame({'txId': [1, 2, 3], 'time_step': [1]*3, 'label': [0, 1, np.nan],
                              **{c: [2., 4., 10.] for c in AGG}})
        test = pd.DataFrame({'txId': [4], 'time_step': [2], **{c: [20.] for c in AGG}})
        edges = pd.DataFrame({'txId1': [1, 2, 99], 'txId2': [3, 3, 3]})
        result, _, cols, _ = graph_features(train, test, edges)
        self.assertEqual(len(cols), 18)
        node = result.set_index('txId').loc[3]
        self.assertEqual(node.in_degree, 3)
        self.assertEqual(node.in_known_sources, 2)
        self.assertEqual(node.in_source_feat_2_mean, 3)
        self.assertAlmostEqual(node.in_source_feat_2_std, np.sqrt(2))
        self.assertTrue(np.isnan(result.iloc[0].in_source_feat_2_mean))
        changed = train.copy()
        changed.label = [1, 0, 1]
        pd.testing.assert_frame_equal(result[cols], graph_features(changed, test, edges)[0][cols])
        with self.assertRaises(ValueError):
            graph_features(train, test, pd.DataFrame({'txId1': [1], 'txId2': [4]}))

    def test_unknown_labels_and_cutoff(self):
        data = pd.DataFrame({'time_step': [1, 1, 1, 2, 2], 'label': [0, 1, np.nan, 0, 1]})
        tr, va = split(data, {'train_end': 1, 'valid_start': 2, 'valid_end': 2})
        self.assertEqual(len(tr), 2)
        self.assertLess(tr.time_step.max(), va.time_step.min())

    def test_submission_rejects_reorder_and_nonfinite(self):
        test = pd.DataFrame({'index': [9, 3]})
        good = test.assign(target=[.2, .8])
        validate_submission(good, test, test)
        for bad in [good.iloc[::-1], good.assign(target=[np.nan, .4]), good.assign(index=[9, 9])]:
            with self.assertRaises(ValueError):
                validate_submission(bad, test, test)


if __name__ == '__main__':
    unittest.main()
