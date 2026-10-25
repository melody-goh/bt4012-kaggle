import sys
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from graph_risk_propagation import neighbours,propagate


class GraphRiskTests(unittest.TestCase):
    def test_direction_and_isolated_nodes(self):
        nodes=pd.DataFrame({'txId':[10,20,30]})
        edges=pd.DataFrame({'txId1':[10,99],'txId2':[20,20]})
        a=neighbours(nodes,edges)
        scores=np.array([.8,.2,.4])
        np.testing.assert_allclose(propagate(scores,a['upstream'],.25),[.8,.35,.4])
        np.testing.assert_allclose(propagate(scores,a['downstream'],.25),[.65,.2,.4])
        np.testing.assert_allclose(propagate(scores,a['undirected'],.25),[.65,.35,.4])

    def test_labels_are_unused_and_logodds_bounded(self):
        nodes=pd.DataFrame({'txId':[10,20,30],'label':[0,1,np.nan]})
        edges=pd.DataFrame({'txId1':[10],'txId2':[20]})
        a=neighbours(nodes,edges)
        b=neighbours(nodes.assign(label=[1,0,1]),edges)
        self.assertEqual((a['undirected']!=b['undirected']).nnz,0)
        p=propagate(np.array([1.,0.,.4]),a['undirected'],.25,True)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(((p>=0)&(p<=1)).all())
        self.assertEqual(p[2],.4)


if __name__=='__main__':
    unittest.main()
