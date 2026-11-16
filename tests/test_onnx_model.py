import tempfile
from pathlib import Path
import unittest
import numpy as np
from service.common import bins, gbdt
from service.onnx_model import export_forest, session, predict


class ONNXTests(unittest.TestCase):
    def test_threshold_edges_multiple_trees_and_base_score(self):
        schema = bins.build_schema(['x'], {'x':'binary'}, [2], {})
        forest = gbdt.Forest(schema, base_score=0.37)
        for low, high in [(-1.2, 0.8), (0.3, -0.7)]:
            tree = gbdt.Tree(1)
            left, right = tree.add_split(0, 0, 0)
            tree.set_leaf(left, low)
            tree.set_leaf(right, high)
            forest.trees.append(tree)
        X = np.array([[0], [1], [0], [1]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = session(export_forest(forest, Path(tmp)/'model.onnx'))
            np.testing.assert_allclose(predict(runtime, X), forest.predict_proba(X), atol=2e-7)

    def test_unfinished_tree_refused(self):
        schema = bins.build_schema(['x'], {'x':'binary'}, [2], {})
        forest = gbdt.Forest(schema)
        forest.trees.append(gbdt.Tree(1))
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, 'Unfinished'):
            export_forest(forest, Path(tmp)/'model.onnx')
