"""Portable federated trees; fixed public binning stays on the POS device."""
import json
from pathlib import Path
import numpy as np


def export_forest(forest, path):
    import onnx
    from onnx import helper as h, TensorProto as T
    if not forest.trees:
        raise ValueError('Cannot deploy an empty forest')
    attrs = {k: [] for k in ('nodes_treeids', 'nodes_nodeids', 'nodes_featureids',
        'nodes_modes', 'nodes_values', 'nodes_truenodeids', 'nodes_falsenodeids',
        'target_treeids', 'target_nodeids', 'target_ids', 'target_weights')}
    for tid, tree in enumerate(forest.trees):
        for nid, feature in enumerate(tree.feature):
            leaf = feature < 0
            if leaf and not np.isfinite(tree.value[nid]):
                raise ValueError('Unfinished tree cannot be exported')
            for key, value in dict(nodes_treeids=tid, nodes_nodeids=nid,
                nodes_featureids=max(feature, 0), nodes_modes='LEAF' if leaf else 'BRANCH_LEQ',
                nodes_values=0.0 if leaf else float(tree.threshold[nid]),
                nodes_truenodeids=0 if leaf else tree.left[nid],
                nodes_falsenodeids=0 if leaf else tree.right[nid]).items():
                attrs[key].append(value)
            if leaf:
                for key, value in dict(target_treeids=tid, target_nodeids=nid,
                    target_ids=0, target_weights=float(tree.value[nid])).items():
                    attrs[key].append(value)
    nodes = [h.make_node('TreeEnsembleRegressor', ['binned'], ['raw'], domain='ai.onnx.ml',
        n_targets=1, base_values=[float(forest.base_score)], **attrs),
        h.make_node('Sigmoid', ['raw'], ['risk'])]
    graph = h.make_graph(nodes, 'UPI federated GBDT',
        [h.make_tensor_value_info('binned', T.FLOAT, [None, forest.schema['n_features']])],
        [h.make_tensor_value_info('risk', T.FLOAT, [None, 1])])
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17), h.make_opsetid('ai.onnx.ml', 3)], ir_version=9)
    h.set_model_props(model, {'schema': json.dumps(forest.schema), 'preprocessing': 'upi_fl.bins.binned_matrix'})
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.onnx.tmp')
    onnx.save(model, tmp)
    tmp.replace(path)
    return path


def session(path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def predict(runtime, binned):
    return runtime.run(['risk'], {'binned': np.asarray(binned, dtype=np.float32)})[0].reshape(-1)
