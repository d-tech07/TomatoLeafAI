"""LiteRT (TFLite) inference for TomatoLeafAI v21 models -- no TensorFlow needed.

Each model is two .tflite files written by tools/export_litert.py:
  <Name>.trunk.tflite  image[1,H,W,3] in [0,1] -> lesion_map[1,14,14,C]
  <Name>.head.tflite   lesion_map -> (logits, embedding, gradcam[1,14,14] [, lesion_gate])
The head's output order is recorded in manifest.json ('head_outputs').
"""
import re
import threading
import numpy as np

try:                                         # preferred: small standalone runtime
    from ai_edge_litert.interpreter import Interpreter
except Exception:                            # pragma: no cover - fallbacks
    try:
        from tflite_runtime.interpreter import Interpreter
    except Exception:
        from tensorflow.lite.python.interpreter import Interpreter  # full TF (local only)

DEFAULT_HEAD_OUTPUTS = ['logits', 'embedding', 'gradcam', 'lesion_gate']


def _load(path, threads):
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    return it


def _ordered_outputs(it):
    """Output tensors in the order the exported tuple returned them
    (names Identity, Identity_1, Identity_2, ...)."""
    def key(d):
        m = re.search(r'_(\d+)$', d['name'])
        return int(m.group(1)) if m else 0
    return sorted(it.get_output_details(), key=key)


def _invoke(it, x):
    d = it.get_input_details()[0]
    it.set_tensor(d['index'], np.asarray(x, d['dtype']))
    it.invoke()
    return [it.get_tensor(o['index']) for o in _ordered_outputs(it)]


class LiteModelPair:
    def __init__(self, trunk_path, head_path, threads=2, head_outputs=None):
        self.trunk = _load(trunk_path, threads)
        self.head = _load(head_path, threads)
        self._lock = threading.Lock()          # interpreters are not re-entrant
        d = self.trunk.get_input_details()[0]
        self.input_hw = (int(d['shape'][1]), int(d['shape'][2]))
        n = len(self.head.get_output_details())
        self.head_outputs = list(head_outputs or DEFAULT_HEAD_OUTPUTS)[:n]

    def run(self, x01):
        """x01: (H,W,3) float32 in [0,1] (background-removed crop)."""
        x = np.asarray(x01, np.float32)[None]
        with self._lock:
            lm = _invoke(self.trunk, x)[0]
            outs = _invoke(self.head, lm.astype(np.float32))
        res = {k: np.array(v[0]) for k, v in zip(self.head_outputs, outs)}
        res['logits'] = res['logits'].astype(np.float64)
        return res


class LiteClassifier:
    """Single-output classifier (the OOD gate)."""

    def __init__(self, path, threads=2):
        self.it = _load(path, threads)
        self._lock = threading.Lock()
        d = self.it.get_input_details()[0]
        self.input_hw = (int(d['shape'][1]), int(d['shape'][2]))

    def logits(self, x01):
        with self._lock:
            return np.asarray(_invoke(self.it, np.asarray(x01, np.float32)[None])[0][0], np.float64)
