"""Export trained TomatoLeafAI v21 models (.keras) to LiteRT (.tflite) for the web app.

Run ONCE on the machine where the models were trained (same TensorFlow/Keras as
training), from the web-app folder:

    python tools/export_litert.py --models-dir /path/to/TomatoLeafAI_Project_v5/models \
                                  --results-dir /path/to/TomatoLeafAI_Project_v5/results \
                                  --out models_litert

For every <Name>_final.keras it writes two files:
    <Name>.trunk.tflite     image (1,224,224,3 in [0,1])  -> lesion map (1,14,14,C)
    <Name>.head.tflite      lesion map -> logits, embedding, Grad-CAM (14x14) of the
                            predicted class [, lesion gate for the hybrid]
Grad-CAM is computed INSIDE the head model (tf.GradientTape traced into the graph and
lowered to LiteRT builtin ops), so the web server needs no TensorFlow at all.
Weights are stored as float16 (half the size; computation stays float32).
Also converts ood_gate.keras and copies gate_v21_config.json / feature_ood_v21.json /
calibration + evaluation summaries, then writes manifest.json (sizes + sha256).
"""
import argparse, glob, hashlib, json, os, shutil, sys

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, '..')]

import numpy as np                                  # noqa: E402
import tensorflow as tf                             # noqa: E402
import keras                                        # noqa: E402
import keras_layers_v21 as L                       # noqa: E402  (registers the custom layers)

SERVED = ['Hybrid_CNN_ViT', 'EfficientNetB7', 'VGG16', 'RegNetY008', 'CNN_Only',
          'NaiveConcat_CNN_ViT', 'ViT_Branch', 'Hybrid_CNN_ViT_noCAMreg']


def to_float32(model):
    """Models were trained under mixed_float16; rebuild with float32 layers for
    CPU/LiteRT (identical weights, no fp16 CPU kernels)."""
    js = (model.to_json().replace('"mixed_float16"', '"float32"')
          .replace('"dtype": "float16"', '"dtype": "float32"'))
    m32 = keras.models.model_from_json(js)
    m32.set_weights([np.asarray(w, np.float32) for w in model.get_weights()])
    return m32


# ------------------------------------------------------------------------ export patches
def _erf(x):
    """Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7) built from LiteRT builtin ops,
    so d/dx of exact-GELU converts without the unsupported Erfc gradient op."""
    s = tf.sign(x); a = tf.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
               + 0.254829592) * t * tf.exp(-a * a)
    return s * y


def gelu_export(x):
    return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0).astype(np.float32)))


def _patch_for_export():
    def mhsa_call(self, inputs, training=None, return_attention=False):
        x, log_s = inputs
        B, N = tf.shape(x)[0], tf.shape(x)[1]
        qkv = tf.reshape(self.qkv(x), [B, N, 3, self.heads, self.hd])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = tf.unstack(qkv, num=3, axis=0)                 # grad = stack (builtin)
        logits = tf.matmul(tf.cast(q, tf.float32), tf.cast(k, tf.float32), transpose_b=True) / np.float32(np.sqrt(self.hd))
        if self.use_lesion_bias:
            beta = tf.nn.softplus(tf.cast(self.beta_raw, tf.float32))[None, :, None, None]
            ls = tf.expand_dims(tf.expand_dims(tf.cast(log_s, tf.float32), 1), 1)   # (B,1,1,N): grad = reshape
            logits = logits + beta * ls
        attn = tf.nn.softmax(logits, axis=-1)
        out = tf.matmul(tf.cast(attn, v.dtype), v)
        out = tf.reshape(tf.transpose(out, [0, 2, 1, 3]), [B, N, self.dim])
        return self.proj(out)

    def split_call(self, z):
        cls, tok = tf.split(z, [1, -1], axis=1)                   # grad = concat (builtin)
        return [tf.squeeze(cls, 1), tok]

    def logprior_call(self, s):
        s = tf.cast(s, tf.float32)
        return tf.pad(tf.math.log(s + 1e-4), [[0, 0], [1, 0]])     # log(1)=0 for [CLS]

    def pool_call(self, inputs):
        tok, s = inputs
        s = tf.clip_by_value(tf.cast(s, tf.float32), 1e-4, 1 - 1e-4)
        w = tf.nn.softmax(tf.nn.softplus(tf.cast(self.tau_raw, tf.float32)) * tf.math.log(s / (1 - s)), axis=-1)
        return tf.reduce_sum(tf.expand_dims(tf.cast(w, tok.dtype), -1) * tok, axis=1)

    def fusion_call(self, inputs):
        w = tf.nn.softmax(tf.cast(self.scorer(tf.concat(inputs, -1)), tf.float32), -1)
        st = tf.stack(inputs, 1)
        return tf.reduce_sum(tf.expand_dims(tf.cast(w, st.dtype), -1) * st, 1)

    L.LesionBiasedMHSA.call = mhsa_call
    L.LesionTokenPool.call = pool_call
    L.GatedFusionV21.call = fusion_call
    L.SplitClsTokens.call = split_call
    L.LogLesionPrior.call = logprior_call


def relu_export(x):
    """Exact ReLU whose gradient (step function) is built from builtin ops
    (tf.ReluGrad has no LiteRT kernel)."""
    return x * tf.stop_gradient(tf.cast(x > 0, x.dtype))


def _use_export_gelu(model):
    """Swap GELU/ReLU activations inside the HEAD for numerically identical
    versions whose gradients convert to LiteRT builtins."""
    n = 0
    for l in model.layers:
        name = getattr(getattr(l, 'activation', None), '__name__', '')
        if isinstance(l, (keras.layers.Dense, keras.layers.Activation)) and name in ('gelu', 'relu'):
            l.activation = gelu_export if name == 'gelu' else relu_export
            n += 1
    return n


def _convert(concrete_fns, trackable, fp16=True):
    """fp16=True -> float16 weights (half size); 'int8' -> dynamic-range int8 weights (quarter size,
    activations stay float); False -> float32."""
    # Freeze every variable (incl. ones read inside the GradientTape) into constants;
    # otherwise LiteRT keeps READ_VARIABLE ops that fail at runtime.
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
    concrete_fns = [convert_variables_to_constants_v2(cf) for cf in concrete_fns]
    c = tf.lite.TFLiteConverter.from_concrete_functions(concrete_fns)
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if fp16 == 'int8':
        c.optimizations = [tf.lite.Optimize.DEFAULT]          # dynamic-range: int8 weights, float compute
    elif fp16:
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
    return c.convert()


def export_model(model, name, out_dir, fp16=True):
    m = to_float32(model)
    trunk = next(l for l in m.layers if l.name.endswith('_Trunk'))
    head = next(l for l in m.layers if l.name.endswith('_Head') or l.name == 'BaselineHead')
    n_gelu = _use_export_gelu(head)
    img_shape = tuple(int(d) for d in trunk.input.shape[1:])
    lm_shape = tuple(int(d) for d in trunk.output.shape[1:])
    has_gate = 'aux_pseudo_lesion_head' in head.output

    @tf.function(input_signature=[tf.TensorSpec((1,) + img_shape, tf.float32)])
    def trunk_fn(x):
        return tf.cast(trunk(x, training=False), tf.float32)

    @tf.function(input_signature=[tf.TensorSpec((1,) + lm_shape, tf.float32)])
    def head_fn(lm):
        with tf.GradientTape() as t:
            t.watch(lm)
            o = head(lm, training=False)
            logits = tf.cast(o['logits'], tf.float32)
            onehot = tf.stop_gradient(tf.one_hot(tf.argmax(logits, axis=-1), tf.shape(logits)[-1]))
            score = tf.reduce_sum(logits * onehot)
        g = t.gradient(score, lm)
        alpha = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
        out = [logits, tf.cast(o['embedding'], tf.float32), tf.nn.relu(tf.reduce_sum(alpha * lm, axis=-1))]
        if has_gate:
            out.append(tf.cast(o['aux_pseudo_lesion_head'], tf.float32)[..., 0])
        return tuple(out)          # order recorded in the manifest as 'head_outputs'

    tb = _convert([trunk_fn.get_concrete_function()], trunk, fp16)
    hb = _convert([head_fn.get_concrete_function()], head, fp16)
    open(os.path.join(out_dir, f'{name}.trunk.tflite'), 'wb').write(tb)
    open(os.path.join(out_dir, f'{name}.head.tflite'), 'wb').write(hb)
    kind = trunk.name.replace('_Trunk', '')
    return {'name': name, 'backbone': kind, 'head': head.name, 'img_size': list(img_shape[:2]),
            'lesion_map': list(lm_shape), 'has_lesion_gate': bool(has_gate),
            'head_outputs': ['logits', 'embedding', 'gradcam'] + (['lesion_gate'] if has_gate else []),
            'params': int(model.count_params()), 'gelu_layers_patched': n_gelu,
            'trunk_mb': round(len(tb) / 1e6, 1), 'head_mb': round(len(hb) / 1e6, 1)}


def export_ood_gate(path, out_dir, fp16=True):
    m0 = keras.models.load_model(path, compile=False)
    shape = tuple(int(d) for d in m0.input_shape[1:])
    try:
        m = to_float32(m0)
        m(np.zeros((1,) + shape, np.float32))          # builds a Sequential rebuilt from JSON
    except Exception:                                   # noqa: BLE001 - fall back to the original
        m = m0

    @tf.function(input_signature=[tf.TensorSpec((1,) + shape, tf.float32)])
    def f(x):
        return tf.cast(m(x, training=False), tf.float32)
    b = _convert([f.get_concrete_function()], m, fp16)
    open(os.path.join(out_dir, 'ood_gate.tflite'), 'wb').write(b)
    return round(len(b) / 1e6, 1)


def verify(model, name, out_dir, n=3, seed=0):
    """Compares LiteRT against Keras on random inputs: probabilities and Grad-CAM."""
    from tomatoleaf_web.litert_engine import LiteModelPair
    rng = np.random.default_rng(seed)
    m = to_float32(model)
    trunk = next(l for l in m.layers if l.name.endswith('_Trunk'))
    head = next(l for l in m.layers if l.name.endswith('_Head') or l.name == 'BaselineHead')
    has_gate = 'aux_pseudo_lesion_head' in head.output
    lp = LiteModelPair(os.path.join(out_dir, f'{name}.trunk.tflite'), os.path.join(out_dir, f'{name}.head.tflite'),
                       head_outputs=['logits', 'embedding', 'gradcam'] + (['lesion_gate'] if has_gate else []))
    worst_p = worst_cam = 0.0
    for _ in range(n):
        x = rng.random((1,) + tuple(trunk.input.shape[1:])).astype(np.float32)
        with tf.GradientTape() as t:
            lm = trunk(x, training=False); t.watch(lm)
            lg = tf.cast(head(lm, training=False)['logits'], tf.float32)
            c = int(tf.argmax(lg[0])); s = lg[0, c]
        g = t.gradient(s, lm).numpy()[0]; A = lm.numpy()[0]
        cam_tf = np.maximum((A * g.mean((0, 1))).sum(-1), 0)
        p_tf = tf.nn.softmax(lg)[0].numpy()
        r = lp.run(x[0])
        p_lt = np.exp(r['logits'] - r['logits'].max()); p_lt /= p_lt.sum()
        worst_p = max(worst_p, float(np.abs(p_tf - p_lt).max()))
        den = cam_tf.max() + 1e-8
        worst_cam = max(worst_cam, float(np.abs(cam_tf / den - r['gradcam'] / (r['gradcam'].max() + 1e-8)).max()))
    return {'max_abs_prob_diff': worst_p, 'max_abs_normalised_cam_diff': worst_cam}


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--results-dir', default=None)
    ap.add_argument('--out', default='models_litert')
    ap.add_argument('--only', nargs='*', default=None, help='model names to export (default: all found)')
    ap.add_argument('--no-fp16', action='store_true', help='keep float32 weights (largest files)')
    ap.add_argument('--int8', action='store_true',
                    help='int8 weights (~4x smaller than float32): fits every model in one Vercel function')
    ap.add_argument('--no-verify', action='store_true')
    a = ap.parse_args()
    QUANT = 'int8' if a.int8 else (False if a.no_fp16 else True)
    os.makedirs(a.out, exist_ok=True)
    _patch_for_export()
    names = a.only or [n for n in SERVED if os.path.exists(os.path.join(a.models_dir, f'{n}_final.keras'))]
    if not names:
        sys.exit(f'No <Name>_final.keras found in {a.models_dir}')
    manifest = {'format': 'tomatoleaf-litert-v21', 'weights': {'int8': 'int8-dynamic', True: 'float16', False: 'float32'}[QUANT],
                'models': {}, 'files': {}}
    for n in names:
        print(f'Exporting {n} ...', flush=True)
        km = keras.models.load_model(os.path.join(a.models_dir, f'{n}_final.keras'), compile=False)
        info = export_model(km, n, a.out, fp16=QUANT)
        if not a.no_verify:
            info['verification'] = verify(km, n, a.out)
            print('   verification vs Keras:', info['verification'])
        manifest['models'][n] = info
        del km; keras.backend.clear_session()
    og = os.path.join(a.models_dir, 'ood_gate.keras')
    if os.path.exists(og):
        manifest['ood_gate_mb'] = export_ood_gate(og, a.out, fp16=QUANT)
        print('Exported ood_gate.tflite')
    for src_dir, fn in [(a.models_dir, 'feature_ood_v21.json'), (a.models_dir, 'class_indices.json'),
                        (a.results_dir, 'gate_v21_config.json'), (a.results_dir, 'calibration_summary.json'),
                        (a.results_dir, 'final_comparison_table.csv'), (a.results_dir, 'xai_v21_summary.csv'),
                        (a.results_dir, 'hypothesis_verdicts_v21.json'), (a.results_dir, 'latency_model_size_v21.csv')]:
        if src_dir and os.path.exists(os.path.join(src_dir, fn)):
            shutil.copy2(os.path.join(src_dir, fn), os.path.join(a.out, fn))
            print('Copied', fn)
    for p in sorted(glob.glob(os.path.join(a.out, '*'))):
        if os.path.basename(p) != 'manifest.json':
            manifest['files'][os.path.basename(p)] = {'bytes': os.path.getsize(p), 'sha256': sha256(p)}
    json.dump(manifest, open(os.path.join(a.out, 'manifest.json'), 'w'), indent=2)
    tot = sum(v['bytes'] for v in manifest['files'].values()) / 1e6
    print(f'\nDone: {len(names)} models -> {a.out}/ ({tot:.0f} MB total). manifest.json written.')


if __name__ == '__main__':
    main()
