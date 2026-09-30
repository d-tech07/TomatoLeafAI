"""TomatoLeafAI v21 inference pipeline for the web app (TensorFlow-free).

photo -> quality + leaf-structure gate -> background removal V3 (crop to leaf)
      -> deep tomato-identity check (OOD-gate CNN + Mahalanobis + energy)
      -> classifier(s) -> calibrated confidence -> Grad-CAM + Leaf-Focus Score
      -> disease regions + affected-area estimate -> treatment advice
"""
import base64, hashlib, io, json, os, tempfile, threading, time, urllib.request

import cv2
import numpy as np
from PIL import Image, ImageOps

from .seg_v21 import BackgroundAwarePreprocessorV3
from .gate_v21 import TomatoLeafGateV21
from .ood_v21 import FeatureOODV21
from .explain import upsample, lesion_proxy_mask, focus_metrics, disease_regions, overlay, overlay_focus
from .litert_engine import LiteModelPair, LiteClassifier
from . import treatment as T

DEFAULT_CLASSES = sorted([
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'])

MODEL_META = {
    'Hybrid_CNN_ViT': {'label': 'LAG-HViT (proposed hybrid)', 'short': 'Hybrid CNN-ViT',
                       'kind': 'proposed',
                       'desc': 'Lesion-attention-guided hybrid: CNN lesion map steers a Vision Transformer '
                               '(Lesion-Biased MHSA), gated fusion, deep supervision.'},
    'EfficientNetB7': {'label': 'EfficientNetB7', 'short': 'EfficientNetB7', 'kind': 'baseline',
                       'desc': 'ImageNet-pretrained EfficientNetB7 on the shared v21 trunk.'},
    'VGG16': {'label': 'VGG16', 'short': 'VGG16', 'kind': 'baseline',
              'desc': 'ImageNet-pretrained VGG16 on the shared v21 trunk.'},
    'RegNetY008': {'label': 'RegNetY-008', 'short': 'RegNetY008', 'kind': 'baseline',
                   'desc': 'RegNetY-style network trained from scratch (grouped conv + squeeze-excitation).'},
    'CNN_Only': {'label': 'Custom CNN', 'short': 'CNN', 'kind': 'baseline',
                 'desc': 'The proposal\'s custom convolutional network, trained from scratch.'},
    'NaiveConcat_CNN_ViT': {'label': 'Naive concat (ablation)', 'short': 'NaiveConcat', 'kind': 'ablation',
                            'desc': 'Same hybrid without lesion bias / lesion pooling / gated fusion.'},
    'ViT_Branch': {'label': 'ViT branch only (ablation)', 'short': 'ViT branch', 'kind': 'ablation',
                   'desc': 'Transformer branch alone on the CNN-stem tokens.'},
    'Hybrid_CNN_ViT_noCAMreg': {'label': 'Hybrid without CAM regulariser (ablation)', 'short': 'Hybrid (no reg)',
                                'kind': 'ablation', 'desc': 'Proposed hybrid trained without the Leaf-Focus loss.'},
}
MODEL_ORDER = list(MODEL_META)


def pretty(cls):
    s = cls.replace('Tomato___', '').replace('_', ' ').replace('  ', ' ').strip()
    s = s.replace('Spider mites Two-spotted spider mite', 'Spider mites (two-spotted)')
    return s[:1].upper() + s[1:]


def _b64_jpeg(rgb, q=85):
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(np.asarray(rgb, np.uint8), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class ModelStore:
    """Finds model files in a bundled folder, or downloads them once from
    MODEL_BASE_URL into a writable cache (/tmp on Vercel), checking sha256
    against manifest.json."""

    def __init__(self, local_dir, base_url=None, cache_dir=None):
        self.local_dir = local_dir
        self.base_url = (base_url or '').rstrip('/') or None
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'tomatoleaf_models')
        self._lock = threading.Lock()
        self._missing = set()
        self.manifest = self._read_manifest()

    def _read_manifest(self):
        p = os.path.join(self.local_dir, 'manifest.json')
        if os.path.exists(p):
            return json.load(open(p))
        if self.base_url:
            try:
                with urllib.request.urlopen(self.base_url + '/manifest.json', timeout=20) as r:
                    return json.loads(r.read().decode())
            except Exception as e:  # noqa: BLE001
                return {'models': {}, 'files': {}, 'error': f'manifest download failed: {e}'}
        return {'models': {}, 'files': {}}

    def exists(self, fname):
        return (os.path.exists(os.path.join(self.local_dir, fname))
                or fname in self.manifest.get('files', {}))

    def path(self, fname):
        local = os.path.join(self.local_dir, fname)
        if os.path.exists(local):
            return local
        cached = os.path.join(self.cache_dir, fname)
        if os.path.exists(cached):
            return cached
        if not self.base_url:
            raise FileNotFoundError(f'{fname} not found in {self.local_dir} and MODEL_BASE_URL is not set')
        with self._lock:
            if os.path.exists(cached):
                return cached
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = cached + '.part'
            with urllib.request.urlopen(f'{self.base_url}/{fname}', timeout=120) as r, open(tmp, 'wb') as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            want = self.manifest.get('files', {}).get(fname, {}).get('sha256')
            if want and _sha256(tmp) != want:
                os.remove(tmp)
                raise IOError(f'checksum mismatch for {fname}')
            os.replace(tmp, cached)
        return cached

    def _small(self, fname):
        """Path of a small config/result file, or None. With MODEL_BASE_URL it is tried even if the
        manifest does not list it; a miss is remembered so it is not re-fetched on every request."""
        if fname in self._missing or not (self.exists(fname) or self.base_url):
            return None
        try:
            return self.path(fname)
        except Exception:  # noqa: BLE001
            self._missing.add(fname)
            return None

    def read_json(self, fname):
        p = self._small(fname)
        try:
            return json.load(open(p)) if p else None
        except Exception:  # noqa: BLE001
            return None

    def read_text(self, fname):
        p = self._small(fname)
        try:
            return open(p).read() if p else None
        except Exception:  # noqa: BLE001
            return None


class TomatoLeafEngine:
    def __init__(self, store, conf_threshold=0.60, threads=2):
        self.store = store
        self.conf_threshold = float(conf_threshold)
        self.threads = threads
        ci = store.read_json('class_indices.json')
        if isinstance(ci, dict) and ci:
            self.class_names = [k for k, _ in sorted(ci.items(), key=lambda kv: kv[1])]
        else:
            self.class_names = DEFAULT_CLASSES
        self.gate_cfg = store.read_json('gate_v21_config.json') or {}
        fo = store.read_json('feature_ood_v21.json')
        self.feature_ood = FeatureOODV21.from_state(fo) if fo else None
        cal = store.read_json('calibration_summary.json') or {}
        self.temperatures = {k: float(v.get('temperature', 1.0)) for k, v in cal.items() if isinstance(v, dict)}
        self.prep = BackgroundAwarePreprocessorV3(img_size=(224, 224))
        self.gate = TomatoLeafGateV21(self.prep, accept_threshold=self.gate_cfg.get('accept_threshold', 0.62),
                                      bands=self.gate_cfg.get('bands'),
                                      deep_thresholds=self.gate_cfg.get('deep_thresholds'))
        self._models, self._ood_gate = {}, None
        self._lock = threading.Lock()
        self.gate_model_name = self.gate_cfg.get('feature_model', 'Hybrid_CNN_ViT')

    # ------------------------------------------------------------------ models
    def available_models(self):
        man = self.store.manifest.get('models', {})
        names = [n for n in MODEL_ORDER if n in man
                 or (self.store.exists(f'{n}.trunk.tflite') and self.store.exists(f'{n}.head.tflite'))]
        names += [n for n in man if n not in names]
        return names

    @property
    def default_model(self):
        av = self.available_models()
        return 'Hybrid_CNN_ViT' if 'Hybrid_CNN_ViT' in av else (av[0] if av else None)

    def model(self, name):
        with self._lock:
            if name not in self._models:
                info = self.store.manifest.get('models', {}).get(name, {})
                self._models[name] = LiteModelPair(self.store.path(f'{name}.trunk.tflite'),
                                                   self.store.path(f'{name}.head.tflite'), self.threads,
                                                   head_outputs=info.get('head_outputs'))
            return self._models[name]

    def ood_gate(self):
        with self._lock:
            if self._ood_gate is None and self.store.exists('ood_gate.tflite'):
                self._ood_gate = LiteClassifier(self.store.path('ood_gate.tflite'), self.threads)
            return self._ood_gate

    def models_info(self):
        man = self.store.manifest.get('models', {})
        out = []
        for n in self.available_models():
            m = dict(MODEL_META.get(n, {'label': n, 'short': n, 'kind': 'other', 'desc': ''}))
            info = man.get(n, {})
            m.update({'name': n, 'params': info.get('params'), 'backbone': info.get('backbone'),
                      'size_mb': (info.get('trunk_mb') or 0) + (info.get('head_mb') or 0) or None,
                      'temperature': self.temperatures.get(n), 'default': n == self.default_model})
            out.append(m)
        return out

    def status(self):
        dt = self.gate_cfg.get('deep_thresholds') or {}
        return {'models': self.available_models(), 'default_model': self.default_model,
                'gate_mode': ('heuristic + deep tomato-identity (' + ', '.join(sorted(dt)) + ')') if dt
                else 'heuristic stages only (run notebook Cell 9.5 and re-export to enable the deep stage)',
                'feature_ood': self.feature_ood is not None, 'ood_gate_cnn': self.store.exists('ood_gate.tflite'),
                'calibrated_models': sorted(self.temperatures), 'manifest_error': self.store.manifest.get('error')}

    # --------------------------------------------------------------- helpers
    @staticmethod
    def decode(image_bytes, max_side=1600):
        im = Image.open(io.BytesIO(image_bytes))
        im = ImageOps.exif_transpose(im).convert('RGB')
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)

    def _probs(self, name, logits):
        t = self.temperatures.get(name, 1.0)
        z = logits / max(t, 1e-3)
        p = np.exp(z - z.max())
        return p / p.sum(), name in self.temperatures

    def _deep_scores(self, img_bgr, gate_out):
        ds = {}
        if self.feature_ood is not None and gate_out is not None:
            pv = self.feature_ood.pvalues(gate_out['embedding'][None].astype(np.float64),
                                          gate_out['logits'][None].astype(np.float64))
            ds['maha_score'] = float(pv['maha_score'][0]); ds['energy_score'] = float(pv['energy_score'][0])
        og = self.ood_gate()
        if og is not None:
            h, w = og.input_hw
            x = cv2.resize(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), (w, h)).astype(np.float32) / 255.0
            z = og.logits(x); e = np.exp(z - z.max())
            ds['ood_gate_prob'] = float(e[1] / e.sum())             # classes ['Other', 'Tomato']
        if gate_out is not None:
            p, _ = self._probs(self.gate_model_name, gate_out['logits'])
            ds['msp'] = float(p.max())
        keep = set((self.gate_cfg.get('deep_thresholds') or {}).keys())
        return {k: v for k, v in ds.items() if k in keep}, ds

    def _explain(self, name, out, seg, rgb, les_mask, healthy):
        leaf = seg['leaf_mask']
        cam = upsample(out['gradcam'], rgb.shape[:2])
        met = focus_metrics(cam, leaf, None if healthy else les_mask)
        leaf_cam = cam * (leaf > 0)
        guided = leaf_cam * (0.25 + 0.75 * cv2.GaussianBlur(les_mask.astype(np.float32), (0, 0), 2))
        guided = guided / (guided.max() + 1e-8)
        regs = [] if healthy else disease_regions(cam, les_mask)
        boxed = overlay_focus(rgb, guided)
        for r in regs[:3]:
            x, y, w, h = r['box']
            cv2.rectangle(boxed, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 255, 255), 2)
        if not healthy:
            cnts, _ = cv2.findContours(les_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(boxed, cnts, -1, (255, 0, 60), 1)
        imgs = {'gradcam': _b64_jpeg(overlay(rgb, cam)), 'disease_regions': _b64_jpeg(boxed)}
        if out.get('lesion_gate') is not None:
            g = np.asarray(out['lesion_gate'], np.float32)
            g = (g - g.min()) / (g.max() - g.min() + 1e-8)          # min-max for display only
            imgs['lesion_attention'] = _b64_jpeg(overlay_focus(rgb, upsample(g, rgb.shape[:2])))
        return met, regs, imgs

    def _classify(self, name, out, seg, rgb, les_mask):
        p, calibrated = self._probs(name, out['logits'])
        order = np.argsort(-p)
        k = int(order[0]); cls = self.class_names[k]; conf = float(p[k])
        healthy = 'healthy' in cls.lower()
        met, regs, imgs = self._explain(name, out, seg, rgb, les_mask, healthy)
        return {'model': name, 'label': MODEL_META.get(name, {}).get('label', name),
                'class': cls, 'display_name': pretty(cls), 'confidence': conf, 'calibrated': calibrated,
                'confident': conf >= self.conf_threshold, 'healthy': healthy,
                'top': [{'class': self.class_names[i], 'display_name': pretty(self.class_names[i]),
                         'prob': float(p[i])} for i in order[:5]],
                'focus': met, 'disease_regions': regs, 'images': imgs}

    # ------------------------------------------------------------------ main
    def analyze(self, image_bytes, model_name=None, compare=False):
        """Generator of progress events; the last event has type 'result'."""
        t0 = time.time()

        def ev(stage, pct, detail):
            return {'type': 'progress', 'stage': stage, 'pct': pct, 'detail': detail}

        yield ev('decode', 5, 'Reading photo')
        img = self.decode(image_bytes)
        yield ev('segment', 15, 'Removing background (soil, stones, shadows) and checking the photo')
        seg = self.prep.process(img)
        g = self.gate.evaluate(img, seg=seg)
        res = {'accepted': False, 'stage': 'gate', 'message': g['message'], 'reasons': g['reasons'],
               'preview': {'background_removed': _b64_jpeg(cv2.cvtColor(seg['background_removed'], cv2.COLOR_BGR2RGB)),
                           'leaf_mask': _b64_jpeg(cv2.cvtColor(seg['stages']['overlay'], cv2.COLOR_BGR2RGB))},
               'segmentation': {k: seg[k] for k in ('leaf_coverage', 'segmentation_confidence', 'is_reliable',
                                                     'lesion_fraction', 'background_composition')}}
        if g['decision'] == 'REJECT' and g['veto'] not in (None, 'low_confidence'):
            res['timing_ms'] = int((time.time() - t0) * 1000)
            yield {'type': 'result', 'result': res}
            return

        avail = self.available_models()
        if not avail:
            res.update({'stage': 'setup', 'message': 'No models are installed on the server yet.'})
            yield {'type': 'result', 'result': res}
            return
        rgb = cv2.cvtColor(seg['background_removed'], cv2.COLOR_BGR2RGB)
        x01 = rgb.astype(np.float32) / 255.0
        gname = self.gate_model_name if self.gate_model_name in avail else None
        outs = {}
        yield ev('leafcheck', 30, 'Checking that this is a tomato leaf')
        if gname:
            outs[gname] = self.model(gname).run(x01)
        deep, deep_all = self._deep_scores(img, outs.get(gname))
        g = self.gate.evaluate(img, deep_scores=deep, seg=seg)
        res.update({'gate': {'decision': g['decision'], 'confidence': g['confidence'],
                             'heuristic_score': g['heuristic_score'], 'deep_score': g['deep_score'],
                             'deep_scores': deep_all}, 'message': g['message'], 'reasons': g['reasons']})
        if g['decision'] == 'REJECT':
            res['timing_ms'] = int((time.time() - t0) * 1000)
            yield {'type': 'result', 'result': res}
            return

        les_mask = lesion_proxy_mask(rgb, seg['leaf_mask'], seg.get('lesion_mask'))
        names = avail if compare else [model_name if model_name in avail else self.default_model]
        per = []
        for i, n in enumerate(names):
            yield ev('predict', 40 + int(45 * i / max(1, len(names))),
                     f'Running {MODEL_META.get(n, {}).get("label", n)} + Grad-CAM')
            ts = time.time()
            out = outs.get(n) or self.model(n).run(x01)
            r = self._classify(n, out, seg, rgb, les_mask)
            r['time_ms'] = int((time.time() - ts) * 1000)
            per.append(r)
            yield {'type': 'partial', 'model': r}
        yield ev('advice', 92, 'Preparing treatment advice')
        main = per[0]
        if compare:
            votes = {}
            for r in per:
                votes[r['class']] = votes.get(r['class'], 0) + r['confidence']
            best = max(votes, key=votes.get)
            agree = sum(r['class'] == best for r in per)
            main = next((r for r in per if r['model'] == self.default_model and r['class'] == best),
                        max((r for r in per if r['class'] == best), key=lambda r: r['confidence']))
            res['consensus'] = {'class': best, 'display_name': pretty(best), 'agree': agree, 'total': len(per)}
        leaf_px = max(1, int((seg['leaf_mask'] > 0).sum()))
        area = None if main['healthy'] else float((les_mask & (seg['leaf_mask'] > 0)).sum() / leaf_px)
        advice = T.get_treatment_recommendation(main['class'], float(main['confidence']),
                                                calibrated=bool(main['calibrated']))
        res.update({'accepted': True, 'stage': 'done', 'model': main['model'], 'prediction': main,
                    'results': per, 'compare': bool(compare),
                    'affected_area_estimate': area,
                    'treatment': advice,
                    'message': ('Diagnosis ready.' if main['confident'] else
                                f'Low confidence ({main["confidence"]:.0%}). The leaf may be unclear or the disease '
                                'unusual — treat this as a hint and confirm with an agronomist or retake the photo.')})
        res['timing_ms'] = int((time.time() - t0) * 1000)
        yield {'type': 'result', 'result': res}
