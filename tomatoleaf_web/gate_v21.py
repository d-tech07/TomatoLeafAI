# Web copy of notebook Cell 2.2b (identical logic).
# CELL 2.2b - TomatoLeafGateV21 (v21): multi-cue ACCEPT/REJECT gate with plain-language
# reject messages. Replaces the v8-v20 heuristic-only gate as the object GATE.
#
# ROOT CAUSES of "other images are accepted as tomato leaf" (read from the
# v9/v20 code and outputs):
#   * The v8-v20 gate was HEURISTIC-ONLY in the notebook (its own print says
#     "cnn_leaf_confidence / ood_score hooks ... currently unused") and 30% of
#     its score was "fraction of leaf-coloured pixels" -- any green or brown
#     object (grass, a green bag, a wooden table, a hand) could reach the 0.70
#     threshold. No cue looked at leaf STRUCTURE (veins, serrated margin).
#   * The trained OOD path (Cell 4.4) was tuned on an ood_val split where
#     AUROC was already 1.000, so the weight search had nothing to learn from
#     and picked {'entropy': 0.9, 'ood_gate_prob': 0.1} in v9 -- i.e. it
#     almost ignored the one component trained to say "tomato or not".
#     A perfect score on an easy split says nothing about grass, soil,
#     stones, hands or other crops photographed in the field.
#   * The gate and the OOD score were never fused into ONE decision with
#     hard vetoes, so a strong "no leaf here" signal could be averaged away.
#
# V21 DESIGN -- a cascade; any stage can reject with its own message:
#   Stage 0  image quality        blur / exposure / resolution / blank
#   Stage 1  leaf presence        green-tissue seed found, leaf coverage,
#                                 green fraction inside the object
#   Stage 2  leaf structure       shape (solidity, elongation, margin
#                                 serration/lobing), vein/ridge texture,
#                                 not grass-like texture
#   Stage 3  tomato identity      (needs trained models, Cell 9.5) OOD-gate
#                                 probability + Mahalanobis distance of the
#                                 classifier's embedding to the 10 tomato
#                                 classes + energy score
#   Fusion   weighted evidence >= accept_threshold AND no hard veto.
# Every threshold that depends on real data is re-calibrated in Cell 9.5 on
# tomato validation images at 95% true-accept rate; the defaults below are
# only a safe starting point.
import cv2
import numpy as np
from .seg_v21 import ColorMaps


def _band_score(v, lo, hi, soft):
    """1 inside [lo, hi], linear fall-off to 0 over `soft` outside."""
    if v is None or not np.isfinite(v):
        return 0.0
    if lo <= v <= hi:
        return 1.0
    d = (lo - v) if v < lo else (v - hi)
    return float(np.clip(1.0 - d / soft, 0.0, 1.0))


class ImageQualityCheckV21:
    def __init__(self, min_side=64, blank_std=3.0, blur_lap_var=12.0,
                 dark_mean=18.0, bright_mean=245.0, clip_frac=0.80):
        self.min_side, self.blank_std, self.blur_lap_var = min_side, blank_std, blur_lap_var
        self.dark_mean, self.bright_mean, self.clip_frac = dark_mean, bright_mean, clip_frac

    def check(self, img_bgr):
        m = {}
        if img_bgr is None or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
            return {'passed': False, 'reason': 'unreadable_or_not_rgb', 'metrics': m}
        h, w = img_bgr.shape[:2]
        m['resolution'] = (w, h)
        if min(h, w) < self.min_side:
            return {'passed': False, 'reason': 'resolution_too_low', 'metrics': m}
        s = 512.0 / max(h, w)
        g = cv2.cvtColor(cv2.resize(img_bgr, None, fx=s, fy=s) if s < 1 else img_bgr, cv2.COLOR_BGR2GRAY)
        m['intensity_std'] = float(g.std())
        if m['intensity_std'] < self.blank_std:
            return {'passed': False, 'reason': 'blank_or_solid_colour', 'metrics': m}
        m['laplacian_variance'] = float(cv2.Laplacian(g, cv2.CV_64F).var())
        if m['laplacian_variance'] < self.blur_lap_var:
            return {'passed': False, 'reason': 'too_blurry', 'metrics': m}
        m['mean_intensity'] = float(g.mean())
        m['dark_frac'] = float((g < 10).mean()); m['bright_frac'] = float((g > 246).mean())
        if m['mean_intensity'] < self.dark_mean or m['dark_frac'] > self.clip_frac:
            return {'passed': False, 'reason': 'too_dark', 'metrics': m}
        if m['mean_intensity'] > self.bright_mean or m['bright_frac'] > self.clip_frac:
            return {'passed': False, 'reason': 'overexposed', 'metrics': m}
        return {'passed': True, 'reason': None, 'metrics': m}


class LeafCueExtractorV21:
    """Interpretable leaf evidence computed on the preprocessor's work-frame
    image + mask (no second segmentation pass)."""

    def compute(self, work_bgr, work_mask, seg):
        from_seg = seg.get('segmentation_subscores', {}) or {}
        mask = work_mask > 0
        h, w = mask.shape
        cues = {'coverage': float(mask.mean())}
        if mask.sum() < 50:
            cues.update(dict(green_frac=0.0, leafcolor_frac=0.0, solidity=0.0, elongation=None,
                             serration=0.0, vein_ridge=0.0, texture_energy=None, hue_std=None,
                             convexity_ratio=None))
            return cues
        cm = ColorMaps(work_bgr)
        green = cm.green_tissue(); les = cm.lesion_colored()
        cues['green_frac'] = float(green[mask].mean())
        cues['leafcolor_frac'] = float((green | les)[mask].mean())
        cues['skin_like_frac'] = float((((cm.H < 20) | (cm.H > 170)) & (cm.S > 40) & (cm.S < 170)
                                        & (cm.V > 80) & (cm.a > 8) & (cm.b > 8))[mask].mean())
        u8 = mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c) + 1e-6
        hull = cv2.convexHull(c)
        cues['solidity'] = float(area / (cv2.contourArea(hull) + 1e-6))
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        cues['elongation'] = float(max(rw, rh) / max(1e-6, min(rw, rh)))
        per = cv2.arcLength(c, True); hper = cv2.arcLength(hull, True) + 1e-6
        cues['convexity_ratio'] = float(per / hper)          # 1.0 smooth convex, >1 wavy/serrated
        # serration / lobing: significant convexity defects per unit perimeter
        eq_d = np.sqrt(4 * area / np.pi)
        idx = cv2.convexHull(c, returnPoints=False)
        n_def = 0
        try:
            d = cv2.convexityDefects(c, idx)
            if d is not None:
                n_def = int((d[:, 0, 3] / 256.0 > 0.012 * eq_d).sum())
        except cv2.error:
            pass
        cues['n_margin_teeth'] = n_def
        cues['serration'] = float(np.clip(n_def / 6.0, 0, 1))
        # vein / ridge structure: dark-on-light OR light-on-dark thin ridges
        L = cm.L
        g1 = cv2.GaussianBlur(L, (0, 0), 1.2); g2 = cv2.GaussianBlur(L, (0, 0), 3.0)
        dog = np.abs(g1 - g2)
        inner = cv2.erode(u8, np.ones((7, 7), np.uint8)) > 0
        if inner.sum() < 30:
            inner = mask
        cues['vein_ridge'] = float(dog[inner].mean())
        tex = np.abs(cv2.Laplacian(cv2.GaussianBlur(L, (0, 0), 0.8), cv2.CV_32F))
        cues['texture_energy'] = float(tex[inner].mean())
        # orientation coherence of fine texture: grass/straw = one dominant
        # orientation everywhere; leaf veins = branching, low coherence
        gx = cv2.Sobel(g1, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g1, cv2.CV_32F, 0, 1)
        jxx, jyy, jxy = (cv2.GaussianBlur(gx * gx, (0, 0), 4), cv2.GaussianBlur(gy * gy, (0, 0), 4),
                         cv2.GaussianBlur(gx * gy, (0, 0), 4))
        coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-6)
        cues['orientation_coherence'] = float(coh[inner].mean())
        hue = cm.H[mask]
        cues['hue_std'] = float(np.std(hue))
        # boundary contrast: a real object has a visible edge all round; a
        # texture cluster carved out of grass/foliage does not
        k = np.ones((5, 5), np.uint8)
        inner_ring = mask & ~(cv2.erode(u8, k, iterations=2) > 0)
        outer_ring = (cv2.dilate(u8, k, iterations=2) > 0) & ~mask
        if inner_ring.sum() > 20 and outer_ring.sum() > 20:
            lab_in = np.array([cm.L[inner_ring].mean(), cm.a[inner_ring].mean(), cm.b[inner_ring].mean()])
            lab_out = np.array([cm.L[outer_ring].mean(), cm.a[outer_ring].mean(), cm.b[outer_ring].mean()])
            cues['boundary_contrast'] = float(np.linalg.norm(lab_in - lab_out))
            t_in, t_out = tex[inner].mean(), tex[outer_ring].mean()
            cues['texture_contrast'] = float((t_out - t_in) / (t_out + t_in + 1e-6))
        else:
            cues['boundary_contrast'] = 100.0   # object fills / touches whole frame: no outside to compare
            cues['texture_contrast'] = 0.0
        cues['border_contact'] = float(np.mean([mask[0].mean(), mask[-1].mean(), mask[:, 0].mean(), mask[:, -1].mean()]))
        cues['segmentation_confidence'] = float(seg.get('segmentation_confidence', 0.0))
        cues['green_background_mode'] = bool(seg.get('green_background_mode', False))
        cues['used_fallback_seed'] = bool(seg.get('used_fallback_seed', False))
        return cues


class TomatoLeafGateV21:
    DEFAULT_BANDS = {
        # cue: (lo, hi, soft) -- the "plausible single tomato leaf/leaflet" band
        'coverage': (0.04, 0.95, 0.04),
        'green_frac': (0.30, 1.00, 0.25),
        'leafcolor_frac': (0.80, 1.00, 0.20),
        'solidity': (0.55, 0.985, 0.20),
        'elongation': (1.0, 4.5, 2.0),
        'convexity_ratio': (1.02, 1.80, 0.25),
        'vein_ridge': (1.2, 14.0, 1.0),
        'texture_energy': (1.5, 22.0, 6.0),
        'orientation_coherence': (0.0, 0.70, 0.20),
        'boundary_contrast': (12.0, 400.0, 6.0),
    }
    CUE_WEIGHTS = {'green_frac': 0.16, 'leafcolor_frac': 0.12, 'solidity': 0.14, 'elongation': 0.08,
                   'convexity_ratio': 0.10, 'serration': 0.08, 'vein_ridge': 0.12,
                   'texture_energy': 0.08, 'orientation_coherence': 0.04, 'boundary_contrast': 0.08}

    def __init__(self, preprocessor, accept_threshold=0.62, bands=None,
                 deep_thresholds=None, deep_weight=0.60):
        self.preprocessor = preprocessor
        self.quality = ImageQualityCheckV21()
        self.cues = LeafCueExtractorV21()
        self.accept_threshold = float(accept_threshold)
        self.bands = dict(self.DEFAULT_BANDS, **(bands or {}))
        # filled in by Cell 9.5 calibration; None = deep stage not available yet
        self.deep_thresholds = deep_thresholds or {}
        self.deep_weight = float(deep_weight)

    # -------------------------------------------------------------- messages
    @staticmethod
    def _what_is_it(comp):
        """Only name a specific background type when it clearly dominates the
        frame (>55%); otherwise stay generic rather than guess wrong."""
        if comp:
            k = max(comp, key=comp.get)
            if comp[k] > 0.55 and k != 'other':
                return {'soil_brown': 'bare soil / ground', 'stone_grey': 'stones, gravel or a plain grey surface',
                        'vegetation_green': 'grass or mixed vegetation'}[k]
        return 'an object or scene without a tomato leaf'

    def _heuristic_score(self, c):
        sub = {}
        for k, wgt in self.CUE_WEIGHTS.items():
            if k == 'serration':
                sub[k] = float(np.clip(0.4 + 0.6 * c.get('serration', 0.0), 0, 1))
                continue
            lo, hi, soft = self.bands[k]
            sub[k] = _band_score(c.get(k), lo, hi, soft)
        score = sum(self.CUE_WEIGHTS[k] * sub[k] for k in sub) / sum(self.CUE_WEIGHTS.values())
        return float(score), sub

    def _deep_score(self, deep):
        """deep: dict with any of
             'ood_gate_prob'  P(tomato) from the trained OOD gate CNN (Cell 4.3)
             'maha_score'     in [0,1], 1 = typical tomato embedding (Cell 9.5)
             'energy_score'   in [0,1], 1 = typical tomato energy   (Cell 9.5)
             'msp'            max calibrated softmax of the classifier
           Returns (score in [0,1] or None, list of veto strings)."""
        if not deep:
            return None, []
        th = self.deep_thresholds
        vetoes, parts, wts = [], [], []
        for key, w in (('ood_gate_prob', 0.40), ('maha_score', 0.40), ('energy_score', 0.10), ('msp', 0.10)):
            v = deep.get(key)
            if v is None:
                continue
            parts.append(float(v)); wts.append(w)
            t = th.get(key)
            if t is not None and v < t:
                vetoes.append(key)
        if not parts:
            return None, []
        return float(np.dot(parts, wts) / np.sum(wts)), vetoes

    # --------------------------------------------------------------- evaluate
    def evaluate(self, img_bgr, deep_scores=None, cnn_leaf_confidence=None, ood_score=None, seg=None):
        # backward-compatible hooks from the v8 gate signature
        deep_scores = dict(deep_scores or {})
        if cnn_leaf_confidence is not None:
            deep_scores.setdefault('ood_gate_prob', cnn_leaf_confidence)
        if ood_score is not None:
            deep_scores.setdefault('energy_score', ood_score)

        reasons, veto = [], None
        q = self.quality.check(img_bgr)
        if not q['passed']:
            msg = {'too_blurry': 'The photo is too blurry. Hold the camera steady and focus on one leaf.',
                   'too_dark': 'The photo is too dark. Take it in daylight or with more light.',
                   'overexposed': 'The photo is over-exposed (too bright). Avoid direct glare on the leaf.',
                   'resolution_too_low': 'The image resolution is too low. Use a photo of at least 224x224 pixels.',
                   'blank_or_solid_colour': 'The image is blank or a single colour.'}.get(q['reason'], 'The image could not be read.')
            return self._result('REJECT', msg, [f"quality:{q['reason']}"], q, None, {}, None, None, None)

        seg = seg if seg is not None else self.preprocessor.process(img_bgr)   # reuse a precomputed segmentation
        wimg, wmask = seg['stages']['work_image'], seg['stages']['work_mask']
        c = self.cues.compute(wimg, wmask, seg)
        comp = seg.get('background_composition', {})

        # Stage 1 -- is there a leaf at all?
        if seg['rejection_reason'] == 'no_leaf_found' or c['coverage'] < self.bands['coverage'][0] \
                or c.get('green_frac', 0) < 0.12:
            what = self._what_is_it(comp) if c['coverage'] < 0.5 else 'an object that is not a green leaf'
            veto = 'no_leaf'
            reasons.append(f"no_leaf:coverage={c['coverage']:.3f},green_frac={c.get('green_frac', 0):.2f}")
            msg = f'No leaf detected. The image appears to show {what}. Please photograph a single tomato leaf.'
        # Stage 2 -- is it leaf-SHAPED and leaf-TEXTURED?
        elif c.get('skin_like_frac', 0) > 0.35:
            veto = 'skin'
            reasons.append(f"skin_like:{c['skin_like_frac']:.2f}")
            msg = 'This looks like skin / a hand, not a leaf. Please photograph the leaf on its own.'
        elif c['solidity'] < 0.40:
            veto = 'not_leaf_structure'
            reasons.append(f"structure:solidity={c['solidity']:.2f}")
            msg = ('A green region was found, but its shape and texture look like grass, weeds or scattered '
                   'plants, not one leaf. Please photograph one tomato leaf filling most of the frame.')
        elif c.get('vein_ridge', 0) < 1.2 and (c.get('texture_energy') or 0) < 1.0:
            veto = 'featureless_object'
            reasons.append(f"featureless:vein_ridge={c.get('vein_ridge', 0):.2f},texture={c.get('texture_energy', 0):.2f}")
            msg = ('A green object was found, but it has no leaf veins or leaf surface texture (it looks like '
                   'plastic, paper, a ball or a painted surface). Please photograph a real tomato leaf.')
        elif c['solidity'] > 0.975 and c.get('n_margin_teeth', 0) <= 1 and (c.get('convexity_ratio') or 1) < 1.06:
            veto = 'geometric_shape'
            reasons.append(f"geometric:solidity={c['solidity']:.3f},teeth={c.get('n_margin_teeth', 0)}")
            msg = ('The object has a perfectly regular geometric outline (circle / rectangle), not the toothed '
                   'outline of a tomato leaflet. Please photograph a tomato leaf.')

        h_score, sub = self._heuristic_score(c)
        # Green-on-green (leaf against grass / other foliage) with a weak object boundary:
        # NOT a hard veto -- real tomato leaves photographed against foliage look like this
        # too (tested: boundary contrast 3-10 on genuine leaf-on-grass composites, 9.2 on a
        # pure grass photo). The score is lowered and the deep tomato-identity stage
        # (Cell 9.5) makes the call.
        if c.get('green_background_mode') and c.get('boundary_contrast', 100) < 11.0 \
                and (c.get('texture_contrast') or 0) < 0.15:
            h_score *= 0.85
            reasons.append('weak_object_boundary_on_green_background:needs_deep_confirmation')
        d_score, d_vetoes = self._deep_score(deep_scores)
        if d_score is None:
            conf = h_score
        else:
            conf = (1 - self.deep_weight) * h_score + self.deep_weight * d_score

        if veto is None and d_vetoes:
            veto = 'not_tomato'
            reasons.append('deep_veto:' + ','.join(d_vetoes))
            msg = ('A leaf was detected, but its features do not match a tomato leaf (it is probably another '
                   'plant species or an unusual object). Please upload a tomato leaf.')
        if veto is None and conf < self.accept_threshold:
            veto = 'low_confidence'
            weak = sorted(sub, key=sub.get)[:3]
            reasons.append(f'confidence={conf:.3f}<{self.accept_threshold}; weakest cues: {weak}')
            msg = ('The image does not look enough like a single tomato leaf '
                   f'(confidence {conf:.0%}). Please retake the photo: one leaf, in focus, filling the frame.')
        if veto is None:
            return self._result('ACCEPT', 'Tomato leaf accepted. Proceeding to disease analysis.', [], q, seg, c,
                                h_score, d_score, conf, sub)
        return self._result('REJECT', msg, reasons, q, seg, c, h_score, d_score, conf, sub, veto)

    @staticmethod
    def _result(decision, message, reasons, q, seg, cues, h, d, conf, sub=None, veto=None):
        return {'decision': decision, 'message': message, 'reasons': reasons, 'veto': veto,
                'quality_check': q, 'cues': cues, 'cue_scores': sub or {},
                'heuristic_score': h, 'deep_score': d, 'confidence': conf,
                'segmentation': None if seg is None else {
                    k: seg[k] for k in ('leaf_coverage', 'segmentation_confidence', 'is_reliable',
                                        'rejection_reason', 'background_composition', 'lesion_fraction')},
                '_seg_result': seg}
