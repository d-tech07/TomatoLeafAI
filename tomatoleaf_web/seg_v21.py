# Web copy of notebook Cell 2.1b (identical logic; SciPy replaced by ndi_compat).
# CELL 2.1b - BackgroundAwarePreprocessorV3 (v21): lesion-preserving, soil/stone/shadow-aware
# background removal.
#
# WHY A V3 (root causes found in the v9/v20 outputs, not guesses):
#   1. V2 resized every photo straight to 224x224 with cv2.resize -> non-square
#      field photos were STRETCHED before segmentation (Cell 2.3's aspect-safe
#      letterbox ran afterwards on an already-square, already-stretched image,
#      so it never actually did anything when background removal was on).
#   2. V2's 2-of-3 vote (HSV / Lab / adaptive-Otsu) treats "anything different
#      from the local background" as leaf. On a field photo, soil clods, stones,
#      shadows and neighbouring plants all pass the adaptive/Otsu vote, and a
#      big CLOSE kernel then glues them onto the leaf (Cell 2.5 figure: the
#      cast shadow on the left of the leaf was kept as "leaf").
#   3. Nothing in V2 separated "brown because it is a LESION" from "brown
#      because it is SOIL". V3 does this explicitly with a connectivity rule:
#      a non-green region is kept only if it is enclosed by, or mostly bordered
#      by, green leaf tissue (a lesion), and is dropped if it is mostly bordered
#      by background (a soil wedge between leaf lobes, a shadow, a stone).
#   4. No crop: a small leaf in a big field photo stayed small after resize,
#      so the classifier saw ~60x60 useful pixels. V3 crops to the leaf's
#      bounding box (+margin) from the FULL-RESOLUTION original, then
#      letterboxes to 224 -- the leaf fills the frame like a PlantVillage image.
#
# PUBLIC CONTRACT: process(img_bgr) returns every key BackgroundAwarePreprocessorV2
# returned (background_removed, normalized, leaf_mask, leaf_coverage,
# segmentation_confidence, is_reliable, rejection_reason, n_components_found,
# n_components_kept, stages{...}) so Cells 2.2/2.3/2.5/10.x and the Flask app
# keep working unchanged, plus new keys: lesion_mask, lesion_fraction,
# background_composition, crop_box, leaf_cues.
#
# Leaf pixels are NEVER recoloured: inside the (eroded) mask the output is a
# byte-identical copy of the resized crop; only a 1-2 px feathered rim blends
# into the neutral (230,230,230) background that the rest of the pipeline
# (LFSTrainer leaf-mask threshold 0.90, Grad-CAM leaf mask) already assumes.
import cv2
import numpy as np
try:
    from . import ndi_compat as ndi      # web app: no SciPy dependency
except ImportError:
    from scipy import ndimage as ndi


def _odd(k):
    k = int(max(1, round(k)))
    return k if k % 2 == 1 else k + 1


def _ellipse(k):
    k = _odd(k)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def letterbox(img, size, pad_value):
    """Aspect-preserving resize into a (size[1], size[0]) canvas. Returns
    (canvas, (x0, y0, new_w, new_h)). Works for 3-channel images and 2-D masks."""
    tw, th = size
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
    if img.ndim == 2:
        interp = cv2.INTER_LINEAR
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    if img.ndim == 2:
        canvas = np.full((th, tw), pad_value, dtype=img.dtype)
    else:
        canvas = np.full((th, tw, img.shape[2]), pad_value, dtype=img.dtype)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas, (x0, y0, nw, nh)


class ColorMaps:
    """All per-pixel colour evidence used by segmentation AND by the v21 gate
    (computed once, shared). Channel conventions: OpenCV HSV (H 0-180), Lab
    with a*/b* re-centred on 0."""

    def __init__(self, img_bgr):
        f = img_bgr.astype(np.float32)
        B, G, R = f[..., 0], f[..., 1], f[..., 2]
        s = B + G + R + 1e-6
        r, g, b = R / s, G / s, B / s
        self.exg = 2 * g - r - b                      # excess green
        self.exr = 1.4 * r - g                        # excess red
        self.exgr = self.exg - self.exr               # Meyer & Neto (2008) ExG-ExR
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.H, self.S, self.V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        self.L = lab[..., 0]
        self.a = lab[..., 1] - 128.0
        self.b = lab[..., 2] - 128.0
        self.gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    def green_tissue(self):
        """Healthy/chlorophyll-bearing leaf tissue: green hue band, some
        saturation, not black, and green-dominant chromaticity (ExGR>0) or a
        clearly negative Lab a*. Rejects grey stones, brown soil, skin, sky."""
        hue_ok = (self.H >= 22) & (self.H <= 95)
        chroma_ok = (self.S >= 28) & (self.V >= 28)
        green_dom = (self.exgr > -0.03) | (self.a < -7)
        return hue_ok & chroma_ok & green_dom

    def lesion_colored(self):
        """Colours diseased tomato tissue actually takes: chlorotic yellow
        (early/late blight halos, TYLCV, leaf mould upper surface), tan-brown
        necrosis (early blight rings, septoria, target spot, bacterial spot)
        and dark-brown/olive necrosis. Deliberately ALSO matches soil -- the
        connectivity rule in BackgroundAwarePreprocessorV3 is what separates
        them, colour alone cannot."""
        yellow = (self.H >= 14) & (self.H < 35) & (self.S >= 55) & (self.V >= 70)
        brown = (((self.H < 22) | (self.H >= 165)) & (self.S >= 45)
                 & (self.V >= 25) & (self.V <= 215))
        olive_dark = (self.V < 90) & (self.V >= 15) & (self.S >= 40) & (self.b > 2)
        return yellow | brown | olive_dark

    def soil_like(self):
        """Brown/ochre, low-to-mid saturation, granular -> soil/mulch/dry litter."""
        return (((self.H < 25) | (self.H >= 165)) & (self.S >= 25) & (self.S < 170)
                & (self.V >= 30) & (self.V < 200) & (self.exgr < -0.05))

    def stone_like(self):
        """Low-saturation grey/white/beige -> stones, gravel, concrete, paper."""
        return (self.S < 32) & (self.V >= 45)


class BackgroundAwarePreprocessorV3:
    VERSION = 'v21'

    def __init__(self, img_size=(224, 224), hsv_ranges=None,
                 background_color=(230, 230, 230), work_max_side=384,
                 crop_to_leaf=True, crop_margin=0.10, use_grabcut=True,
                 grabcut_iters=3, plausible_coverage_range=(0.03, 0.97),
                 reliability_threshold=0.45, feather_sigma=1.2,
                 min_marginal_lesion_border_frac=0.40, strong_border_frac=0.62,
                 min_lesion_bg_chroma_dist=9.0, flood_coverage=0.55):
        # hsv_ranges accepted (and ignored) only so Cell 2.1's call signature
        # `BackgroundAwarePreprocessorV3(img_size=..., hsv_ranges=CFG['HSV_RANGES'])`
        # works as a drop-in for V2.
        self.size = tuple(img_size)
        self.background_color = tuple(int(c) for c in background_color)
        self.work_max_side = int(work_max_side)
        self.crop_to_leaf = bool(crop_to_leaf)
        self.crop_margin = float(crop_margin)
        self.use_grabcut = bool(use_grabcut)
        self.grabcut_iters = int(grabcut_iters)
        self.plausible_coverage_range = plausible_coverage_range
        self.reliability_threshold = float(reliability_threshold)
        self.feather_sigma = float(feather_sigma)
        self.min_marginal_lesion_border_frac = float(min_marginal_lesion_border_frac)
        self.strong_border_frac = float(strong_border_frac)
        self.min_lesion_bg_chroma_dist = float(min_lesion_bg_chroma_dist)
        self.flood_coverage = float(flood_coverage)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _components(mask):
        lab, n = ndi.label(mask)
        return lab, n

    @staticmethod
    def _hull(mask_u8):
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = np.zeros_like(mask_u8)
        if cnts:
            pts = np.vstack(cnts)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 255)
        return hull

    def _to_work(self, img_bgr):
        h, w = img_bgr.shape[:2]
        s = self.work_max_side / max(h, w)
        if s < 1.0:
            work = cv2.resize(img_bgr, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                              interpolation=cv2.INTER_AREA)
        else:
            s = 1.0
            work = img_bgr.copy()
        return work, s

    # ------------------------------------------------------- 1. leaf seed
    def _select_leaf_seed(self, green, cm):
        h, w = green.shape
        diag = float(np.hypot(h, w))
        g = cv2.morphologyEx(green.astype(np.uint8) * 255, cv2.MORPH_OPEN, _ellipse(3))
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, _ellipse(max(3, diag * 0.012)))
        lab, n = self._components(g > 0)
        if n == 0:
            return None, 0
        idx = np.arange(1, n + 1)
        areas = ndi.sum(np.ones_like(lab), lab, idx)
        cys, cxs = zip(*ndi.center_of_mass(np.ones_like(lab), lab, idx))
        cy, cx = np.array(cys) / h, np.array(cxs) / w
        dist2 = (cy - 0.5) ** 2 + (cx - 0.5) ** 2
        border = np.zeros_like(lab, dtype=bool)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        border_px = ndi.sum(border, lab, idx)
        approx_perimeter = np.maximum(4.0 * np.sqrt(areas), 1.0)
        border_frac = np.clip(border_px / approx_perimeter, 0, 1)
        area_frac = areas / float(h * w)
        score = area_frac * np.exp(-dist2 / (2 * 0.30 ** 2)) * (1.0 - 0.5 * border_frac)
        score[area_frac < 0.004] = 0
        if score.max() <= 0:
            return None, n
        best = int(idx[np.argmax(score)])
        seed = lab == best
        # merge fragments of the SAME leaf that a lesion/vein split off: they
        # must touch the main blob after a small dilation (~2% of the diagonal)
        grown = cv2.dilate(seed.astype(np.uint8), _ellipse(diag * 0.02)) > 0
        ref = np.array([cm.L[seed].mean(), cm.a[seed].mean(), cm.b[seed].mean()])
        for j, a in zip(idx, areas):
            if j == best or a < 0.0015 * h * w:
                continue
            comp = lab == j
            if (comp & grown).any():
                # same leaf => same colour; a touching weed / neighbouring plant is not merged
                col = np.array([cm.L[comp].mean(), cm.a[comp].mean(), cm.b[comp].mean()])
                if np.linalg.norm(col - ref) < 18.0:
                    seed |= comp
        seed = self._detach_touching_objects(seed, cm, diag)
        return seed, n

    def _detach_touching_objects(self, seed, cm, diag):
        """A weed / neighbouring leaf that TOUCHES the leaf joins the same green
        component. Opening with a ~3.5%-diagonal disc cuts the thin contact neck;
        pieces whose colour differs from the main leaf (dE > 18 in Lab) are
        dropped, then fine leaf detail (teeth, tip) near the kept body is restored."""
        k = _ellipse(max(5, diag * 0.035))
        opened = cv2.morphologyEx(seed.astype(np.uint8), cv2.MORPH_OPEN, k) > 0
        lab, n = self._components(opened)
        if n <= 1:
            return seed
        sizes = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
        main = lab == (int(np.argmax(sizes)) + 1)
        ref = np.array([cm.L[main].mean(), cm.a[main].mean(), cm.b[main].mean()])
        keep = main.copy()
        for j in range(1, n + 1):
            comp = lab == j
            if (comp & main).any():
                continue
            col = np.array([cm.L[comp].mean(), cm.a[comp].mean(), cm.b[comp].mean()])
            if np.linalg.norm(col - ref) < 18.0:
                keep |= comp
        if keep.sum() == opened.sum():
            return seed
        return seed & (cv2.dilate(keep.astype(np.uint8), k) > 0)

    def _fallback_seed(self, cm, img_bgr):
        """No green at all (e.g. a fully necrotic/yellow leaf): take the central
        object that differs most from the image border's colour."""
        h, w = cm.L.shape
        lab = np.dstack([cm.L, cm.a, cm.b])
        border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        ref = np.median(border, axis=0)
        d = np.linalg.norm(lab - ref, axis=-1)
        d8 = cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, th = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, _ellipse(5))
        lab_c, n = self._components(th > 0)
        if n == 0:
            return None
        idx = np.arange(1, n + 1)
        areas = ndi.sum(np.ones_like(lab_c), lab_c, idx)
        coms = ndi.center_of_mass(np.ones_like(lab_c), lab_c, idx)
        best, best_s = None, 0.0
        for j, a, (yy, xx) in zip(idx, areas, coms):
            s = (a / (h * w)) * np.exp(-(((yy / h) - .5) ** 2 + ((xx / w) - .5) ** 2) / (2 * .3 ** 2))
            if s > best_s:
                best, best_s = j, s
        return None if best is None else (lab_c == best)

    # ------------------------------------------ 1b. green-background mode
    @staticmethod
    def _touches_sides(mask):
        return int(mask[0, :].any()) + int(mask[-1, :].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())

    def _flood_seed(self, img_bgr, cm):
        """The whole frame is green (leaf lying on grass / in front of other
        foliage), so colour-thresholding alone finds one blob = the image.
        Separate the leaf by colour + TEXTURE: k-means on smoothed Lab plus
        local-contrast energy, pick the cluster component that looks most
        like one object (central, compact, not wrapped round the frame), then
        re-attach neighbouring components that are equally SMOOTH (a leaf
        whose halves differ in colour -- e.g. a yellowing TYLCV leaf -- is
        split by k-means, but both halves are smooth while grass is not)."""
        h, w = cm.L.shape
        Ls = cv2.GaussianBlur(cm.L, (0, 0), 2)
        tex = np.sqrt(np.maximum(cv2.GaussianBlur(cm.L ** 2, (0, 0), 3) - cv2.GaussianBlur(cm.L, (0, 0), 3) ** 2, 0))
        tex = cv2.GaussianBlur(tex, (0, 0), 4)
        feats = [Ls / 10.0, cv2.GaussianBlur(cm.a, (0, 0), 2) / 5.0,
                 cv2.GaussianBlur(cm.b, (0, 0), 2) / 5.0, tex / 3.0]
        X = np.stack(feats, -1).reshape(-1, 4).astype(np.float32)
        green, lesc = cm.green_tissue(), cm.lesion_colored()
        best, best_s, best_pool = None, 0.0, None
        for k in (2, 3, 4):
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
            _, lbl, _ = cv2.kmeans(X, k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
            lbl = lbl.reshape(h, w)
            pool = []
            for c in range(k):
                m = cv2.morphologyEx((lbl == c).astype(np.uint8), cv2.MORPH_OPEN, _ellipse(5))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ellipse(7)) > 0
                comp_lab, n = self._components(m)
                for j in range(1, n + 1):
                    comp = ndi.binary_fill_holes(comp_lab == j)
                    if comp.mean() >= 0.005:
                        pool.append(comp)
            for comp in pool:
                af = comp.mean()
                if af < 0.03 or af > 0.85 or self._touches_sides(comp) >= 3:
                    continue
                if not (green[comp].mean() > 0.35 or lesc[comp].mean() > 0.35):
                    continue
                yy, xx = np.where(comp)
                cen = np.exp(-(((yy.mean() / h) - .5) ** 2 + ((xx.mean() / w) - .5) ** 2) / (2 * .3 ** 2))
                hull_a = float((self._hull(comp.astype(np.uint8) * 255) > 0).sum()) + 1e-6
                sc = af ** 0.5 * cen * (comp.sum() / hull_a) * (1 - 0.25 * self._touches_sides(comp))
                sc *= 1.0 / (1.0 + tex[comp].mean() / (tex.mean() + 1e-6))   # smoother = more leaf-like
                if sc > best_s:
                    best, best_s, best_pool = comp, sc, pool
        if best is None:
            return None
        t_best = tex[best].mean()
        out = best.copy()
        for _ in range(2):
            grown = cv2.dilate(out.astype(np.uint8), _ellipse(7)) > 0
            for comp in best_pool:
                if (comp & out).sum() > 0.5 * comp.sum() or not (comp & grown).any():
                    continue
                if self._touches_sides(comp) >= 2 or comp.mean() > 0.5:
                    continue
                if tex[comp].mean() <= 1.25 * t_best and (green[comp].mean() + lesc[comp].mean()) > 0.4:
                    out |= comp
        return ndi.binary_fill_holes(out)

    @staticmethod
    def _feature_image(cm):
        """3-channel uint8 'image' of (lightness, yellowness b*, local texture)
        -- lets GrabCut's colour GMMs separate a smooth leaf from high-texture
        grass/foliage even when both are the same green."""
        tex = np.sqrt(np.maximum(cv2.GaussianBlur(cm.L ** 2, (0, 0), 2.5)
                                 - cv2.GaussianBlur(cm.L, (0, 0), 2.5) ** 2, 0))
        ch = [cv2.GaussianBlur(cm.L, (0, 0), 1.5), cv2.GaussianBlur(cm.b, (0, 0), 1.5), tex]
        return np.dstack([cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX) for c in ch]).astype(np.uint8)

    # ------------------------------------------- 2. lesion-preserving growth
    def _add_lesions(self, leaf, cm):
        """Enclosed non-green regions (lesions fully inside the leaf) come in
        via hole filling. Marginal non-green regions (lesions on the leaf
        edge -- very common for early blight / leaf-edge necrosis) are added
        only if they are lesion-coloured, inside the leaf's convex hull, and
        EITHER mostly bordered by green leaf tissue (>= strong_border_frac)
        OR partly bordered by leaf (>= min_marginal_lesion_border_frac) AND
        chromatically different from the background right next to them.
        Soil wedges between lobes and cast shadows fail both tests: they are
        mostly bordered by background and share its chromaticity (a shadow is
        a darker copy of the same soil colour, so its Lab a*/b* barely moves)."""
        filled = ndi.binary_fill_holes(leaf)
        enclosed = filled & ~leaf
        hull = self._hull(filled.astype(np.uint8) * 255) > 0
        cand = cm.lesion_colored() & hull & ~filled
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(3)) > 0
        lab, n = self._components(cand)
        added = np.zeros_like(leaf)
        near_leaf = cv2.dilate(filled.astype(np.uint8), _ellipse(5)) > 0
        for j in range(1, n + 1):
            comp = lab == j
            ring = (cv2.dilate(comp.astype(np.uint8), _ellipse(5)) > 0) & ~comp
            if ring.sum() == 0:
                continue
            frac_leaf = float((ring & filled).sum()) / float(ring.sum())
            if frac_leaf >= self.strong_border_frac:
                added |= comp
                continue
            if frac_leaf < self.min_marginal_lesion_border_frac:
                continue
            around = (cv2.dilate(comp.astype(np.uint8), _ellipse(31)) > 0) & ~near_leaf & ~cand
            if around.sum() < 20:
                added |= comp
                continue
            d_ab = float(np.hypot(cm.a[comp].mean() - np.median(cm.a[around]),
                                  cm.b[comp].mean() - np.median(cm.b[around])))
            if d_ab >= self.min_lesion_bg_chroma_dist:
                added |= comp
        grown = ndi.binary_fill_holes(filled | added)
        lesion = (enclosed | added | (grown & ~filled)) & ~leaf
        return grown, lesion

    # ------------------------------------------------------- 3. GrabCut edge
    def _grabcut(self, img_bgr, leaf, lesion, band_frac=0.035, iters=None):
        h, w = leaf.shape
        diag = float(np.hypot(h, w))
        u8 = leaf.astype(np.uint8)
        sure_fg = cv2.erode(u8, _ellipse(diag * 0.025)) > 0
        band = cv2.dilate(u8, _ellipse(diag * band_frac)) > 0
        gc = np.full((h, w), cv2.GC_BGD, np.uint8)
        gc[band] = cv2.GC_PR_BGD
        gc[leaf] = cv2.GC_PR_FGD
        gc[sure_fg | lesion] = cv2.GC_FGD
        if not sure_fg.any() or (gc == cv2.GC_BGD).sum() < 50:
            return leaf, 1.0
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(img_bgr, gc, None, bgd, fgd, iters or self.grabcut_iters, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            return leaf, 1.0
        out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)) & band
        out |= sure_fg | lesion            # GrabCut may trim the rim, never the core or a lesion
        lab, n = self._components(out)
        if n > 1:
            keep = np.unique(lab[sure_fg | lesion])
            out = np.isin(lab, keep[keep > 0])
        out = ndi.binary_fill_holes(out)
        inter = float((out & leaf).sum()); uni = float((out | leaf).sum()) + 1e-6
        return out, inter / uni

    # ---------------------------------------------------- 4. quality numbers
    def _confidence(self, mask, cm, agreement):
        h, w = mask.shape
        area = float(mask.sum())
        cov = area / (h * w)
        if area < 10:
            return 0.0, {}
        u8 = mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(c)) + 1e-6
        solidity = float(cv2.contourArea(c) / hull_area)
        inside_g = float(cm.exgr[mask].mean())
        outside = ~mask
        outside_g = float(cm.exgr[outside].mean()) if outside.any() else inside_g
        separation = float(np.clip((inside_g - outside_g) / 0.25, 0, 1))
        cov_ok = 1.0 if 0.05 <= cov <= 0.92 else 0.5
        sol_ok = float(np.clip((solidity - 0.35) / 0.4, 0, 1))
        conf = 0.30 * separation + 0.25 * sol_ok + 0.25 * agreement + 0.20 * cov_ok
        return float(np.clip(conf, 0, 1)), {
            'solidity': solidity, 'fg_bg_greenness_separation': separation,
            'grabcut_agreement_iou': agreement, 'coverage_work': cov}

    def _background_composition(self, mask, cm):
        bg = ~mask
        n = float(bg.sum()) + 1e-6
        green = cm.green_tissue() & bg
        soil = cm.soil_like() & bg
        stone = cm.stone_like() & bg & ~soil
        return {'vegetation_green': float(green.sum() / n), 'soil_brown': float(soil.sum() / n),
                'stone_grey': float(stone.sum() / n),
                'other': float(max(0.0, 1 - (green.sum() + soil.sum() + stone.sum()) / n))}

    # --------------------------------------------------------------- process
    def process(self, img_bgr):
        if img_bgr is None or img_bgr.ndim != 3:
            raise ValueError('process() expects a BGR uint8 image (H,W,3)')
        work, s = self._to_work(img_bgr)
        work_blur = cv2.bilateralFilter(work, 5, 30, 5)
        cm = ColorMaps(work_blur)
        green = cm.green_tissue()

        seed, n_found = self._select_leaf_seed(green, cm)
        used_fallback = False
        if seed is None or seed.sum() < 0.01 * seed.size:
            fb = self._fallback_seed(cm, work_blur)
            if fb is not None and (seed is None or fb.sum() > seed.sum()):
                seed, used_fallback = fb, True
        flood_mode = False
        if seed is not None and seed.mean() > self.flood_coverage and self._touches_sides(seed) >= 3:
            fs = self._flood_seed(work_blur, cm)
            if fs is not None:
                seed, flood_mode = fs, True
        rejection_reason = None
        if seed is None or seed.sum() == 0:
            wmask = np.zeros(work.shape[:2], bool)
            lesion_w = np.zeros_like(wmask)
            agreement = 0.0
            rejection_reason = 'no_leaf_found'
        else:
            grown, lesion_w = self._add_lesions(seed, cm)
            if self.use_grabcut and flood_mode:
                wmask, agreement = self._grabcut(self._feature_image(cm), grown, lesion_w,
                                                 band_frac=0.12, iters=5)
            elif self.use_grabcut:
                wmask, agreement = self._grabcut(work, grown, lesion_w)
            else:
                wmask, agreement = grown, 1.0
            wmask = cv2.morphologyEx(wmask.astype(np.uint8), cv2.MORPH_OPEN, _ellipse(3)) > 0
            wmask = cv2.medianBlur(wmask.astype(np.uint8) * 255, 5) > 127
            wmask = ndi.binary_fill_holes(wmask | lesion_w)
            lab, n = self._components(wmask)
            if n > 1:
                sizes = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
                keep = [j + 1 for j, a in enumerate(sizes) if a >= 0.15 * sizes.max()]
                wmask = np.isin(lab, keep)
            lesion_w = lesion_w & wmask

        conf, sub = self._confidence(wmask, cm, agreement)
        composition = self._background_composition(wmask, cm)
        cov_work = float(wmask.mean())

        # ---- crop to leaf (full-resolution original) and letterbox ---------
        H0, W0 = img_bgr.shape[:2]
        if self.crop_to_leaf and wmask.any():
            ys, xs = np.where(wmask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            mh, mw = (y1 - y0) * self.crop_margin, (x1 - x0) * self.crop_margin
            y0, y1 = max(0, int(y0 - mh)), min(wmask.shape[0], int(np.ceil(y1 + mh)))
            x0, x1 = max(0, int(x0 - mw)), min(wmask.shape[1], int(np.ceil(x1 + mw)))
        else:
            y0, y1, x0, x1 = 0, wmask.shape[0], 0, wmask.shape[1]
        oy0, oy1 = int(round(y0 / s)), min(H0, int(round(y1 / s)))
        ox0, ox1 = int(round(x0 / s)), min(W0, int(round(x1 / s)))
        crop = img_bgr[oy0:oy1, ox0:ox1]
        crop_mask = cv2.resize(wmask[y0:y1, x0:x1].astype(np.float32), (crop.shape[1], crop.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
        crop_les = cv2.resize(lesion_w[y0:y1, x0:x1].astype(np.float32), (crop.shape[1], crop.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        resized, _ = letterbox(crop, self.size, self.background_color)
        m_out, _ = letterbox(crop_mask, self.size, 0.0)
        les_out, _ = letterbox(crop_les, self.size, 0.0)
        leaf_mask = (m_out >= 0.5).astype(np.uint8) * 255
        lesion_mask = ((les_out >= 0.5) & (leaf_mask > 0)).astype(np.uint8) * 255

        # ---- composite: interior byte-identical, feathered 1-2 px rim ------
        hard = leaf_mask > 0
        alpha = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), self.feather_sigma) if self.feather_sigma > 0 \
            else hard.astype(np.float32)
        interior = cv2.erode(hard.astype(np.uint8), _ellipse(3)) > 0
        alpha[interior] = 1.0
        alpha = np.clip(alpha, 0, 1)[..., None]
        bgc = np.array(self.background_color, np.float32)[None, None, :]
        comp = (alpha * resized.astype(np.float32) + (1 - alpha) * bgc)
        bg_removed = np.clip(np.round(comp), 0, 255).astype(np.uint8)
        bg_removed[interior] = resized[interior]

        coverage = float(hard.mean())
        is_reliable = rejection_reason is None
        if is_reliable:
            lo, hi = self.plausible_coverage_range
            if cov_work < lo:
                is_reliable, rejection_reason = False, 'coverage_too_low_likely_failed_segmentation'
            elif cov_work > hi:
                is_reliable, rejection_reason = False, 'coverage_too_high_likely_no_background_present'
            elif conf < self.reliability_threshold:
                is_reliable, rejection_reason = False, 'low_segmentation_confidence'

        overlay = resized.copy()
        tint = np.zeros_like(overlay); tint[hard] = (0, 200, 80); tint[lesion_mask > 0] = (0, 60, 255)
        overlay = cv2.addWeighted(overlay, 0.65, tint, 0.35, 0)
        g_out, _ = letterbox(green[y0:y1, x0:x1].astype(np.float32), self.size, 0.0)
        stages = {
            'resized': resized,
            'green_tissue_mask': cv2.cvtColor(((g_out > 0.5) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
            'lesion_added_mask': cv2.cvtColor(lesion_mask, cv2.COLOR_GRAY2BGR),
            'final_leaf_mask': cv2.cvtColor(leaf_mask, cv2.COLOR_GRAY2BGR),
            'background_removed': bg_removed,
            'overlay': overlay,
            'work_image': work, 'work_mask': (wmask * 255).astype(np.uint8),
        }
        return {
            'background_removed': bg_removed,
            'normalized': bg_removed.astype(np.float32) / 255.0,
            'leaf_mask': leaf_mask,
            'lesion_mask': lesion_mask,
            'lesion_fraction': float((lesion_mask > 0).sum() / max(1, hard.sum())),
            'leaf_coverage': coverage,
            'leaf_coverage_in_original': cov_work,
            'segmentation_confidence': conf,
            'segmentation_subscores': sub,
            'is_reliable': bool(is_reliable),
            'rejection_reason': rejection_reason,
            'n_components_found': int(n_found),
            'n_components_kept': int(ndi.label(hard)[1]),
            'used_fallback_seed': used_fallback,
            'green_background_mode': flood_mode,
            'background_composition': composition,
            'crop_box': (ox0, oy0, ox1, oy1),
            'stages': stages,
        }

    def process_path(self, p):
        img = cv2.imread(str(p))
        if img is None:
            raise IOError(f'Cannot read: {p}')
        return self.process(img)
