"""Explanation helpers shared with notebook Cell 10.1b (TensorFlow-free part).
Grad-CAM itself is computed inside each model's head .tflite (tools/export_litert.py)."""
import cv2
import numpy as np


def upsample(cam, hw):
    m = float(cam.max())
    cam = cam / m if m > 0 else cam
    return np.clip(cv2.resize(cam.astype(np.float32), (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR), 0, 1)


def lesion_proxy_mask(rgb_uint8, leaf_mask_u8, seg_lesion_u8=None, thresh=0.35):
    """Binary lesion-proxy (union of V3's lesion mask and an HSV/Lab colour
    abnormality proxy inside the leaf). NOT ground truth."""
    hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hue_d = np.where(h < 35, 35 - h, np.where(h > 85, h - 85, 0.0))
    abn = 0.45 * np.clip(hue_d / 20.0, 0, 1) * (s > 45) + 0.30 * np.clip((lab[..., 1] - 128) / 25, 0, 1) \
        + 0.25 * np.clip((lab[..., 2] - 128 - 25) / 25, 0, 1)
    abn = np.maximum(abn, np.clip((70 - v) / 40, 0, 1) * 0.8)
    leaf = leaf_mask_u8 > 0
    m = (abn >= thresh) & leaf
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    if seg_lesion_u8 is not None:
        m |= (seg_lesion_u8 > 0) & leaf
    return m


def focus_metrics(cam_full, leaf_mask_u8, lesion_mask_bool=None):
    """All on the RAW (max-normalised, bilinear) CAM."""
    leaf = leaf_mask_u8 > 0
    tot = float(cam_full.sum()) + 1e-8
    r = {'lfs': float((cam_full * leaf).sum() / tot)}
    yx = np.unravel_index(np.argmax(cam_full), cam_full.shape)
    r['pointing_leaf'] = bool(leaf[yx])
    top = cam_full >= np.quantile(cam_full, 0.90)
    r['top10_in_leaf'] = float((top & leaf).sum() / max(1, top.sum()))
    if lesion_mask_bool is not None and lesion_mask_bool.sum() > 20:
        les = lesion_mask_bool
        r['lefs'] = float((cam_full * les).sum() / tot)
        r['lesion_area_frac_of_leaf'] = float(les.sum() / max(1, leaf.sum()))
        # >1 means the CAM concentrates on lesions more than a uniform-on-leaf map would
        r['lesion_enrichment'] = float((r['lefs'] / max(r['lfs'], 1e-6)) / max(r['lesion_area_frac_of_leaf'], 1e-6))
        dil = cv2.dilate(les.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        r['pointing_lesion'] = bool(dil[yx])
        r['top10_in_lesion'] = float((top & dil).sum() / max(1, top.sum()))
    else:
        r.update({'lefs': None, 'lesion_area_frac_of_leaf': 0.0, 'lesion_enrichment': None,
                  'pointing_lesion': None, 'top10_in_lesion': None})
    return r


def disease_regions(cam_full, lesion_mask_bool, min_area=25, top_k=5):
    """Lesion components ranked by the CAM mass they receive -> boxes of the
    regions the model actually used as disease evidence."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lesion_mask_bool.astype(np.uint8), 8)
    regs = []
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] < min_area:
            continue
        mass = float(cam_full[lab == j].sum())
        x, y, w, h = stats[j, :4]
        regs.append({'box': (int(x), int(y), int(w), int(h)), 'area': int(stats[j, 4]), 'cam_mass': mass})
    tot = sum(r['cam_mass'] for r in regs) + 1e-8
    for r in regs:
        r['cam_share'] = r['cam_mass'] / tot
    return sorted(regs, key=lambda r: -r['cam_mass'])[:top_k]


def overlay(rgb, heat, alpha=0.45):
    col = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return np.clip((1 - alpha) * rgb + alpha * col, 0, 255).astype(np.uint8)


def overlay_focus(rgb, heat, max_alpha=0.7):
    """Heat-weighted overlay: only high values are tinted, the rest of the photo stays readable."""
    h = np.clip(heat, 0, 1).astype(np.float32)
    col = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].astype(np.float32)
    a = (max_alpha * np.clip((h - 0.15) / 0.85, 0, 1))[..., None]
    return np.clip((1 - a) * rgb + a * col, 0, 255).astype(np.uint8)


