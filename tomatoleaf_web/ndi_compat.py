"""The four scipy.ndimage functions the v21 segmentation uses, re-implemented with
OpenCV + NumPy so the web deployment does not ship SciPy (~110 MB). Same call
signatures and semantics for the way seg_v21.py uses them (2-D boolean/int masks,
8-connectivity is NOT used by scipy's default -> 4-connectivity here too)."""
import cv2
import numpy as np


def label(mask):
    """scipy.ndimage.label with the default (4-connected) structure -> (labels, n)."""
    n, lab = cv2.connectedComponents(np.asarray(mask, np.uint8), connectivity=4)
    return lab.astype(np.int32), int(n - 1)


def binary_fill_holes(mask):
    """Fills background regions not connected to the image border (4-connected
    background, like scipy's default)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return m.copy()
    h, w = m.shape
    bg = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    outside = np.isin(lab, border[border > 0]) & (bg > 0)
    return ~outside


def _index_array(labels, index):
    return np.atleast_1d(np.asarray(index, np.int64))


def sum(values, labels, index):  # noqa: A001 - mirrors scipy's name
    lab = np.asarray(labels, np.int64).ravel()
    val = np.asarray(values, np.float64).ravel()
    tot = np.bincount(lab, weights=val, minlength=int(lab.max()) + 1 if lab.size else 1)
    idx = _index_array(labels, index)
    out = np.array([tot[i] if i < len(tot) else 0.0 for i in idx])
    return out if np.ndim(index) else out[0]


def center_of_mass(values, labels, index):
    lab = np.asarray(labels, np.int64)
    val = np.asarray(values, np.float64)
    h, w = lab.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = int(lab.max()) + 1
    tot = np.bincount(lab.ravel(), weights=val.ravel(), minlength=m)
    sy = np.bincount(lab.ravel(), weights=(val * yy).ravel(), minlength=m)
    sx = np.bincount(lab.ravel(), weights=(val * xx).ravel(), minlength=m)
    idx = _index_array(labels, index)
    res = [(sy[i] / tot[i], sx[i] / tot[i]) if i < m and tot[i] > 0 else (np.nan, np.nan) for i in idx]
    return res if np.ndim(index) else res[0]
