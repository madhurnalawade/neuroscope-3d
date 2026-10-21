"""Overlap and distance metrics between a predicted and a reference BraTS label map, per WT/TC/ET region."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .labels import region_masks


def dice(pred: np.ndarray, ref: np.ndarray):
    """2|P∩R| / (|P|+|R|). Both empty -> 1.0 (BraTS convention)."""
    s = int(pred.sum()) + int(ref.sum())
    return 1.0 if s == 0 else float(2.0 * np.logical_and(pred, ref).sum() / s)


def sensitivity(pred: np.ndarray, ref: np.ndarray):
    """|P∩R| / |R| (true-positive rate). Undefined (None) when the reference region is empty."""
    r = int(ref.sum())
    return None if r == 0 else float(np.logical_and(pred, ref).sum() / r)


def precision(pred: np.ndarray, ref: np.ndarray):
    """|P∩R| / |P| (positive predictive value). Undefined (None) when the prediction is empty."""
    p = int(pred.sum())
    return None if p == 0 else float(np.logical_and(pred, ref).sum() / p)


def iou(pred: np.ndarray, ref: np.ndarray):
    """|P∩R| / |P∪R| (Jaccard). Both empty -> 1.0, matching the Dice convention."""
    u = int(np.logical_or(pred, ref).sum())
    return 1.0 if u == 0 else float(np.logical_and(pred, ref).sum() / u)


def _surface(mask):
    return mask & ~binary_erosion(mask, iterations=1, border_value=0)


def hd95(pred: np.ndarray, ref: np.ndarray, spacing):
    """95th-percentile symmetric Hausdorff distance in mm (max of the two directed 95th percentiles).
    Both empty -> 0.0; exactly one empty -> None (undefined)."""
    if not pred.any() and not ref.any():
        return 0.0
    if not pred.any() or not ref.any():
        return None
    sp = [float(s) for s in spacing[:3]]
    sp_, sr = _surface(pred), _surface(ref)
    d_to_ref = distance_transform_edt(~sr, sampling=sp)[sp_]
    d_to_pred = distance_transform_edt(~sp_, sampling=sp)[sr]
    return float(max(np.percentile(d_to_ref, 95), np.percentile(d_to_pred, 95)))


def case_metrics(pred_labels: np.ndarray, ref_labels: np.ndarray, spacing, with_hd95: bool = True) -> dict:
    p, r = region_masks(pred_labels), region_masks(ref_labels)
    out = {}
    for name in ("WT", "TC", "ET"):
        out[name] = {
            "dice": dice(p[name], r[name]),
            "sensitivity": sensitivity(p[name], r[name]),
            "precision": precision(p[name], r[name]),
            "iou": iou(p[name], r[name]),
            "hd95_mm": hd95(p[name], r[name], spacing) if with_hd95 else None,
            "pred_voxels": int(p[name].sum()),
            "ref_voxels": int(r[name].sum()),
        }
    return out
