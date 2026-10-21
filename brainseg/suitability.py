"""Heuristic input-suitability checks: is this plausibly a skull-stripped, co-registered brain MRI?

These are coarse sanity checks, not quality assessment. They refuse inputs that clearly do not match
what the model was trained on (BraTS-style skull-stripped, co-registered, ~1 mm scans) and warn on
borderline ones.
"""
import numpy as np
from scipy.ndimage import binary_erosion, sobel

# Skull-stripped adult brains are roughly 1.0-1.8 L; a head including skull/scalp is typically > 3 L.
REFUSE_NONZERO_ML = 2600.0
WARN_NONZERO_ML_HIGH = 2000.0
WARN_NONZERO_ML_LOW = 600.0
REFUSE_MIN_ZERO_FRACTION = 0.05   # skull-stripped volumes have an exactly-zero background
WARN_MASK_AGREEMENT = 0.90        # Dice between each sequence's non-zero mask and the union
WARN_MAX_SPACING_MM = 2.0
# "Is this a brain?" heuristics, calibrated on 41 Decathlon brains (extent 129-151 x 157-176 x 129-145 mm,
# bounding-box fill 0.43-0.47, texture 1.06-3.09) vs smooth non-brain phantoms (texture ~0.15, cylinder fill 0.78).
BRAIN_EXTENT_MM = (90.0, 210.0)
BRAIN_FILL = (0.36, 0.52)
MIN_TEXTURE = 0.5


def assess(volumes: dict, spacing_mm) -> dict:
    """volumes: {sequence name: 3D array}. Returns {"refusals": [...], "warnings": [...], "stats": {...}}."""
    spacing = [float(s) for s in spacing_mm[:3]]
    voxel_ml = float(np.prod(spacing)) / 1000.0
    masks = {k: np.asarray(v) != 0 for k, v in volumes.items()}
    union = np.logical_or.reduce(list(masks.values()))
    nonzero_ml = float(union.sum()) * voxel_ml
    zero_fraction = 1.0 - float(union.mean())

    agreement = {}
    for k, m in masks.items():
        s = m.sum() + union.sum()
        agreement[k] = float(2.0 * (m & union).sum() / s) if s else 1.0

    extent_mm = [0.0, 0.0, 0.0]
    if union.any():
        idx = np.nonzero(union)
        extent_mm = [float((i.max() - i.min() + 1) * sp) for i, sp in zip(idx, spacing)]

    refusals, warnings = [], []
    if zero_fraction < REFUSE_MIN_ZERO_FRACTION:
        refusals.append(
            f"Only {zero_fraction:.1%} of voxels are background (zero). Skull-stripped brain MRI has a zero "
            "background; this input does not appear to be skull-stripped."
        )
    if nonzero_ml > REFUSE_NONZERO_ML:
        refusals.append(
            f"Non-zero tissue volume is {nonzero_ml:.0f} mL, far above a skull-stripped brain (~1000-1800 mL). "
            "The input appears to include skull, scalp or other non-brain tissue."
        )
    elif nonzero_ml > WARN_NONZERO_ML_HIGH:
        warnings.append(
            f"Non-zero tissue volume is {nonzero_ml:.0f} mL, larger than typical for a skull-stripped brain; "
            "check that skull stripping was applied."
        )
    if nonzero_ml < WARN_NONZERO_ML_LOW:
        warnings.append(
            f"Non-zero tissue volume is only {nonzero_ml:.0f} mL; the input may be cropped, partial or not a "
            "whole-brain scan."
        )
    for k, a in agreement.items():
        if a < WARN_MASK_AGREEMENT:
            warnings.append(
                f"The {k} brain region overlaps the other sequences poorly (Dice {a:.2f}); sequences may not be "
                "co-registered or skull-stripped consistently."
            )
    if max(spacing) > WARN_MAX_SPACING_MM:
        warnings.append(
            f"Voxel spacing {tuple(round(s, 2) for s in spacing)} mm is coarser than the 1 mm data the model was "
            "trained on; the image is resampled and predictions may be less reliable."
        )
    for k, v in volumes.items():
        if float(np.min(v)) < 0:
            warnings.append(f"{k} contains negative intensities; this is unusual for BraTS-style MRI.")

    # Out-of-distribution: does the non-zero region look like a skull-stripped brain?
    fill, texture = brain_shape_stats(volumes, union, spacing)
    if not refusals and union.any():
        odd = [f"{n} {e:.0f} mm" for n, e in zip(("left-right", "front-back", "top-bottom"), extent_mm)
               if not BRAIN_EXTENT_MM[0] <= e <= BRAIN_EXTENT_MM[1]]
        if odd:
            warnings.append(f"Unusual object size ({', '.join(odd)}; adult brains are about 130-180 mm). "
                            "This may not be a whole-brain scan.")
        if not BRAIN_FILL[0] <= fill <= BRAIN_FILL[1]:
            warnings.append(f"The object's shape is unlike a brain (fills {fill:.0%} of its bounding box; brains fill "
                            "about 43-47%). This may not be a brain MRI.")
        if texture < MIN_TEXTURE:
            warnings.append(f"The image interior is unusually smooth (texture score {texture:.2f}; brain MRI scores "
                            "above 1). This may be synthetic or not a brain MRI.")

    return {
        "refusals": refusals,
        "warnings": warnings,
        "stats": {
            "nonzero_volume_mL": round(nonzero_ml, 1),
            "background_fraction": round(zero_fraction, 4),
            "brain_extent_mm": [round(e, 1) for e in extent_mm],
            "mask_agreement_dice": {k: round(v, 3) for k, v in agreement.items()},
            "bounding_box_fill": round(fill, 3),
            "texture_score": round(texture, 3),
        },
    }


def brain_shape_stats(volumes: dict, union: np.ndarray, spacing) -> tuple:
    """Bounding-box fill of the non-zero region and a texture score (mean gradient / mean intensity in its
    interior, minimum over sequences). Computed on a 2x-downsampled grid for speed."""
    if not union.any():
        return 0.0, 0.0
    idx = np.argwhere(union)
    fill = float(union.sum() / np.prod(idx.max(0) - idx.min(0) + 1))
    m = union[::2, ::2, ::2]
    core = binary_erosion(m, iterations=2)
    if core.sum() < 50:
        return fill, 0.0
    scores = []
    for v in volumes.values():
        s = np.asarray(v, dtype=np.float32)[::2, ::2, ::2]
        grad = np.sqrt(sum(sobel(s, axis=a) ** 2 for a in range(3)))
        mean = float(np.abs(s[core]).mean())
        scores.append(float(grad[core].mean()) / mean if mean > 0 else 0.0)
    return fill, min(scores)
