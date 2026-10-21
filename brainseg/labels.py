"""Label-code conversions and region/volume helpers."""
import numpy as np

from .constants import BACKGROUND, ED, ET, MSD_TO_BRATS, NCR, REGIONS


def msd_to_brats(label_map: np.ndarray) -> np.ndarray:
    """Convert an MSD Task01 label map (1=edema, 2=non-enhancing, 3=enhancing) to BraTS codes (1/2/4)."""
    values = set(np.unique(label_map).tolist())
    unknown = values - set(MSD_TO_BRATS)
    if unknown:
        raise ValueError(f"Unexpected MSD label values {sorted(unknown)}; expected 0-3")
    lut = np.zeros(max(MSD_TO_BRATS) + 1, dtype=np.uint8)
    for src, dst in MSD_TO_BRATS.items():
        lut[src] = dst
    return lut[label_map.astype(np.intp)]


def channels_to_brats(tc: np.ndarray, wt: np.ndarray, et: np.ndarray) -> np.ndarray:
    """Turn binary TC/WT/ET model channels into one BraTS label map.

    Same precedence as the MONAI bundle: ET -> 4, else TC -> 1 (NCR), else WT -> 2 (ED).
    """
    out = np.full(tc.shape, BACKGROUND, dtype=np.uint8)
    out[wt] = ED
    out[tc] = NCR
    out[et] = ET
    return out


def region_masks(label_map: np.ndarray) -> dict:
    """Derive nested WT/TC/ET boolean masks from a BraTS label map."""
    return {name: np.isin(label_map, codes) for name, codes in REGIONS.items()}


def region_volumes(label_map: np.ndarray, spacing_mm) -> dict:
    """Estimated volume per label and region: voxel count x voxel volume (mm^3); mL = mm^3 / 1000."""
    voxel_mm3 = float(np.prod([float(s) for s in spacing_mm[:3]]))
    masks = {"NCR": label_map == NCR, "ED": label_map == ED, **region_masks(label_map)}
    out = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        mm3 = count * voxel_mm3
        out[name] = {"voxels": count, "voxel_volume_mm3": voxel_mm3, "mm3": mm3, "mL": mm3 / 1000.0}
    return out
