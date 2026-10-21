"""Export one Decathlon case as four 3D sequence files + a BraTS-coded seg file, then verify by reloading
through the production four-file loader that shape, affine, spacing and every voxel value match exactly.

Same affine and spacing as the source, header reduced to 3D, no rotation/flip/resample (no np.rot90).
"""
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np

from ..io.case import CanonicalCase, load_four_files
from ..io.sequences import SEQUENCES
from .msd import DatasetError, DatasetInfo

FILE_TAG = {"FLAIR": "flair", "T1": "t1", "T1Gd": "t1gd", "T2": "t2"}


def _header_3d(src: nib.Nifti1Image, arr: np.ndarray) -> nib.Nifti1Header:
    hdr = src.header.copy()
    hdr.set_data_shape(arr.shape)
    hdr.set_data_dtype(arr.dtype)
    hdr.set_slope_inter(None, None)          # values are written exactly as loaded, no rescaling
    zooms = src.header.get_zooms()
    hdr.set_zooms(tuple(float(z) for z in zooms[:3]))
    return hdr


def check_output_dir(out_dir: Path, dataset_root: Path) -> Path:
    out = Path(out_dir).expanduser().resolve()
    ds = Path(dataset_root).expanduser().resolve()
    if out == ds or ds in out.parents:
        raise DatasetError("The output directory must not be inside the dataset root.")
    return out


def export_case(info: DatasetInfo, case: CanonicalCase, case_id: str, out_dir, overwrite: bool = False) -> Dict[str, Path]:
    """Write <case>_{flair,t1,t1gd,t2}.nii.gz and <case>_seg.nii.gz (if the case has a mask)."""
    out = check_output_dir(out_dir, info.root)
    out.mkdir(parents=True, exist_ok=True)
    src = nib.load(str(info.root / info.find(case_id).image), mmap=False)   # header only, for qform/sform codes
    paths = {seq: out / f"{case_id}_{FILE_TAG[seq]}.nii.gz" for seq in SEQUENCES}
    if case.seg is not None:
        paths["seg"] = out / f"{case_id}_seg.nii.gz"
    clash = [p.name for p in paths.values() if p.exists()]
    if clash and not overwrite:
        raise DatasetError(f"Refusing to overwrite existing files: {', '.join(clash)} (use --overwrite).")
    for key, p in paths.items():
        arr = case.seg if key == "seg" else case.arrays[key]
        # The copied header keeps the source qform/sform fields and codes verbatim; passing its own best
        # affine leaves them untouched, so the saved affine is bit-identical to the source.
        hdr = _header_3d(src, arr)
        img = nib.Nifti1Image(arr, hdr.get_best_affine(), hdr)
        if not np.array_equal(img.affine, case.affine):
            raise DatasetError("Internal check failed: exported header affine differs from the source affine.")
        nib.save(img, str(p))
    return paths


def verify_export(case: CanonicalCase, paths: Dict[str, Path]) -> List[str]:
    """Reload the exported files through the production loader; return a list of mismatches (empty = exact)."""
    res = load_four_files(explicit={s: paths[s] for s in SEQUENCES}, seg=paths.get("seg"))
    if not res.ok:
        return [str(i) for i in res.issues]
    got = res.case
    problems = []
    if got.shape != case.shape:
        problems.append(f"shape {got.shape} != {case.shape}")
    if not np.array_equal(got.affine, case.affine):
        problems.append(f"affine differs (max {np.abs(got.affine - case.affine).max():.3g})")
    if not np.allclose(got.spacing, case.spacing, rtol=0, atol=1e-6):
        problems.append(f"spacing {got.spacing} != {case.spacing}")
    for s in SEQUENCES:
        if got.arrays[s].dtype != case.arrays[s].dtype or not np.array_equal(got.arrays[s], case.arrays[s]):
            problems.append(f"{s} voxel values or dtype differ")
    if case.seg is not None and not np.array_equal(got.seg, case.seg):
        problems.append("seg voxel values differ")
    return problems
