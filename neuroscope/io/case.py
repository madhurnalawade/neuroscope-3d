"""Cross-file checks and the canonical case object shared by the upload path and the dev adapter.

Arrays are keyed by sequence name (FLAIR, T1, T1Gd, T2). No model channel order is defined here: the
inference code must build its input order from the model's own configuration.
Nothing is reordered, resampled, flipped or realigned.
"""
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from . import errors as E
from .sequences import SEQUENCES, assign_sequences
from .validate import MAX_DATA_BYTES, MAX_FILE_BYTES, FileCheck, validate_file

AFFINE_ATOL_MM = 1e-3     # affines must agree to 1 micrometre (absolute, per element)
SPACING_ATOL_MM = 1e-4


@dataclass
class CanonicalCase:
    arrays: Dict[str, np.ndarray] = field(repr=False)    # sequence name -> 3D array, as stored
    affine: np.ndarray
    shape: Tuple[int, int, int]
    spacing: Tuple[float, float, float]
    orientation: Tuple[str, str, str]                     # nibabel.aff2axcodes
    file_names: Dict[str, str]                            # sequence -> sanitised short name
    sha256: Dict[str, str]                                # sequence -> SHA-256 of the input file
    seg: Optional[np.ndarray] = field(default=None, repr=False)   # expert mask, BraTS 2021 codes 0/1/2/4
    seg_file: Optional[str] = None
    seg_sha256: Optional[str] = None
    source: str = "upload"                                # 'upload' or 'msd-dev'
    notes: List[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Printable summary: geometry, hashes and intensity ranges. Contains no voxel arrays."""
        out = {
            "source": self.source,
            "shape": list(self.shape),
            "spacing_mm": [round(s, 6) for s in self.spacing],
            "orientation": "".join(self.orientation),
            "affine": [[round(float(v), 4) for v in row] for row in self.affine],
            "sequences": {},
        }
        for seq in SEQUENCES:
            a = self.arrays[seq]
            nz = a[a != 0]
            out["sequences"][seq] = {
                "file": self.file_names[seq],
                "sha256": self.sha256[seq],
                "dtype": str(a.dtype),
                "min": round(float(a.min()), 1),
                "max": round(float(a.max()), 1),
                "p1_p99_nonzero": [round(float(v), 1) for v in np.percentile(nz, [1, 99])] if nz.size else None,
            }
        if self.seg is not None:
            codes, counts = np.unique(self.seg, return_counts=True)
            out["expert_mask"] = {"file": self.seg_file, "sha256": self.seg_sha256,
                                  "voxels_per_code": {int(c): int(n) for c, n in zip(codes, counts)}}
        if self.notes:
            out["notes"] = list(self.notes)
        return out


@dataclass
class LoadResult:
    ok: bool
    case: Optional[CanonicalCase] = None
    issues: List[E.Issue] = field(default_factory=list)
    checks: Dict[str, FileCheck] = field(default_factory=dict)


def compare_geometry(checks: Dict[str, FileCheck], reference: str,
                     affine_atol: float = AFFINE_ATOL_MM, spacing_atol: float = SPACING_ATOL_MM) -> List[E.Issue]:
    """Compare every file with ``reference``; report exactly which file differs and how."""
    ref = checks[reference]
    issues = []
    for key, c in checks.items():
        if key == reference:
            continue
        who = f"{key} '{c.name}'"
        than = f"{reference} '{ref.name}'"
        if c.shape != ref.shape:
            issues.append(E.issue(E.SHAPE_MISMATCH, f"{who} has shape {c.shape} but {than} has {ref.shape}.",
                                  file=c.name, shape=list(c.shape), reference_shape=list(ref.shape)))
        if not np.allclose(c.spacing, ref.spacing, rtol=0, atol=spacing_atol):
            issues.append(E.issue(E.SPACING_MISMATCH, f"{who} has voxel spacing {_mm(c.spacing)} mm but {than} has "
                                  f"{_mm(ref.spacing)} mm.", file=c.name))
        diff = np.abs(c.affine - ref.affine)
        if diff.max() > affine_atol:
            parts = []
            if diff[:3, :3].max() > affine_atol:
                parts.append(f"rotation/scaling differs by up to {diff[:3, :3].max():.4g}")
            if diff[:3, 3].max() > affine_atol:
                parts.append(f"origin differs by up to {diff[:3, 3].max():.4g} mm")
            if c.orientation != ref.orientation:
                parts.append(f"orientation {''.join(c.orientation)} vs {''.join(ref.orientation)}")
            issues.append(E.issue(E.AFFINE_MISMATCH, f"{who} is not on the same voxel grid as {than}: "
                                  + "; ".join(parts) + f" (tolerance {affine_atol} mm). Equal shapes do not prove "
                                  "alignment; the files were not realigned.", file=c.name,
                                  max_abs_diff=float(diff.max())))
    return issues


def _mm(sp):
    return "x".join(f"{s:g}" for s in sp)


def load_four_files(explicit: Optional[Dict[str, object]] = None, unlabelled: Iterable[object] = (),
                    seg=None, *, max_bytes: int = MAX_FILE_BYTES, max_data_bytes: int = MAX_DATA_BYTES) -> LoadResult:
    """Assign, validate and cross-check four 3D sequences (+ optional BraTS-coded expert mask).
    Returns a LoadResult; ``case`` is set only when every check passed."""
    assignment = assign_sequences(explicit, unlabelled)
    if not assignment.ok:
        return LoadResult(ok=False, issues=assignment.issues)

    checks = {seq: validate_file(assignment.files[seq], kind="image", max_bytes=max_bytes,
                                 max_data_bytes=max_data_bytes) for seq in SEQUENCES}
    issues = [c.issue for c in checks.values() if not c.ok]
    seg_check = None
    if seg is not None:
        seg_check = validate_file(seg, kind="label", max_bytes=max_bytes, max_data_bytes=max_data_bytes)
        if not seg_check.ok:
            issues.append(seg_check.issue)
    if issues:
        return LoadResult(ok=False, issues=issues, checks=checks)

    geometry = dict(checks)
    if seg_check is not None:
        geometry["expert mask"] = seg_check
    issues = compare_geometry(geometry, reference=SEQUENCES[0])
    if issues:
        return LoadResult(ok=False, issues=issues, checks=checks)

    ref = checks[SEQUENCES[0]]
    case = CanonicalCase(
        arrays={s: checks[s].data for s in SEQUENCES},
        affine=ref.affine.copy(),
        shape=ref.shape,
        spacing=ref.spacing,
        orientation=ref.orientation,
        file_names={s: checks[s].name for s in SEQUENCES},
        sha256={s: checks[s].sha256 for s in SEQUENCES},
        seg=seg_check.data if seg_check else None,
        seg_file=seg_check.name if seg_check else None,
        seg_sha256=seg_check.sha256 if seg_check else None,
    )
    return LoadResult(ok=True, case=case, checks=checks)
