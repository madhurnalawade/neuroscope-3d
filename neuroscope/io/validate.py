"""Per-file validation in a safe order; returns a structured result and never raises for bad input.

Order: exists -> size on disk (before reading anything) -> extension -> gzip magic and full gzip stream
(constant memory, capped decompressed size) -> NIfTI header (lazy) -> dimensionality -> header-promised
data size -> spacing -> affine -> voxel data -> finite values -> constant volume (images) or label codes (masks).
"""
import gzip
import hashlib
import math
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np

from ..security.hygiene import describe_exception, safe_name
from . import errors as E

MAX_FILE_BYTES = 1_000_000_000          # "over 1 GB" is rejected; checked on disk before loading
MAX_DATA_BYTES = 4 * 1024 ** 3          # cap on uncompressed voxel data (guards against gzip bombs)
GZIP_SLACK_BYTES = 64 * 1024 * 1024    # header + extensions allowance on top of MAX_DATA_BYTES
BRATS_CODES = frozenset({0, 1, 2, 4})   # expert mask codes (BraTS 2021): 1 NCR, 2 ED, 4 ET
_CHUNK = 1 << 20


@dataclass
class FileCheck:
    ok: bool
    name: str                               # sanitised short name
    issue: Optional[E.Issue] = None
    size_bytes: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None
    spacing: Optional[Tuple[float, float, float]] = None
    affine: Optional[np.ndarray] = None
    orientation: Optional[Tuple[str, str, str]] = None
    dtype: Optional[str] = None
    sha256: Optional[str] = None
    data: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def code(self) -> Optional[str]:
        return self.issue.code if self.issue else None

    @property
    def message(self) -> str:
        return self.issue.message if self.issue else f"'{self.name}' is valid."


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _fail(short, code, msg, **kw) -> FileCheck:
    return FileCheck(ok=False, name=short, issue=E.issue(code, msg, file=short), **kw)


def _gzip_problem(path: Path, limit: int) -> Optional[Tuple[str, str]]:
    """Decompress the whole stream in chunks (constant memory). Returns (code, reason) or None."""
    total = 0
    try:
        with gzip.open(path, "rb") as g:
            while True:
                chunk = g.read(_CHUNK)
                if not chunk:
                    return None
                total += len(chunk)
                if total > limit:
                    return E.DATA_TOO_LARGE, f"decompresses to more than {limit / 1e9:.1f} GB"
    except EOFError:
        return E.CORRUPT_GZIP, "the gzip stream ends early (truncated file)"
    except (gzip.BadGzipFile, zlib.error, OSError) as exc:
        return E.CORRUPT_GZIP, f"the gzip stream is damaged ({describe_exception(exc)})"


def validate_file(path, *, kind: str = "image", max_bytes: int = MAX_FILE_BYTES,
                  max_data_bytes: int = MAX_DATA_BYTES, keep_data: bool = True) -> FileCheck:
    """Validate one 3D NIfTI file. ``kind`` is 'image' (sequence) or 'label' (expert mask, BraTS codes)."""
    p = Path(path)
    short = safe_name(p.name)
    try:
        if not p.exists():
            return _fail(short, E.FILE_NOT_FOUND, f"'{short}' was not found.")
        if not p.is_file():
            return _fail(short, E.NOT_A_FILE, f"'{short}' is not a regular file.")
        size = p.stat().st_size
        if size == 0:
            return _fail(short, E.EMPTY_FILE, f"'{short}' is empty (0 bytes).", size_bytes=0)
        if size > max_bytes:
            return _fail(short, E.FILE_TOO_LARGE,
                         f"'{short}' is {size / 1e9:.2f} GB; the limit is {max_bytes / 1e9:.2f} GB. It was not loaded.",
                         size_bytes=size)
        low = p.name.lower()
        if not (low.endswith(".nii") or low.endswith(".nii.gz")):
            return _fail(short, E.BAD_EXTENSION, f"'{short}' must be a NIfTI file ending in .nii or .nii.gz.",
                         size_bytes=size)
        with open(p, "rb") as f:
            magic = f.read(2)
        if low.endswith(".gz"):
            if magic != b"\x1f\x8b":
                return _fail(short, E.NOT_GZIP,
                             f"'{short}' is named .nii.gz but is not gzip-compressed, so it is not a NIfTI-GZ file.",
                             size_bytes=size)
            problem = _gzip_problem(p, max_data_bytes + GZIP_SLACK_BYTES)
            if problem:
                return _fail(short, problem[0], f"'{short}' cannot be used: {problem[1]}.", size_bytes=size)
        elif magic == b"\x1f\x8b":
            return _fail(short, E.NOT_NIFTI, f"'{short}' is gzip-compressed but named .nii; rename it to .nii.gz.",
                         size_bytes=size)

        try:
            img = nib.load(str(p), mmap=False)   # header only; mmap off so no file handle stays open
        except Exception as exc:  # noqa: BLE001 - any parser failure means "not NIfTI"
            return _fail(short, E.NOT_NIFTI, f"'{short}' is not a readable NIfTI file ({describe_exception(exc)}).",
                         size_bytes=size)
        if not isinstance(img, (nib.Nifti1Image, nib.Nifti2Image)):
            return _fail(short, E.NOT_NIFTI, f"'{short}' is not a NIfTI-1/NIfTI-2 image.", size_bytes=size)

        shape = tuple(int(s) for s in img.shape)
        if len(shape) != 3:
            hint = " Pass each sequence as its own 3D file." if len(shape) == 4 else ""
            return _fail(short, E.NOT_3D, f"'{short}' has {len(shape)} dimensions {shape}; a 3D volume is required.{hint}",
                         size_bytes=size, shape=shape)
        if min(shape) < 2:
            return _fail(short, E.NOT_3D, f"'{short}' has shape {shape}: a 2D slice, not a 3D volume.",
                         size_bytes=size, shape=shape)

        dtype = img.header.get_data_dtype()
        if math.prod(shape) * max(dtype.itemsize, 1) > max_data_bytes:
            return _fail(short, E.DATA_TOO_LARGE, f"'{short}' declares more voxel data than the "
                         f"{max_data_bytes / 1e9:.1f} GB limit.", size_bytes=size, shape=shape)

        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        if not all(math.isfinite(z) and z > 0 for z in zooms):
            return _fail(short, E.BAD_SPACING, f"'{short}' has invalid voxel spacing {zooms} mm.",
                         size_bytes=size, shape=shape)
        affine = np.asarray(img.affine, dtype=np.float64)
        if affine.shape != (4, 4) or not np.all(np.isfinite(affine)) or abs(np.linalg.det(affine[:3, :3])) < 1e-9:
            return _fail(short, E.BAD_AFFINE, f"'{short}' has a missing, non-finite or singular affine.",
                         size_bytes=size, shape=shape)

        try:
            data = np.asarray(img.dataobj)
        except MemoryError:
            return _fail(short, E.DATA_TOO_LARGE, f"'{short}' is too large to load into memory.", size_bytes=size)
        except Exception as exc:  # noqa: BLE001
            return _fail(short, E.UNREADABLE, f"'{short}': voxel data could not be read; the file is truncated or "
                         f"damaged ({describe_exception(exc)}).", size_bytes=size, shape=shape)
        if data.ndim != 3:
            return _fail(short, E.NOT_3D, f"'{short}' voxel data is not 3D.", size_bytes=size, shape=shape)

        if data.dtype.kind in "fc":
            bad = int(np.count_nonzero(~np.isfinite(data)))
            if bad:
                return _fail(short, E.NON_FINITE, f"'{short}' contains {bad} NaN or infinite voxel value(s).",
                             size_bytes=size, shape=shape)

        if kind == "label":
            if data.dtype.kind in "fc" and not np.array_equal(data, np.round(data)):
                return _fail(short, E.BAD_LABELS, f"'{short}' has non-integer values; an expert mask must hold "
                             "BraTS 2021 codes 0, 1, 2, 4.", size_bytes=size, shape=shape)
            values = sorted(int(v) for v in np.unique(data))
            extra = [v for v in values if v not in BRATS_CODES]
            if extra:
                hint = (" This looks like Decathlon coding (1 oedema, 2 non-enhancing, 3 enhancing); convert it to "
                        "BraTS 2021 codes first.") if 3 in extra else ""
                return _fail(short, E.BAD_LABELS, f"'{short}' contains label value(s) {extra}; allowed BraTS 2021 "
                             f"codes are 0, 1 (NCR), 2 (ED), 4 (ET).{hint}", size_bytes=size, shape=shape)
            data = data.astype(np.uint8, copy=False)
        elif data.min() == data.max():
            return _fail(short, E.CONSTANT, f"'{short}' is constant (every voxel has the same value, e.g. all "
                         "zeros), so it holds no image.", size_bytes=size, shape=shape)

        return FileCheck(ok=True, name=short, size_bytes=size, shape=shape, spacing=zooms, affine=affine,
                         orientation=tuple(nib.aff2axcodes(affine)), dtype=str(data.dtype),
                         sha256=sha256_file(p), data=data if keep_data else None)
    except Exception as exc:  # noqa: BLE001 - last resort: never crash on user input
        return _fail(short, E.UNREADABLE, f"'{short}' could not be checked ({describe_exception(exc)}).")
