"""
Production-quality NIfTI validation engine for NeuroScope 3D.
Enforces all 11 medical imaging validation rules with official nibabel integration.
"""

from dataclasses import dataclass
import gzip
import math
import os
from pathlib import Path
import struct
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError

from app.core.exceptions import (
    AffineMismatchError,
    ConstantImageError,
    FileTooLargeError,
    InvalidDimensionsError,
    InvalidExtensionError,
    InvalidNIfTIFormatError,
    InvalidVoxelSpacingError,
    MissingModalityError,
    NonFiniteValuesError,
    ShapeMismatchError,
    SingularAffineError,
)
from app.services.modality_detector import VALID_MODALITIES

MAX_FILE_SIZE_BYTES = 1024 * 1024 * 1024  # 1 GiB


@dataclass
class SingleSequenceInfo:
    """Dimensional and spatial metadata for a validated NIfTI sequence."""
    modality: str
    filename: str
    saved_path: Path
    file_size_bytes: int
    shape: Tuple[int, ...]
    zooms: Tuple[float, ...]
    orientation: Tuple[str, ...]
    affine: np.ndarray


class NIfTIValidator:
    """
    Validates single and multi-sequence brain MRI studies.
    Strictly uses official nibabel API and numpy for numerical fidelity.
    """

    def __init__(self, max_file_size_bytes: int = MAX_FILE_SIZE_BYTES):
        self.max_file_size_bytes = max_file_size_bytes

    def save_upload_stream(
        self,
        file_stream: Any,
        dest_path: Path,
        filename: str,
        modality: Optional[str] = None,
    ) -> int:
        """
        Stream uploaded file to destination path on disk.
        Enforces 1 GiB limit and valid file extensions (.nii, .nii.gz).
        Deletes partial files immediately if limit is exceeded.
        """
        lower_name = (filename or "").lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            if "." in filename:
                ext = "." + filename.rsplit(".", 1)[1]
            else:
                ext = ""
            raise InvalidExtensionError(filename=filename, received_ext=ext, modality=modality)

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        total_bytes = 0
        chunk_size = 64 * 1024  # 64 KB

        try:
            with open(dest_path, "wb") as f_out:
                while True:
                    chunk = file_stream.read(chunk_size)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > self.max_file_size_bytes:
                        f_out.close()
                        if dest_path.exists():
                            try:
                                dest_path.unlink()
                            except Exception:
                                pass
                        raise FileTooLargeError(
                            filename=filename,
                            max_bytes=self.max_file_size_bytes,
                            received_bytes=total_bytes,
                            modality=modality,
                        )
                    f_out.write(chunk)
        except FileTooLargeError:
            raise
        except Exception as e:
            if dest_path.exists():
                try:
                    dest_path.unlink()
                except Exception:
                    pass
            raise e

        return total_bytes

    def validate_single_file(
        self,
        file_path: Path,
        original_filename: str,
        modality: Optional[str] = None,
        file_size_bytes: Optional[int] = None,
    ) -> SingleSequenceInfo:
        """
        Validates an individual NIfTI file for format integrity, dimensions,
        voxel intensity validity, spacing, and spatial transformations.
        """
        lower = original_filename.lower()
        if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
            ext = ("." + original_filename.rsplit(".", 1)[1]) if "." in original_filename else ""
            raise InvalidExtensionError(filename=original_filename, received_ext=ext, modality=modality)

        if file_size_bytes is None:
            if not file_path.exists():
                raise InvalidNIfTIFormatError(
                    filename=original_filename,
                    reason="Uploaded file was not found on disk",
                    modality=modality,
                )
            file_size_bytes = file_path.stat().st_size

        if file_size_bytes > self.max_file_size_bytes:
            raise FileTooLargeError(
                filename=original_filename,
                max_bytes=self.max_file_size_bytes,
                received_bytes=file_size_bytes,
                modality=modality,
            )

        # 1. Readability with nibabel
        try:
            img = nib.load(str(file_path), mmap=False)
        except (
            ImageFileError,
            HeaderDataError,
            gzip.BadGzipFile,
            zlib.error,
            EOFError,
            struct.error,
            OSError,
            ValueError,
        ) as e:
            raise InvalidNIfTIFormatError(
                filename=original_filename,
                reason=str(e),
                modality=modality,
            )
        except Exception as e:
            raise InvalidNIfTIFormatError(
                filename=original_filename,
                reason=f"Unexpected error loading NIfTI volume: {str(e)}",
                modality=modality,
            )

        # 2. Dimensionality: exactly 3D volume
        shape = tuple(img.shape)
        if len(shape) != 3:
            raise InvalidDimensionsError(
                filename=original_filename,
                shape=shape,
                modality=modality,
            )

        # 3. Read voxel data and verify payload integrity
        try:
            data = img.get_fdata()
        except Exception as e:
            raise InvalidNIfTIFormatError(
                filename=original_filename,
                reason=f"Failed decompressing or parsing voxel data array: {str(e)}",
                modality=modality,
            )

        if len(data.shape) != 3:
            raise InvalidDimensionsError(
                filename=original_filename,
                shape=data.shape,
                modality=modality,
            )

        # 4. Check for NaN or infinite voxel values
        if not np.all(np.isfinite(data)):
            non_finite_count = int(np.sum(~np.isfinite(data)))
            raise NonFiniteValuesError(
                filename=original_filename,
                modality=modality,
                non_finite_count=non_finite_count,
            )

        # 5. Check constant / all-zero image
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if data_min == data_max:
            raise ConstantImageError(
                filename=original_filename,
                constant_value=data_min,
                modality=modality,
            )

        # 6. Validate voxel spacing (zooms)
        try:
            zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        except Exception as e:
            raise InvalidVoxelSpacingError(
                filename=original_filename,
                zooms=(0.0, 0.0, 0.0),
                reason=f"Unable to read voxel spacing from header: {str(e)}",
                modality=modality,
            )

        for z in zooms:
            if not (math.isfinite(z) and z > 0):
                raise InvalidVoxelSpacingError(
                    filename=original_filename,
                    zooms=zooms,
                    reason=f"Voxel spacing must be strictly positive and finite (found {z})",
                    modality=modality,
                )

        # 7. Validate affine matrix
        affine = np.asarray(img.affine, dtype=np.float64)
        if affine.shape != (4, 4):
            raise SingularAffineError(
                filename=original_filename,
                determinant=0.0,
                modality=modality,
            )

        det = float(np.linalg.det(affine[:3, :3]))
        if not np.isfinite(det) or abs(det) < 1e-6:
            raise SingularAffineError(
                filename=original_filename,
                determinant=det,
                modality=modality,
            )

        # 8. Compute axis orientation
        try:
            orientation = tuple(nib.orientations.aff2axcodes(affine))
        except Exception:
            orientation = ("R", "A", "S")

        return SingleSequenceInfo(
            modality=modality or "",
            filename=original_filename,
            saved_path=file_path,
            file_size_bytes=file_size_bytes,
            shape=shape,
            zooms=zooms,
            orientation=orientation,
            affine=affine,
        )

    def validate_study(
        self,
        staged_files: Dict[str, Tuple[Path, str, int]],
    ) -> Tuple[bool, Dict[str, SingleSequenceInfo], Dict[str, Any], float]:
        """
        Performs full multi-sequence consistency validation across all 4 MRI modalities.
        Checks shape equality and 4x4 affine congruence.
        """
        start_time = time.time()

        missing = [m for m in VALID_MODALITIES if m not in staged_files]
        if missing:
            raise MissingModalityError(missing_modalities=missing)

        # Validate each sequence individually
        validated_seqs: Dict[str, SingleSequenceInfo] = {}
        for mod in VALID_MODALITIES:
            path, orig_name, bytes_len = staged_files[mod]
            info = self.validate_single_file(
                file_path=path,
                original_filename=orig_name,
                modality=mod,
                file_size_bytes=bytes_len,
            )
            validated_seqs[mod] = info

        # Cross-sequence consistency checks against reference modality (T1)
        ref_mod = "T1" if "T1" in validated_seqs else VALID_MODALITIES[0]
        ref_info = validated_seqs[ref_mod]
        ref_shape = ref_info.shape
        ref_affine = ref_info.affine

        for mod in VALID_MODALITIES:
            if mod == ref_mod:
                continue
            seq_info = validated_seqs[mod]

            # Shape congruence
            if seq_info.shape != ref_shape:
                raise ShapeMismatchError(
                    mismatched_modality=mod,
                    received_shape=seq_info.shape,
                    reference_modality=ref_mod,
                    expected_shape=ref_shape,
                )

            # 4x4 Affine congruence (rtol=1e-3, atol=1e-3)
            if not np.allclose(seq_info.affine, ref_affine, rtol=1e-3, atol=1e-3):
                max_diff = float(np.max(np.abs(seq_info.affine - ref_affine)))
                raise AffineMismatchError(
                    mismatched_modality=mod,
                    reference_modality=ref_mod,
                    max_diff=max_diff,
                    tolerance=1e-3,
                )

        duration = time.time() - start_time
        total_size_bytes = sum(i.file_size_bytes for i in validated_seqs.values())

        metadata: Dict[str, Any] = {
            "shape": list(ref_info.shape),
            "voxel_spacing_mm": [float(z) for z in ref_info.zooms[:3]],
            "orientation": list(ref_info.orientation),
            "affine_consistent": True,
            "all_finite": True,
            "all_non_constant": True,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "validation_duration_seconds": round(duration, 3),
        }

        return True, validated_seqs, metadata, duration
