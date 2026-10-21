"""
Custom domain exceptions for NIfTI upload and validation module.
Every exception provides structured error codes, user-friendly messages,
and diagnostic details.
"""

from typing import Any, Dict, Optional


class NIfTIValidationError(Exception):
    """Base exception for all NIfTI validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        status_code: int = 400,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.status_code = status_code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": "error",
            "valid": False,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "ready_for_segmentation": False,
        }


class InvalidExtensionError(NIfTIValidationError):
    def __init__(self, filename: str, received_ext: str, modality: Optional[str] = None):
        msg = (
            f"The uploaded file '{filename}' has an unsupported extension '{received_ext}'. "
            f"Only valid NIfTI files with '.nii' or '.nii.gz' extensions are accepted. "
            f"Please convert or rename your scan to a valid NIfTI format."
        )
        super().__init__(
            message=msg,
            error_code="INVALID_FILE_EXTENSION",
            details={
                "filename": filename,
                "received_extension": received_ext,
                "allowed_extensions": [".nii", ".nii.gz"],
                "modality": modality,
            },
        )


class FileTooLargeError(NIfTIValidationError):
    def __init__(self, filename: str, max_bytes: int, received_bytes: Optional[int] = None, modality: Optional[str] = None):
        max_gib = max_bytes / (1024**3)
        msg = (
            f"The uploaded file '{filename}' exceeds the maximum allowed file size of {max_gib:.1f} GiB "
            f"({max_bytes:,} bytes). Upload processing was stopped immediately to protect server resources."
        )
        super().__init__(
            message=msg,
            error_code="FILE_TOO_LARGE",
            details={
                "filename": filename,
                "max_size_bytes": max_bytes,
                "received_bytes": received_bytes,
                "modality": modality,
            },
            status_code=413,
        )


class InvalidNIfTIFormatError(NIfTIValidationError):
    def __init__(self, filename: str, reason: str, modality: Optional[str] = None):
        mod_prefix = f"for modality {modality} " if modality else ""
        msg = (
            f"The uploaded file '{filename}' {mod_prefix}could not be read as a valid NIfTI MRI volume "
            f"({reason}). Please check that the file is not corrupted or truncated and upload it again."
        )
        super().__init__(
            message=msg,
            error_code="INVALID_NIFTI_FORMAT",
            details={"filename": filename, "reason": reason, "modality": modality},
        )


class InvalidDimensionsError(NIfTIValidationError):
    def __init__(self, filename: str, shape: tuple, modality: Optional[str] = None):
        ndim = len(shape)
        msg = (
            f"The uploaded MRI sequence '{modality or filename}' has {ndim} dimensions {shape}. "
            f"Voxels requires exactly 3D MRI volumes (e.g., 240 × 240 × 155). "
            f"2D slices and 4D time-series/multi-channel volumes cannot be processed directly."
        )
        super().__init__(
            message=msg,
            error_code="INVALID_DIMENSIONS",
            details={
                "filename": filename,
                "received_dimensions": ndim,
                "received_shape": list(shape),
                "expected_dimensions": 3,
                "modality": modality,
            },
        )


class NonFiniteValuesError(NIfTIValidationError):
    def __init__(self, filename: str, modality: Optional[str] = None, non_finite_count: Optional[int] = None):
        msg = (
            f"The uploaded {modality or 'MRI'} volume in '{filename}' contains invalid non-finite intensity values "
            f"(NaN, +Infinity, or -Infinity). Valid MRI scans must have finite floating-point or integer intensities."
        )
        super().__init__(
            message=msg,
            error_code="NON_FINITE_VALUES",
            details={
                "filename": filename,
                "modality": modality,
                "non_finite_count": non_finite_count,
            },
        )


class ConstantImageError(NIfTIValidationError):
    def __init__(self, filename: str, constant_value: float, modality: Optional[str] = None):
        msg = (
            f"The uploaded {modality or 'MRI'} volume in '{filename}' has a constant intensity of {constant_value} "
            f"across all voxels. A valid brain MRI must contain meaningful anatomical intensity variation."
        )
        super().__init__(
            message=msg,
            error_code="CONSTANT_IMAGE",
            details={
                "filename": filename,
                "modality": modality,
                "constant_value": float(constant_value),
            },
        )


class ShapeMismatchError(NIfTIValidationError):
    def __init__(
        self,
        mismatched_modality: str,
        received_shape: tuple,
        reference_modality: str,
        expected_shape: tuple,
    ):
        rx, ry, rz = received_shape[:3]
        ex, ey, ez = expected_shape[:3]
        msg = (
            f"{mismatched_modality} has dimensions {rx} × {ry} × {rz}, but the reference {reference_modality} "
            f"sequence has dimensions {ex} × {ey} × {ez}. All four MRI sequences must share identical dimensions. "
            f"Please co-register and resample your scans before uploading."
        )
        super().__init__(
            message=msg,
            error_code="SHAPE_MISMATCH",
            details={
                "modality": mismatched_modality,
                "received_shape": list(received_shape),
                "reference_modality": reference_modality,
                "expected_shape": list(expected_shape),
            },
        )


class AffineMismatchError(NIfTIValidationError):
    def __init__(
        self,
        mismatched_modality: str,
        reference_modality: str,
        max_diff: float,
        tolerance: float,
    ):
        msg = (
            f"The spatial transformation (affine matrix) for {mismatched_modality} does not match "
            f"the reference {reference_modality} sequence (maximum coordinate discrepancy: {max_diff:.4f} mm, "
            f"tolerance: {tolerance:.4f} mm). Scans must be in the same stereotactic coordinate space."
        )
        super().__init__(
            message=msg,
            error_code="AFFINE_MISMATCH",
            details={
                "modality": mismatched_modality,
                "reference_modality": reference_modality,
                "max_difference": float(max_diff),
                "tolerance": float(tolerance),
            },
        )


class InvalidVoxelSpacingError(NIfTIValidationError):
    def __init__(self, filename: str, zooms: tuple, reason: str, modality: Optional[str] = None):
        msg = (
            f"The voxel spacing {zooms} for {modality or filename} is invalid: {reason}. "
            f"Voxel dimensions must be strictly positive and finite numbers."
        )
        super().__init__(
            message=msg,
            error_code="INVALID_VOXEL_SPACING",
            details={
                "filename": filename,
                "zooms": list(zooms),
                "reason": reason,
                "modality": modality,
            },
        )


class SingularAffineError(NIfTIValidationError):
    def __init__(self, filename: str, determinant: float, modality: Optional[str] = None):
        msg = (
            f"The spatial transformation affine matrix for {modality or filename} is singular or degenerate "
            f"(determinant = {determinant:.6e}). The volume cannot be properly mapped to physical 3D space."
        )
        super().__init__(
            message=msg,
            error_code="SINGULAR_AFFINE",
            details={
                "filename": filename,
                "determinant": float(determinant),
                "modality": modality,
            },
        )


class MissingModalityError(NIfTIValidationError):
    def __init__(self, missing_modalities: list):
        missing_str = ", ".join(missing_modalities)
        msg = (
            f"Required MRI sequence missing: {missing_str}. "
            f"Please upload all four required MRI sequences (T1, T1Gd, T2, FLAIR) to complete the study."
        )
        super().__init__(
            message=msg,
            error_code="MISSING_MODALITY",
            details={
                "missing_modalities": missing_modalities,
                "required_modalities": ["T1", "T1Gd", "T2", "FLAIR"],
            },
        )


class DuplicateModalityError(NIfTIValidationError):
    def __init__(self, modality: str, count: int):
        msg = (
            f"Multiple files ({count}) are assigned to modality '{modality}'. "
            f"Each modality (T1, T1Gd, T2, FLAIR) must be provided exactly once."
        )
        super().__init__(
            message=msg,
            error_code="DUPLICATE_MODALITY",
            details={"modality": modality, "count": count},
        )


class StudyNotFoundError(NIfTIValidationError):
    def __init__(self, study_id: str):
        msg = f"MRI Study ID '{study_id}' was not found. Please upload and validate your MRI study first."
        super().__init__(
            message=msg,
            error_code="STUDY_NOT_FOUND",
            details={"study_id": study_id},
            status_code=404,
        )


class StudyExpiredError(NIfTIValidationError):
    def __init__(self, study_id: str):
        msg = (
            f"MRI Study '{study_id}' has expired. Validated studies are retained for 60 minutes for security. "
            f"Please re-upload and validate your study to run segmentation."
        )
        super().__init__(
            message=msg,
            error_code="STUDY_EXPIRED",
            details={"study_id": study_id},
            status_code=410,
        )
