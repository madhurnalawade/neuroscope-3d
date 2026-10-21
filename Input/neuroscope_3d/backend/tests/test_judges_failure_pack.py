"""
Mandatory Judges' Failure Pack Test Suite for NeuroScope 3D.
Covers all 8 required judge failure scenarios with synthetic NIfTI data.
"""

import gzip
import io
import shutil
import tempfile
from pathlib import Path
import unittest

import nibabel as nib
import numpy as np
import pytest

from app.core.exceptions import (
    ConstantImageError,
    FileTooLargeError,
    InvalidDimensionsError,
    InvalidNIfTIFormatError,
    MissingModalityError,
    NonFiniteValuesError,
    ShapeMismatchError,
)
from app.services.nifti_validator import NIfTIValidator


def make_synthetic_nifti(
    dest_path: Path,
    shape=(64, 64, 32),
    zooms=(1.0, 1.0, 1.0),
    affine=None,
    data=None,
    constant_val=None,
    has_nan=False,
    has_inf=False,
    is_2d=False,
) -> Path:
    """Helper to generate strictly valid or specifically faulted synthetic NIfTI files."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if is_2d:
        shape = (shape[0], shape[1])
        zooms = (zooms[0], zooms[1])

    if data is not None:
        arr = data
    elif constant_val is not None:
        arr = np.full(shape, constant_val, dtype=np.float32)
    else:
        # Non-constant anatomical-like volume
        arr = np.zeros(shape, dtype=np.float32)
        slices = tuple(slice(max(1, s // 4), max(2, 3 * s // 4)) for s in shape)
        arr[slices] = 42.0

    if has_nan:
        mid = tuple(s // 2 for s in shape)
        arr[mid] = np.nan
    if has_inf:
        mid = tuple(s // 2 for s in shape)
        arr[mid] = np.inf

    if affine is None:
        affine = np.eye(4, dtype=np.float64)
        for i in range(min(3, len(zooms))):
            affine[i, i] = zooms[i]

    img = nib.Nifti1Image(arr, affine)
    nib.save(img, str(dest_path))
    return dest_path


class TestJudgesFailurePack:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="neuroscope_judge_tests_"))
        self.validator = NIfTIValidator()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_fake_nifti_file(self):
        """Failure Case 1: Plain text file renamed to .nii.gz."""
        fake_path = self.temp_dir / "fake_scan.nii.gz"
        fake_path.write_text("This is an ordinary text file, not a real NIfTI MRI volume!")

        with pytest.raises(InvalidNIfTIFormatError) as exc_info:
            self.validator.validate_single_file(
                file_path=fake_path,
                original_filename="fake_scan.nii.gz",
                modality="T1",
            )
        assert exc_info.value.error_code == "INVALID_NIFTI_FORMAT"

    def test_02_corrupted_gzip_file(self):
        """Failure Case 2: Corrupted or truncated Gzip archive."""
        corrupt_path = self.temp_dir / "corrupted_archive.nii.gz"
        corrupt_bytes = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + b"Corrupted broken gzip stream"
        corrupt_path.write_bytes(corrupt_bytes)

        with pytest.raises(InvalidNIfTIFormatError) as exc_info:
            self.validator.validate_single_file(
                file_path=corrupt_path,
                original_filename="corrupted_archive.nii.gz",
                modality="T2",
            )
        assert exc_info.value.error_code == "INVALID_NIFTI_FORMAT"

    def test_03_missing_mri_modality(self):
        """Failure Case 3: Missing MRI modality (FLAIR omitted from study)."""
        staged = {}
        for mod in ["T1", "T1Gd", "T2"]:
            p = make_synthetic_nifti(self.temp_dir / f"{mod}.nii.gz")
            staged[mod] = (p, f"{mod}.nii.gz", p.stat().st_size)

        with pytest.raises(MissingModalityError) as exc_info:
            self.validator.validate_study(staged)
        assert exc_info.value.error_code == "MISSING_MODALITY"
        assert "FLAIR" in exc_info.value.details["missing_modalities"]

    def test_04_mismatched_image_shapes(self):
        """Failure Case 4: Mismatched dimensions across MRI sequences."""
        staged = {}
        for mod in ["T1", "T1Gd", "T2"]:
            p = make_synthetic_nifti(self.temp_dir / f"{mod}.nii.gz", shape=(64, 64, 32))
            staged[mod] = (p, f"{mod}.nii.gz", p.stat().st_size)

        flair_p = make_synthetic_nifti(self.temp_dir / "FLAIR.nii.gz", shape=(64, 64, 48))
        staged["FLAIR"] = (flair_p, "FLAIR.nii.gz", flair_p.stat().st_size)

        with pytest.raises(ShapeMismatchError) as exc_info:
            self.validator.validate_study(staged)
        assert exc_info.value.error_code == "SHAPE_MISMATCH"
        assert exc_info.value.details["modality"] == "FLAIR"

    def test_05_two_dimensional_image(self):
        """Failure Case 5: 2D image rejected (NeuroScope requires strictly 3D)."""
        p_2d = make_synthetic_nifti(self.temp_dir / "slice_2d.nii.gz", is_2d=True)

        with pytest.raises(InvalidDimensionsError) as exc_info:
            self.validator.validate_single_file(
                file_path=p_2d,
                original_filename="slice_2d.nii.gz",
                modality="T1",
            )
        assert exc_info.value.error_code == "INVALID_DIMENSIONS"
        assert exc_info.value.details["received_dimensions"] == 2

    def test_06_all_zero_image(self):
        """Failure Case 6: All-zero constant image rejected."""
        zero_path = make_synthetic_nifti(self.temp_dir / "zero_volume.nii.gz", constant_val=0.0)

        with pytest.raises(ConstantImageError) as exc_info:
            self.validator.validate_single_file(
                file_path=zero_path,
                original_filename="zero_volume.nii.gz",
                modality="T1Gd",
            )
        assert exc_info.value.error_code == "CONSTANT_IMAGE"
        assert exc_info.value.details["constant_value"] == 0.0

    def test_07_nan_values(self):
        """Failure Case 7: Non-finite NaN voxel values rejected."""
        nan_path = make_synthetic_nifti(self.temp_dir / "nan_volume.nii.gz", has_nan=True)

        with pytest.raises(NonFiniteValuesError) as exc_info:
            self.validator.validate_single_file(
                file_path=nan_path,
                original_filename="nan_volume.nii.gz",
                modality="FLAIR",
            )
        assert exc_info.value.error_code == "NON_FINITE_VALUES"
        assert exc_info.value.details["non_finite_count"] >= 1

    def test_08_file_larger_than_1gb(self):
        """Failure Case 8: Upload exceeding 1 GiB triggers immediate streaming cutoff."""
        max_bytes = 1024 * 1024 * 1024
        dest = self.temp_dir / "oversized.nii.gz"

        class LargeStream:
            def __init__(self, size_to_yield):
                self.remaining = size_to_yield
                self.chunk = b"X" * (128 * 1024)

            def read(self, n=None):
                if self.remaining <= 0:
                    return b""
                to_read = min(len(self.chunk), self.remaining)
                self.remaining -= to_read
                return self.chunk[:to_read]

        stream = LargeStream(max_bytes + 100 * 1024)

        with pytest.raises(FileTooLargeError) as exc_info:
            self.validator.save_upload_stream(
                file_stream=stream,
                dest_path=dest,
                filename="oversized.nii.gz",
                modality="T1",
            )
        assert exc_info.value.error_code == "FILE_TOO_LARGE"
        assert exc_info.value.status_code == 413
        assert not dest.exists(), "Oversized partial file was not cleaned up!"
