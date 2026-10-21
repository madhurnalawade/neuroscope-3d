"""
Extended Validation Test Suite for NeuroScope 3D.
Covers 20 edge-case and robustness validation scenarios.
"""

import gzip
import io
import shutil
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from app.core.exceptions import (
    AffineMismatchError,
    ConstantImageError,
    DuplicateModalityError,
    InvalidExtensionError,
    InvalidNIfTIFormatError,
    NonFiniteValuesError,
    SingularAffineError,
    StudyExpiredError,
    StudyNotFoundError,
)
from app.core.security import sanitize_filename
from app.services.nifti_validator import NIfTIValidator
from app.services.segmentation_engine import segmentation_pipeline
from app.services.study_store import study_store


def make_test_nifti(
    dest_path: Path,
    shape=(32, 32, 16),
    zooms=(1.0, 1.0, 1.0),
    affine=None,
    constant_val=None,
    has_pos_inf=False,
    has_neg_inf=False,
    is_singular=False,
) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if constant_val is not None:
        arr = np.full(shape, constant_val, dtype=np.float32)
    else:
        arr = np.zeros(shape, dtype=np.float32)
        arr[8:24, 8:24, 4:12] = 25.0

    if has_pos_inf:
        arr[shape[0] // 2, shape[1] // 2, shape[2] // 2] = np.inf
    if has_neg_inf:
        arr[shape[0] // 2, shape[1] // 2, shape[2] // 2] = -np.inf

    if affine is None:
        affine = np.eye(4, dtype=np.float64)
        for i in range(min(3, len(zooms))):
            affine[i, i] = zooms[i]

    img = nib.Nifti1Image(arr, affine)
    if is_singular:
        # Real nibabel cannot encode a singular matrix as a qform, so write it only as the sform.
        singular = affine.copy()
        singular[0, :] = 0.0
        img.header.set_qform(None, code=0)
        img.header.set_sform(singular, code=1)
        img = nib.Nifti1Image(arr, None, img.header)
    nib.save(img, str(dest_path))
    return dest_path


class TestAdditionalValidation:
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="neuroscope_ext_tests_"))
        self.validator = NIfTIValidator()
        study_store.clear()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        study_store.clear()

    def test_01_valid_brats_study(self):
        """Scenario 1: Full 4-modality study with matching dimensions and affines."""
        staged = {}
        for mod in ["T1", "T1Gd", "T2", "FLAIR"]:
            p = make_test_nifti(self.temp_dir / f"{mod}.nii.gz")
            staged[mod] = (p, f"{mod}.nii.gz", p.stat().st_size)

        valid, validated_seqs, metadata, duration = self.validator.validate_study(staged)
        assert valid is True
        assert len(validated_seqs) == 4
        assert metadata["affine_consistent"] is True
        assert metadata["all_finite"] is True
        assert metadata["all_non_constant"] is True

    def test_02_nii_upload(self):
        """Scenario 2: Uncompressed .nii upload is accepted."""
        p = make_test_nifti(self.temp_dir / "t1.nii")
        info = self.validator.validate_single_file(p, "t1.nii", "T1")
        assert info.shape == (32, 32, 16)

    def test_03_niigz_upload(self):
        """Scenario 3: Compressed .nii.gz upload is accepted."""
        p = make_test_nifti(self.temp_dir / "t1ce.nii.gz")
        info = self.validator.validate_single_file(p, "t1ce.nii.gz", "T1Gd")
        assert info.shape == (32, 32, 16)

    def test_04_mixed_nii_and_niigz(self):
        """Scenario 4: Mixed study containing both .nii and .nii.gz."""
        staged = {}
        exts = {"T1": ".nii", "T1Gd": ".nii.gz", "T2": ".nii", "FLAIR": ".nii.gz"}
        for mod, ext in exts.items():
            p = make_test_nifti(self.temp_dir / f"{mod}{ext}")
            staged[mod] = (p, f"{mod}{ext}", p.stat().st_size)

        valid, seqs, meta, dur = self.validator.validate_study(staged)
        assert valid is True

    def test_05_duplicate_modalities(self):
        """Scenario 5: Duplicate modality detection error."""
        err = DuplicateModalityError(modality="T1", count=2)
        assert err.error_code == "DUPLICATE_MODALITY"
        assert err.details["modality"] == "T1"
        assert err.details["count"] == 2

    def test_06_wrong_extensions(self):
        """Scenario 6: Rejection of invalid non-NIfTI file extensions."""
        p = self.temp_dir / "brain_scan.dcm"
        p.write_bytes(b"DICOM_DUMMY_DATA")

        with pytest.raises(InvalidExtensionError) as exc_info:
            self.validator.validate_single_file(p, "brain_scan.dcm", "T1")
        assert exc_info.value.error_code == "INVALID_FILE_EXTENSION"
        assert exc_info.value.details["received_extension"] == ".dcm"

    def test_07_positive_infinity(self):
        """Scenario 7: Rejection of +Infinity voxel intensities."""
        p = make_test_nifti(self.temp_dir / "pos_inf.nii.gz", has_pos_inf=True)
        with pytest.raises(NonFiniteValuesError) as exc_info:
            self.validator.validate_single_file(p, "pos_inf.nii.gz", "T1")
        assert exc_info.value.error_code == "NON_FINITE_VALUES"

    def test_08_negative_infinity(self):
        """Scenario 8: Rejection of -Infinity voxel intensities."""
        p = make_test_nifti(self.temp_dir / "neg_inf.nii.gz", has_neg_inf=True)
        with pytest.raises(NonFiniteValuesError) as exc_info:
            self.validator.validate_single_file(p, "neg_inf.nii.gz", "T2")
        assert exc_info.value.error_code == "NON_FINITE_VALUES"

    def test_09_constant_nonzero_image(self):
        """Scenario 9: Rejection of non-zero constant volumes (all voxels = 100.0)."""
        p = make_test_nifti(self.temp_dir / "const_100.nii.gz", constant_val=100.0)
        with pytest.raises(ConstantImageError) as exc_info:
            self.validator.validate_single_file(p, "const_100.nii.gz", "FLAIR")
        assert exc_info.value.error_code == "CONSTANT_IMAGE"
        assert exc_info.value.details["constant_value"] == 100.0

    def test_10_affine_mismatch(self):
        """Scenario 10: Detection of spatial transformation discrepancy."""
        staged = {}
        for mod in ["T1", "T1Gd", "T2"]:
            p = make_test_nifti(self.temp_dir / f"{mod}.nii.gz")
            staged[mod] = (p, f"{mod}.nii.gz", p.stat().st_size)

        shifted_affine = np.eye(4, dtype=np.float64)
        shifted_affine[0, 3] = 15.0
        flair_p = make_test_nifti(self.temp_dir / "FLAIR.nii.gz", affine=shifted_affine)
        staged["FLAIR"] = (flair_p, "FLAIR.nii.gz", flair_p.stat().st_size)

        with pytest.raises(AffineMismatchError) as exc_info:
            self.validator.validate_study(staged)
        assert exc_info.value.error_code == "AFFINE_MISMATCH"
        assert exc_info.value.details["modality"] == "FLAIR"

    def test_11_orientation_mismatch(self):
        """Scenario 11: Rejection when scan axes are in conflicting coordinate spaces (LPS vs RAS)."""
        staged = {}
        for mod in ["T1", "T1Gd", "T2"]:
            p = make_test_nifti(self.temp_dir / f"{mod}.nii.gz")
            staged[mod] = (p, f"{mod}.nii.gz", p.stat().st_size)

        lps_affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        flair_p = make_test_nifti(self.temp_dir / "FLAIR.nii.gz", affine=lps_affine)
        staged["FLAIR"] = (flair_p, "FLAIR.nii.gz", flair_p.stat().st_size)

        with pytest.raises(AffineMismatchError) as exc_info:
            self.validator.validate_study(staged)
        assert exc_info.value.error_code == "AFFINE_MISMATCH"

    def test_12_different_voxel_spacing(self):
        """Scenario 12: Mismatched voxel spacing across sequences."""
        staged = {}
        for mod in ["T1", "T1Gd", "T2"]:
            p = make_test_nifti(self.temp_dir / f"{mod}.nii.gz", zooms=(1.0, 1.0, 1.0))
            staged[mod] = (p, f"{mod}.nii.gz", p.stat().st_size)

        flair_p = make_test_nifti(self.temp_dir / "FLAIR.nii.gz", zooms=(2.0, 2.0, 2.0))
        staged["FLAIR"] = (flair_p, "FLAIR.nii.gz", flair_p.stat().st_size)

        with pytest.raises(AffineMismatchError) as exc_info:
            self.validator.validate_study(staged)
        assert exc_info.value.error_code == "AFFINE_MISMATCH"

    def test_13_valid_anisotropic_spacing(self):
        """Scenario 13: Clinical anisotropic spacing (1×1×2 mm) is supported and valid."""
        p = make_test_nifti(self.temp_dir / "anisotropic.nii.gz", zooms=(1.0, 1.0, 2.0))
        info = self.validator.validate_single_file(p, "anisotropic.nii.gz", "T1")
        assert info.zooms[:3] == (1.0, 1.0, 2.0)

    def test_14_singular_affine(self):
        """Scenario 14: Degenerate/singular spatial transformation matrix rejected."""
        p = make_test_nifti(self.temp_dir / "singular.nii.gz", is_singular=True)
        with pytest.raises(SingularAffineError) as exc_info:
            self.validator.validate_single_file(p, "singular.nii.gz", "T1")
        assert exc_info.value.error_code == "SINGULAR_AFFINE"

    def test_15_truncated_nifti_payload(self):
        """Scenario 15: Valid header but truncated data array rejected."""
        valid_p = make_test_nifti(self.temp_dir / "valid_tmp.nii")
        data = valid_p.read_bytes()
        trunc_p = self.temp_dir / "truncated.nii"
        trunc_p.write_bytes(data[: len(data) // 2])

        with pytest.raises(InvalidNIfTIFormatError) as exc_info:
            self.validator.validate_single_file(trunc_p, "truncated.nii", "T1")
        assert exc_info.value.error_code == "INVALID_NIFTI_FORMAT"

    def test_16_corrupt_voxel_data_with_valid_header(self):
        """Scenario 16: Corrupt voxel data following a valid header."""
        hdr = nib.Nifti1Header()
        hdr.set_data_shape((100, 100, 100))
        hdr.set_data_dtype(np.float32)
        hdr_bytes = hdr.binaryblock
        corrupt_p = self.temp_dir / "corrupt_data.nii"
        corrupt_p.write_bytes(hdr_bytes + bytes([0, 0, 0, 0]) + b"CORRUPT_VOXEL_PAYLOAD")

        with pytest.raises(InvalidNIfTIFormatError) as exc_info:
            self.validator.validate_single_file(corrupt_p, "corrupt_data.nii", "T2")
        assert exc_info.value.error_code == "INVALID_NIFTI_FORMAT"

    def test_17_malicious_filename_sanitization(self):
        """Scenario 17: Filename sanitization strips traversal, null bytes, and dangerous characters."""
        dangerous = "../../../etc/passwd" + chr(0) + ";rm -rf /;evil.nii.gz"
        safe = sanitize_filename(dangerous)
        assert "/" not in safe
        assert ".." not in safe
        assert chr(0) not in safe
        assert safe.endswith(".nii.gz")

    def test_18_interrupted_upload_cleanup(self):
        """Scenario 18: File cleaned up immediately when upload stream fails."""
        dest = self.temp_dir / "failed_upload.nii.gz"

        class FailingStream:
            def read(self, n=None):
                raise IOError("Network connection reset mid-upload")

        with pytest.raises(IOError):
            self.validator.save_upload_stream(FailingStream(), dest, "failed_upload.nii.gz", "T1")
        assert not dest.exists(), "Temporary file was not cleaned up on stream failure"

    def test_19_expired_study_id_rejection(self):
        """Scenario 19: Expired study IDs are rejected according to 60-minute retention policy."""
        study_id = "test_expired_study"
        study_dir = self.temp_dir / study_id
        study_dir.mkdir(parents=True, exist_ok=True)

        study_store.register_study(
            study_id=study_id,
            study_dir=study_dir,
            modalities={"T1": {"status": "valid"}},
            metadata={"shape": [32, 32, 16]},
            ttl_seconds=0.05,
        )

        time.sleep(0.08)
        with pytest.raises(StudyExpiredError) as exc_info:
            study_store.get_validated_study(study_id)
        assert exc_info.value.error_code == "STUDY_EXPIRED"
        assert exc_info.value.status_code == 410

    def test_20_invalid_study_id_segmentation_rejection(self):
        """Scenario 20: Segmentation pipeline rejects unregistered study IDs."""
        with pytest.raises(StudyNotFoundError) as exc_info:
            segmentation_pipeline.run_segmentation("non_existent_study_id_99999")
        assert exc_info.value.error_code == "STUDY_NOT_FOUND"
        assert exc_info.value.status_code == 404
