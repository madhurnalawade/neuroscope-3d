"""
FastAPI API Integration Test Suite for NeuroScope 3D.
Tests end-to-end multipart form uploads, single-slot validation, study registration,
and segmentation pipeline access control with Starlette TestClient.
"""

import io
import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.services.study_store import study_store


def create_in_memory_nifti(shape=(32, 32, 16), zooms=(1.0, 1.0, 1.0), constant_val=None) -> bytes:
    """Generate in-memory .nii.gz bytes for multipart upload testing."""
    if constant_val is not None:
        arr = np.full(shape, constant_val, dtype=np.float32)
    else:
        arr = np.zeros(shape, dtype=np.float32)
        arr[8:24, 8:24, 4:12] = 30.0

    affine = np.eye(4, dtype=np.float64)
    for i in range(min(3, len(zooms))):
        affine[i, i] = zooms[i]

    img = nib.Nifti1Image(arr, affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        nib.save(img, str(tmp_path))
        return tmp_path.read_bytes()
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


class TestAPIIntegration:
    def setup_method(self):
        self.client = TestClient(app)
        study_store.clear()

    def teardown_method(self):
        study_store.clear()

    def test_01_health_check_endpoint(self):
        """API Test 1: Service health check endpoint responds with 200 OK."""
        res = self.client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Voxels"
        assert data["ready_for_uploads"] is True

    def test_02_single_file_validation_endpoint_valid(self):
        """API Test 2: Single-file validation endpoint accepts valid NIfTI scan."""
        nii_bytes = create_in_memory_nifti(shape=(32, 32, 16), zooms=(1.0, 1.0, 1.0))
        files = {"file": ("BraTS2021_00001_t1.nii.gz", nii_bytes, "application/gzip")}
        data = {"modality": "T1"}

        res = self.client.post("/api/upload/validate-single", files=files, data=data)
        assert res.status_code == 200
        body = res.json()
        assert body["valid"] is True
        assert body["modality"] == "T1"
        assert body["shape"] == [32, 32, 16]
        assert body["zooms"] == [1.0, 1.0, 1.0]

    def test_03_single_file_validation_endpoint_invalid(self):
        """API Test 3: Single-file validation endpoint rejects corrupted scan."""
        fake_bytes = b"NOT_A_VALID_NIFTI_FILE"
        files = {"file": ("BraTS2021_00001_t2.nii.gz", fake_bytes, "application/gzip")}
        data = {"modality": "T2"}

        res = self.client.post("/api/upload/validate-single", files=files, data=data)
        assert res.status_code == 400
        body = res.json()
        assert body["valid"] is False
        assert body["error_code"] == "INVALID_NIFTI_FORMAT"

    def test_04_full_study_upload_and_validation(self):
        """API Test 4: Complete multipart upload of all four modalities."""
        t1_bytes = create_in_memory_nifti()
        t1ce_bytes = create_in_memory_nifti()
        t2_bytes = create_in_memory_nifti()
        flair_bytes = create_in_memory_nifti()

        files = [
            ("t1", ("BraTS2021_00001_t1.nii.gz", t1_bytes, "application/gzip")),
            ("t1gd", ("BraTS2021_00001_t1ce.nii.gz", t1ce_bytes, "application/gzip")),
            ("t2", ("BraTS2021_00001_t2.nii.gz", t2_bytes, "application/gzip")),
            ("flair", ("BraTS2021_00001_flair.nii.gz", flair_bytes, "application/gzip")),
        ]

        res = self.client.post("/api/upload/validate", files=files)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["valid"] is True
        assert body["ready_for_segmentation"] is True
        assert "study_id" in body
        study_id = body["study_id"]

        status_res = self.client.get(f"/api/segmentation/study/{study_id}")
        assert status_res.status_code == 200
        status_body = status_res.json()
        assert status_body["status"] == "success"
        assert status_body["validated"] is True
        assert len(status_body["modalities"]) == 4

    def test_05_segmentation_endpoint_security_and_dependency_reporting(self):
        """API Test 5: Rejects unauthorized study IDs and reports exact missing dependencies."""
        res_fake = self.client.post("/api/segmentation/run", json={"study_id": "fake_study_12345"})
        assert res_fake.status_code == 404
        assert res_fake.json()["error_code"] == "STUDY_NOT_FOUND"

        t1_bytes = create_in_memory_nifti()
        files = [
            ("t1", ("BraTS2021_00001_t1.nii.gz", t1_bytes, "application/gzip")),
            ("t1gd", ("BraTS2021_00001_t1ce.nii.gz", t1_bytes, "application/gzip")),
            ("t2", ("BraTS2021_00001_t2.nii.gz", t1_bytes, "application/gzip")),
            ("flair", ("BraTS2021_00001_flair.nii.gz", t1_bytes, "application/gzip")),
        ]
        upload_res = self.client.post("/api/upload/validate", files=files)
        assert upload_res.status_code == 200
        study_id = upload_res.json()["study_id"]

        seg_res = self.client.post("/api/segmentation/run", json={"study_id": study_id})
        if seg_res.status_code != 200:
            assert seg_res.status_code == 500
            err_msg = seg_res.json()["message"]
            assert "Missing components:" in err_msg
            assert "PyTorch (torch)" in err_msg
            assert "MONAI (monai)" in err_msg
        else:
            body = seg_res.json()
            assert body["status"] == "success"
            assert body["label_codes"] == {"0": "background", "1": "NCR", "2": "ED", "4": "ET"}
