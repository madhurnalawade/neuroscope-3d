"""One 4-channel NIfTI upload must behave exactly like uploading the four sequences separately."""

import gzip

import nibabel as nib
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.services.segmentation_engine import segmentation_pipeline
from app.services.study_store import study_store
from tests.test_upload_to_prediction import lps_aniso_affine, phantom

ready, missing = segmentation_pipeline.check_dependencies()
MSD_ORDER = ("flair", "t1", "t1ce", "t2")  # Decathlon channel order


def _stack_4d(data, affine, order=MSD_ORDER):
    vols = [np.asarray(nib.Nifti1Image.from_bytes(data[k]).dataobj) for k in order]
    return nib.Nifti1Image(np.stack(vols, axis=-1).astype(np.float32), affine).to_bytes()


def _label_map(client, url):
    return np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(client.get(url).content)).dataobj)


@pytest.mark.skipif(not ready, reason=f"inference unavailable: {missing}")
def test_4d_upload_matches_four_file_upload():
    client = TestClient(app)
    study_store.clear()
    shape, aff = (128, 160, 96), lps_aniso_affine()
    data = phantom(shape, aff)
    try:
        four = client.post("/api/upload/validate", files=[
            ("t1", ("c_t1.nii", data["t1"], "application/octet-stream")),
            ("t1gd", ("c_t1ce.nii", data["t1ce"], "application/octet-stream")),
            ("t2", ("c_t2.nii", data["t2"], "application/octet-stream")),
            ("flair", ("c_flair.nii", data["flair"], "application/octet-stream")),
        ]).json()
        one = client.post("/api/upload/validate-4d",
                          files={"file": ("case.nii", _stack_4d(data, aff), "application/octet-stream")}).json()
        assert one["valid"] and one["metadata"]["source_4d"]["channel_order"] == ["FLAIR", "T1", "T1Gd", "T2"]
        assert len(one["metadata"]["source_4d"]["sha256"]) == 64

        r4 = client.post("/api/segmentation/run", json={"study_id": four["study_id"]}).json()
        r1 = client.post("/api/segmentation/run", json={"study_id": one["study_id"]}).json()
        a = _label_map(client, r4["prediction_url"])
        b = _label_map(client, r1["prediction_url"])
        assert a.shape == b.shape == shape
        assert np.array_equal(a, b)

        # Custom channel order is honoured
        custom = client.post("/api/upload/validate-4d",
                             data={"channel_order": "t1ce,t2,flair,t1"},
                             files={"file": ("case.nii", _stack_4d(data, aff, ("t1ce", "t2", "flair", "t1")),
                                             "application/octet-stream")}).json()
        rc = client.post("/api/segmentation/run", json={"study_id": custom["study_id"]}).json()
        assert np.array_equal(_label_map(client, rc["prediction_url"]), a)
    finally:
        study_store.clear()


def test_4d_upload_rejects_wrong_channel_count_and_bad_order():
    client = TestClient(app)
    three = nib.Nifti1Image(np.random.rand(32, 32, 16, 3).astype(np.float32), np.eye(4)).to_bytes()
    res = client.post("/api/upload/validate-4d", files={"file": ("x.nii", three, "application/octet-stream")})
    assert res.status_code == 400 and res.json()["error_code"] == "NOT_FOUR_CHANNEL_STUDY"

    plain3d = nib.Nifti1Image(np.random.rand(32, 32, 16).astype(np.float32), np.eye(4)).to_bytes()
    res = client.post("/api/upload/validate-4d", files={"file": ("x.nii", plain3d, "application/octet-stream")})
    assert res.json()["error_code"] == "NOT_FOUR_CHANNEL_STUDY"

    four = nib.Nifti1Image(np.random.rand(32, 32, 16, 4).astype(np.float32), np.eye(4)).to_bytes()
    res = client.post("/api/upload/validate-4d", data={"channel_order": "t1,t1,t2,flair"},
                      files={"file": ("x.nii", four, "application/octet-stream")})
    assert res.json()["error_code"] == "INVALID_CHANNEL_ORDER"

    res = client.post("/api/upload/validate-4d", files={"file": ("x.nii.gz", b"garbage", "application/gzip")})
    assert res.status_code == 400 and res.json()["error_code"] == "INVALID_NIFTI_FORMAT"
