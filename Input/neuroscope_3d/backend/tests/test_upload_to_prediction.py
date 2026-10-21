"""
End-to-end test: multipart upload -> validation -> SegResNet inference -> NIfTI label map download.
Uses a synthetic phantom (not real MRI), so it checks plumbing and output format, not accuracy.
Skipped when torch/MONAI/model weights are not installed.
"""

import io

import nibabel as nib
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.services.segmentation_engine import segmentation_pipeline
from app.services.study_store import study_store

ready, missing = segmentation_pipeline.check_dependencies()
pytestmark = pytest.mark.skipif(not ready, reason=f"inference unavailable: {missing}")

# Relative intensity of (brain, oedema, necrotic core, enhancing rim) per sequence
CONTRAST = {
    "t1": (0.60, 0.50, 0.30, 0.50),
    "t1ce": (0.60, 0.55, 0.30, 1.30),
    "t2": (0.50, 1.00, 1.20, 0.80),
    "flair": (0.50, 1.10, 0.70, 0.90),
}


def phantom(shape, affine):
    """Ellipsoid 'brain' with a nested lesion: oedema shell, enhancing rim, necrotic core."""
    rng = np.random.default_rng(0)
    g = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    brain = (g[0] / 0.75) ** 2 + (g[1] / 0.85) ** 2 + (g[2] / 0.85) ** 2 <= 1
    r = np.sqrt((g[0] - 0.3) ** 2 + (g[1] - 0.1) ** 2 + g[2] ** 2)
    files = {}
    for seq, (b, ed, ncr, et) in CONTRAST.items():
        v = np.where(brain, b, 0.0)
        v[(r < 0.35) & brain] = ed
        v[r < 0.2] = et
        v[r < 0.12] = ncr
        v = (v + rng.normal(0, 0.02, shape) * brain) * 1000 * brain
        buf = io.BytesIO()
        img = nib.Nifti1Image(v.astype(np.float32), affine)
        buf.write(img.to_bytes())
        files[seq] = buf.getvalue()
    return files


def lps_aniso_affine():
    aff = np.diag([-1.0, -1.0, 1.5, 1.0])
    aff[:3, 3] = (90.0, 100.0, -60.0)
    return aff


@pytest.mark.parametrize(
    "shape,affine",
    [((160, 192, 128), np.eye(4)), ((160, 192, 96), lps_aniso_affine())],
    ids=["RAS-1mm", "LPS-1x1x1.5mm"],
)
def test_upload_to_brats_label_map(shape, affine):
    client = TestClient(app)
    study_store.clear()
    try:
        data = phantom(shape, affine)
        files = [
            ("t1", ("Case01_t1.nii", data["t1"], "application/octet-stream")),
            ("t1gd", ("Case01_t1ce.nii", data["t1ce"], "application/octet-stream")),
            ("t2", ("Case01_t2.nii", data["t2"], "application/octet-stream")),
            ("flair", ("Case01_flair.nii", data["flair"], "application/octet-stream")),
        ]
        up = client.post("/api/upload/validate", files=files)
        assert up.status_code == 200, up.json()
        study_id = up.json()["study_id"]

        # No prediction before inference has run
        assert client.get(f"/api/segmentation/prediction/{study_id}").status_code == 404

        seg = client.post("/api/segmentation/run", json={"study_id": study_id})
        assert seg.status_code == 200, seg.json()
        body = seg.json()
        assert body["label_codes"] == {"0": "background", "1": "NCR", "2": "ED", "4": "ET"}
        assert body["shape"] == list(shape)
        assert len(body["provenance"]["inputs"]) == 4
        assert all(len(v["sha256"]) == 64 for v in body["provenance"]["inputs"].values())
        assert "Not a diagnosis" in body["disclaimer"]

        dl = client.get(body["prediction_url"])
        assert dl.status_code == 200
        pred = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(dl.content))
        lab = np.asarray(pred.dataobj)

        # Output sits on the uploaded grid and affine, in BraTS codes only
        assert pred.shape == shape
        assert np.allclose(pred.affine, affine)
        assert lab.dtype == np.uint8
        assert set(np.unique(lab).tolist()) <= {0, 1, 2, 4}

        # Reported volumes == voxel count x voxel volume
        voxel_mm3 = abs(np.linalg.det(affine[:3, :3]))
        vols = body["estimated_volumes"]
        for name, codes in {"NCR": [1], "ED": [2], "ET": [4], "TC": [1, 4], "WT": [1, 2, 4]}.items():
            count = int(np.isin(lab, codes).sum())
            assert vols[name]["voxels"] == count
            assert vols[name]["mm3"] == pytest.approx(count * voxel_mm3)
            assert vols[name]["mL"] == pytest.approx(count * voxel_mm3 / 1000)
        assert body["whole_tumor_volume_cm3"] == round(vols["WT"]["mL"], 2)

        # The phantom lesion should be picked up somewhere inside the brain
        assert vols["WT"]["voxels"] > 0
    finally:
        study_store.clear()


def test_single_call_predict_endpoint():
    client = TestClient(app)
    study_store.clear()
    try:
        shape, affine = (128, 160, 96), lps_aniso_affine()
        data = phantom(shape, affine)
        files = [(k, (f"Case02_{n}.nii", data[n], "application/octet-stream"))
                 for k, n in (("t1", "t1"), ("t1gd", "t1ce"), ("t2", "t2"), ("flair", "flair"))]
        res = client.post("/api/segmentation/predict", files=files)
        assert res.status_code == 200
        assert res.headers["x-study-id"].startswith("brats_")
        pred = nib.Nifti1Image.from_bytes(__import__("gzip").decompress(res.content))
        assert pred.shape == shape and np.allclose(pred.affine, affine)
        assert set(np.unique(np.asarray(pred.dataobj)).tolist()) <= {0, 1, 2, 4}
    finally:
        study_store.clear()


def test_rejected_upload_never_reaches_inference():
    client = TestClient(app)
    study_store.clear()
    good = phantom((64, 64, 48), np.eye(4))
    files = [
        ("t1", ("Case01_t1.nii", good["t1"], "application/octet-stream")),
        ("t1gd", ("Case01_t1ce.nii", good["t1ce"], "application/octet-stream")),
        ("t2", ("Case01_t2.nii", good["t2"], "application/octet-stream")),
        ("flair", ("Case01_flair.nii.gz", b"not a nifti", "application/gzip")),
    ]
    up = client.post("/api/upload/validate", files=files)
    assert up.status_code == 400
    assert up.json()["valid"] is False
    assert "study_id" not in up.json()
    assert client.post("/api/segmentation/run", json={"study_id": "brats_doesnotexist"}).status_code == 404
