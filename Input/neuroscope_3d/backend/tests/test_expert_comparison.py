"""Expert ground-truth comparison: mask checks, overlap scores, doctor's volume readings and the report.

Synthetic data only. To keep the model out of these plumbing tests, a known label map is written as the study's
prediction; the expected scores then follow by hand from the cube geometry."""

import nibabel as nib
import numpy as np
import pytest
from starlette.testclient import TestClient

from app.main import app
from app.services import viewer_data
from tests.test_upload_to_prediction import lps_aniso_affine, phantom

SHAPE, AFF = (160, 192, 96), lps_aniso_affine()        # 1 x 1 x 1.5 mm, LPS
VOX_ML = 1.0 * 1.0 * 1.5 / 1000


def cubes(dx=0):
    """ED shell 20^3 with an ET 10^3 block and an NCR 4^3 core, shifted dx voxels along axis 0."""
    a = np.zeros(SHAPE, np.uint8)
    a[60 + dx:80 + dx, 80:100, 40:60] = 2
    a[65 + dx:75 + dx, 85:95, 45:55] = 4
    a[68 + dx:72 + dx, 88:92, 48:52] = 1
    return a


def nii(arr, aff=AFF):
    return nib.Nifti1Image(arr, aff).to_bytes()


@pytest.fixture(scope="module")
def sid():
    client = TestClient(app)
    data = phantom(SHAPE, AFF)
    files = [(k, (f"Case01_{n}.nii", data[n], "application/octet-stream"))
             for k, n in (("t1", "t1"), ("t1gd", "t1ce"), ("t2", "t2"), ("flair", "flair"))]
    up = client.post("/api/upload/validate", files=files)
    assert up.status_code == 200, up.json()
    study = viewer_data.get_study(up.json()["study_id"])
    nib.save(nib.Nifti1Image(cubes(), AFF), f"{study.study_dir}/{viewer_data.PREDICTION_FILENAME}")
    return study.study_id


def post_mask(client, sid, arr, aff=AFF, name="expert_seg.nii"):
    return client.post(f"/api/viewer/{sid}/expert", files={"file": (name, nii(arr, aff), "application/octet-stream")})


def test_overlap_scores_match_hand_calculation(sid):
    client = TestClient(app)
    assert post_mask(client, sid, cubes(dx=10)).status_code == 200
    ag = client.get(f"/api/viewer/{sid}/meta").json()["agreement"]
    # WT: 20^3 = 8000 voxels each, shifted by 10 -> overlap 10*20*20 = 4000
    assert ag["WT"]["dice"] == pytest.approx(0.5) and ag["WT"]["iou"] == pytest.approx(1 / 3)
    assert ag["WT"]["sensitivity"] == pytest.approx(0.5) and ag["WT"]["precision"] == pytest.approx(0.5)
    assert ag["WT"]["hd95_mm"] > 0
    # TC (1+4) is a 10^3 block, shifted by 10 -> no overlap at all
    assert ag["TC"]["pred_voxels"] == ag["TC"]["ref_voxels"] == 1000
    for k in ("dice", "iou", "sensitivity", "precision"):
        assert ag["TC"][k] == 0.0 and ag["ET"][k] == 0.0


def test_mask_checks_name_what_differs(sid):
    client = TestClient(app)
    flipped = AFF.copy(); flipped[0, 0] = 1.0; flipped[0, 3] = -90.0               # RPS instead of LPS
    r = post_mask(client, sid, cubes(), flipped)
    assert r.status_code == 400 and r.json()["error_code"] == "EXPERT_MASK_ORIENTATION_MISMATCH"
    coarse = AFF.copy(); coarse[2, 2] = 3.0                                        # 3 mm slices instead of 1.5
    r = post_mask(client, sid, cubes(), coarse)
    assert r.json()["error_code"] == "EXPERT_MASK_SPACING_MISMATCH"
    moved = AFF.copy(); moved[:3, 3] += (5.0, 0, 0)                                # same grid, shifted 5 mm
    r = post_mask(client, sid, cubes(), moved)
    assert r.json()["error_code"] == "EXPERT_MASK_AFFINE_MISMATCH" and "5.00 mm" in r.json()["message"]
    r = post_mask(client, sid, np.zeros((10, 10, 10), np.uint8))
    assert r.json()["error_code"] == "EXPERT_MASK_SHAPE_MISMATCH"
    bad = cubes(); bad[0, 0, 0] = 3
    assert post_mask(client, sid, bad).json()["error_code"] == "INVALID_EXPERT_LABELS"
    r = client.post(f"/api/viewer/{sid}/expert", files={"file": ("mask.png", b"not nifti", "image/png")})
    assert r.status_code >= 400


def test_model_prediction_is_refused_as_expert_reference(sid):
    client = TestClient(app)
    r = post_mask(client, sid, cubes())                                            # identical to the prediction
    assert r.status_code == 400 and r.json()["error_code"] == "EXPERT_MASK_IS_PREDICTION"


def test_empty_expert_mask_gives_undefined_not_perfect_scores(sid):
    client = TestClient(app)
    assert post_mask(client, sid, np.zeros(SHAPE, np.uint8)).status_code == 200
    m = client.get(f"/api/viewer/{sid}/meta").json()
    wt = m["agreement"]["WT"]
    assert wt["dice"] == 0.0 and wt["iou"] == 0.0 and wt["precision"] == 0.0
    assert wt["sensitivity"] is None and wt["hd95_mm"] is None
    assert any("empty" in n for n in m["expert_notes"])
    assert client.delete(f"/api/viewer/{sid}/expert").status_code == 200
    m = client.get(f"/api/viewer/{sid}/meta").json()
    assert not m["has_expert"] and m["agreement"] is None


def test_doctor_readings_give_volume_differences_only(sid):
    client = TestClient(app)
    url = f"/api/viewer/{sid}/readings"
    assert client.post(url, json={}).json()["error_code"] == "INVALID_READING"
    assert client.post(url, json={"WT": -1}).json()["error_code"] == "INVALID_READING"
    assert client.post(url, json={"WT": 5000}).json()["error_code"] == "INVALID_READING"
    assert client.post(url, json={"WT": 10, "TC": 20}).json()["error_code"] == "INCONSISTENT_READING"
    assert client.post(url, json={"WT": 10, "TC": 2, "ET": 0}).status_code == 200

    m = client.get(f"/api/viewer/{sid}/meta").json()
    rc = m["readings_comparison"]
    model_wt = 8000 * VOX_ML                                                       # 12.0 mL
    assert rc["WT"]["model_mL"] == pytest.approx(model_wt)
    assert rc["WT"]["difference_mL"] == pytest.approx(model_wt - 10)
    assert rc["WT"]["percent_difference"] == pytest.approx(100 * (model_wt - 10) / 10)
    assert rc["TC"]["difference_mL"] == pytest.approx(1000 * VOX_ML - 2)
    assert rc["ET"]["percent_difference"] is None                                  # entered 0: % undefined
    assert set(rc["WT"]) == {"model_mL", "doctor_mL", "difference_mL", "abs_difference_mL", "percent_difference"}


def test_report_shows_comparison_apart_from_benchmark(sid):
    client = TestClient(app)
    assert post_mask(client, sid, cubes(dx=10)).status_code == 200
    client.post(f"/api/viewer/{sid}/readings", json={"WT": 10})
    page = client.get(f"/report/{sid}").text
    assert "Expert comparison (this case only)" in page and "Precision" in page and "IoU" in page
    assert "Doctor&#x27;s volume readings" in page or "Doctor's volume readings" in page
    assert page.index("Expert comparison (this case only)") < page.index("<h2>Part B")      # before the benchmark
    pdf = client.get(f"/api/report/{sid}.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
