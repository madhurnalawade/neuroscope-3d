"""Viewer data, expert mask, suitability refusal and report endpoints (synthetic phantom; plumbing only)."""

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


def _nifti_bytes(arr, affine):
    return nib.Nifti1Image(arr, affine).to_bytes()


def _upload(client, data, extra=None):
    files = [
        ("t1", ("Case01_t1.nii", data["t1"], "application/octet-stream")),
        ("t1gd", ("Case01_t1ce.nii", data["t1ce"], "application/octet-stream")),
        ("t2", ("Case01_t2.nii", data["t2"], "application/octet-stream")),
        ("flair", ("Case01_flair.nii", data["flair"], "application/octet-stream")),
    ] + (extra or [])
    return client.post("/api/upload/validate", files=files)


def test_not_skull_stripped_input_is_refused():
    client = TestClient(app)
    rng = np.random.default_rng(1)
    noisy = {k: _nifti_bytes((rng.random((96, 96, 64)) * 100 + 1).astype(np.float32), np.eye(4))
             for k in ("t1", "t1ce", "t2", "flair")}
    res = _upload(client, noisy)
    assert res.status_code == 422
    assert res.json()["error_code"] == "UNSUITABLE_INPUT"
    assert "skull" in res.json()["message"]


def test_expert_mask_with_decathlon_codes_is_rejected():
    client = TestClient(app)
    shape, aff = (96, 96, 64), np.eye(4)
    bad = np.zeros(shape, np.uint8); bad[40:50, 40:50, 30:40] = 3
    res = _upload(client, phantom(shape, aff), [("expert", ("mask.nii", _nifti_bytes(bad, aff), "application/octet-stream"))])
    assert res.status_code == 400
    assert res.json()["error_code"] == "INVALID_EXPERT_LABELS"
    assert "Decathlon" in res.json()["message"]


def test_stl_export_is_a_closed_outward_solid_in_mm(monkeypatch):
    from collections import Counter

    from app.core.exceptions import NIfTIValidationError
    from app.services import viewer_data

    ball = np.zeros((40, 40, 40), np.uint8)
    g = np.meshgrid(*[np.arange(40)] * 3, indexing="ij")
    ball[sum((a - 20) ** 2 for a in g) < 12 ** 2] = 1
    mirror = np.diag([-1.0, 1.0, 1.0, 1.0])  # a mirrored affine flips the winding; export must undo it
    cached = {"brain": viewer_data._mesh(ball, mirror @ np.diag([0.5, 0.5, 0.5, 1]), 1.0, 1), "pred_ET": {"color": "#e8b030"}}
    monkeypatch.setattr(viewer_data, "meshes", lambda study: cached)

    stl = viewer_data.stl_bytes(None, ["brain"])
    n = int(np.frombuffer(stl[80:84], "<u4")[0])
    assert not stl.startswith(b"solid") and n == cached["brain"]["n_faces"] and len(stl) == 84 + 50 * n
    rec = np.frombuffer(stl[84:], viewer_data.STL_RECORD)
    tri = rec["v"].astype(np.float64)
    assert np.allclose(tri.reshape(-1, 3).min(0), 0)
    assert 10 < tri.reshape(-1, 3).max(0).max() < 14          # 24-voxel ball at 0.5 mm is about 12 mm across
    volume = np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6
    assert abs(volume - 4 / 3 * np.pi * 6 ** 3) / volume < 0.1  # positive: normals point outward
    edges = Counter(tuple(sorted((tuple(t[a]), tuple(t[b])))) for t in tri for a, b in ((0, 1), (1, 2), (2, 0)))
    assert set(edges.values()) == {2}                           # watertight: every edge shared by two triangles
    assert np.all(np.einsum("ij,ij->i", rec["normal"], tri.mean(1) - tri.reshape(-1, 3).mean(0)) > 0)

    with pytest.raises(NIfTIValidationError) as e:
        viewer_data.stl_bytes(None, ["pred_ET"])                # no vertices: nothing to print
    assert e.value.status_code == 404


@pytest.mark.skipif(not ready, reason=f"inference unavailable: {missing}")
def test_viewer_and_report_with_expert_mask():
    client = TestClient(app)
    study_store.clear()
    shape, aff = (160, 192, 96), lps_aniso_affine()
    expert = np.zeros(shape, np.uint8)
    g = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    r = np.sqrt((g[0] - 0.3) ** 2 + (g[1] - 0.1) ** 2 + g[2] ** 2)
    expert[r < 0.35] = 2; expert[r < 0.2] = 4; expert[r < 0.12] = 1
    try:
        up = _upload(client, phantom(shape, aff),
                     [("expert", ("Case01_seg.nii", _nifti_bytes(expert, aff), "application/octet-stream"))])
        assert up.status_code == 200, up.json()
        sid = up.json()["study_id"]
        assert up.json()["expert_mask"] is True
        assert client.post("/api/segmentation/run", json={"study_id": sid}).status_code == 200

        meta = client.get(f"/api/viewer/{sid}/meta").json()
        X, Y, Z = meta["canonical_shape"]
        assert sorted(meta["canonical_shape"]) == sorted(shape)
        assert meta["has_prediction"] and meta["has_expert"]
        assert set(meta["agreement"]) == {"WT", "TC", "ET"}
        assert meta["volumes"]["expert"]["WT"]["voxels"] == int((expert > 0).sum())
        # canonical axes point R, A, S
        A = np.array(meta["canonical_affine"])
        assert A[0, 0] > 0 and A[1, 1] > 0 and A[2, 2] > 0

        assert meta["has_uncertainty"] and meta["uncertainty"]["passes"] == 4
        unc = np.frombuffer(client.get(f"/api/viewer/{sid}/volume/uncert").content, np.uint8)
        assert unc.max() <= 4
        abl = client.post(f"/api/viewer/{sid}/ablation").json()
        assert set(abl["removed_sequence"]) == {"t1gd", "t1", "t2", "flair"} and "not clinical evidence" in abl["label"]

        for name in ("flair", "pred", "expert", "uncert"):
            vol = client.get(f"/api/viewer/{sid}/volume/{name}")
            assert vol.status_code == 200 and len(vol.content) == X * Y * Z
        # expert volume in canonical order matches the uploaded mask reoriented
        exp_canon = np.frombuffer(client.get(f"/api/viewer/{sid}/volume/expert").content, np.uint8).reshape((X, Y, Z), order="F")
        assert np.array_equal(exp_canon, nib.as_closest_canonical(nib.Nifti1Image(expert, aff)).get_fdata().astype(np.uint8))

        meshes = client.get(f"/api/viewer/{sid}/meshes").json()
        assert "brain" in meshes and meshes["brain"]["n_faces"] > 0
        assert any(k.startswith("expert_") for k in meshes)
        verts = np.frombuffer(__import__("base64").b64decode(meshes["brain"]["vertices"]), np.float32).reshape(-1, 3)
        world_corners = np.array([aff @ [i, j, k, 1] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])])[:, :3]
        assert (verts.min(0) >= world_corners.min(0) - 3).all() and (verts.max(0) <= world_corners.max(0) + 3).all()

        stl = client.get(f"/api/viewer/{sid}/stl", params={"layers": "brain,pred_ED"})
        assert stl.status_code == 200 and "attachment" in stl.headers["content-disposition"]
        n = int(np.frombuffer(stl.content[80:84], "<u4")[0])
        assert n == meshes["brain"]["n_faces"] + meshes["pred_ED"]["n_faces"] and len(stl.content) == 84 + 50 * n
        assert client.get(f"/api/viewer/{sid}/stl", params={"layers": "nope"}).status_code == 404

        page = client.get(f"/report/{sid}")
        assert page.status_code == 200
        for text in ("Not a diagnosis", "SHA-256", "Formula", "Provenance", "Limitations", "Agreement",
                     "How much to trust it", "not clinical evidence"):
            assert text in page.text
        pdf = client.get(f"/api/report/{sid}.pdf")
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
        assert client.get(f"/viewer/{sid}").status_code == 200
    finally:
        study_store.clear()
