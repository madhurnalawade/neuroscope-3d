"""The retention overview shows when each study is deleted without exposing other studies' IDs or file names."""

import pytest
from starlette.testclient import TestClient

from app.api.backup import BACKUP_DIR
from app.main import app
from app.services.study_store import study_store

pytestmark = pytest.mark.skipif(not (BACKUP_DIR / "backup.json").exists(), reason="no backup case prepared")


def test_retention_lists_expiry_without_ids_or_names():
    client = TestClient(app)
    try:
        sid = client.get("/backup", follow_redirects=False).headers["location"].rsplit("/", 1)[-1]
        res = client.get("/api/privacy/retention")
        assert res.status_code == 200
        body = res.json()
        assert body["ttl_hours"] < 24
        (s,) = [x for x in body["studies"] if sid.split("_", 1)[1].startswith(x["ref"])]
        assert s["kind"] == "precomputed sample" and s["n_files"] > 0 and s["total_bytes"] > 0
        assert 0 < s["expires_ts"] - body["now_ts"] <= 24 * 3600
        assert sid not in res.text and ".nii" not in res.text

        mine = client.post("/api/privacy/retention/mine", json={"study_ids": [sid, "brats_" + "0" * 32]}).json()
        assert mine["live_ids"] == [sid]
        names = [f["name"] for f in mine["studies"][0]["files"]]
        assert "prediction.nii.gz" in names
    finally:
        study_store.clear()


def test_retention_mine_rejects_too_many_ids():
    res = TestClient(app).post("/api/privacy/retention/mine", json={"study_ids": ["x"] * 51})
    assert res.status_code == 422
