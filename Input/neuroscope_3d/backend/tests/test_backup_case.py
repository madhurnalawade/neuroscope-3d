"""The precomputed backup case must always be labelled as such in the viewer data and the report."""

import pytest
from starlette.testclient import TestClient

from app.api.backup import BACKUP_DIR
from app.main import app
from app.services.study_store import study_store

pytestmark = pytest.mark.skipif(not (BACKUP_DIR / "backup.json").exists(), reason="no backup case prepared")


def test_backup_case_is_labelled_everywhere():
    client = TestClient(app)
    try:
        res = client.get("/backup", follow_redirects=False)
        assert res.status_code == 303
        sid = res.headers["location"].rsplit("/", 1)[-1]
        assert sid.startswith("backup_")
        meta = client.get(f"/api/viewer/{sid}/meta").json()
        assert meta["backup"]["label"] == "PRECOMPUTED BACKUP CASE"
        assert meta["has_prediction"]
        page = client.get(f"/report/{sid}").text
        assert "PRECOMPUTED BACKUP CASE" in page and "NOT a live run" in page
        assert client.get(f"/api/report/{sid}.pdf").content[:4] == b"%PDF"
    finally:
        study_store.clear()
