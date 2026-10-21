"""The full core pipeline under the network guard, plus a scan of everything it logged.

Covers sequence assignment, validation, loading, hashing, upload storage, rejected-upload removal,
expiry cleanup, and the dev adapter + export on a synthetic Decathlon case. Passes only if no connection
or DNS lookup was attempted and no log line contains voxel values, array dumps, absolute paths, the
source folder name or raw (unsanitised) file names.
"""
import json
import logging
import re
from datetime import datetime, timedelta, timezone

from conftest import make_msd, save, volume
from neuroscope.dev.export import export_case, verify_export
from neuroscope.dev.msd import load_msd_case, read_dataset
from neuroscope.io import load_four_files
from neuroscope.security import netguard
from neuroscope.storage import UploadStore

SECRET_DIR = "PatientDoeJane_19700101"      # identifiable-looking folder name: must never be logged
MARKER_VALUE = 1234.5678                    # distinctive voxel value: must never be logged or summarised
SEQS = ("flair", "t1", "t1gd", "t2")


def _inputs(folder):
    files = {}
    for i, seq in enumerate(SEQS):
        v = volume(i)
        v[0, 0, 0], v[1, 1, 1], v[5, 5, 5] = 0, 5000, MARKER_VALUE
        files[seq] = save(folder / f"Doe^Jane_{seq}.nii.gz", v)
    return files


def test_full_pipeline_offline_with_clean_logs(tmp_path, caplog):
    caplog.set_level(logging.DEBUG)
    log = logging.getLogger("neuroscope.app")
    now = [datetime(2026, 10, 17, 9, tzinfo=timezone.utc)]

    with netguard.network_guard() as attempts:
        # 1. local four-file path
        files = _inputs(tmp_path / SECRET_DIR)
        res = load_four_files(unlabelled=files.values())
        assert res.ok, res.issues
        summary = json.dumps(res.case.summary())

        # 2. upload storage: store, re-validate the stored copies, hashes unchanged
        store = UploadStore(tmp_path / "uploads", clock=lambda: now[0], sleep=lambda s: None)
        cid = store.create_case()
        stored = {}
        for seq, path in files.items():
            with open(path, "rb") as fh:
                stored[seq] = store.add_upload(cid, seq, f"DoeJane_{seq}.nii.gz", fh)
        again = load_four_files(explicit=stored)
        assert again.ok and {k.lower(): v for k, v in again.case.sha256.items()} == \
            {k.lower(): v for k, v in res.case.sha256.items()}
        store.set_status(cid, "done")

        # 3. a rejected upload is logged by code and removed at once
        bad_id = store.create_case()
        bad = dict(stored)
        bad["t2"] = store.add_upload(bad_id, "T2", "broken_t2.nii.gz", b"not a nifti")
        rejected = load_four_files(explicit=bad)
        assert not rejected.ok
        for issue in rejected.issues:
            log.warning("upload rejected: %s", issue)
        store.reject_case(bad_id, rejected.issues[0].code)
        assert not (store.root / bad_id).exists()

        # 4. expiry
        now[0] += timedelta(hours=24, seconds=1)
        assert store.sweep().deleted == [(cid, "expired")]

        # 5. dev adapter + export round trip on a synthetic Decathlon case
        root = make_msd(tmp_path / "ds")
        info = read_dataset(root)
        case = load_msd_case(info, "BRATS_001")
        assert verify_export(case, export_case(info, case, "BRATS_001", tmp_path / "export")) == []

    assert attempts == [], attempts

    text = caplog.text
    assert "deleted case=" in text and "rejected:NOT_GZIP" in text      # the scan below is not vacuous
    forbidden = [str(tmp_path), str(tmp_path).replace("\\", "/"), SECRET_DIR, "Doe^Jane", "1234.5", "array(",
                 "ndarray"]
    for needle in forbidden:
        assert needle not in text, f"log contains forbidden content: {needle!r}"
        if needle not in (str(tmp_path), str(tmp_path).replace("\\", "/")):
            assert needle not in summary, f"summary contains forbidden content: {needle!r}"
    assert not re.search(r"(-?\d+\.\d+[,\s]+){8,}", text), "log contains a run of numbers (array dump?)"
    for issue in rejected.issues:
        assert str(tmp_path) not in issue.message and "\\" not in issue.message
