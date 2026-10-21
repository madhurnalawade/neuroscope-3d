"""Upload storage and automatic deletion, with a fake clock instead of waiting 24 hours."""
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from neuroscope.storage import (
    BackgroundSweeper, CloudSyncedRootError, UnsafeNameError, UnsafeRootError, UploadStore, UploadTooLargeError,
    cloud_sync_reasons, redirect_library_temp,
)
from neuroscope.storage.uploads import AUDIT_DIR, DELETION_LOG, MARKER_NAME

T0 = datetime(2026, 10, 17, 9, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self, now=T0):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def store(tmp_path, clock):
    return UploadStore(tmp_path / "uploads", clock=clock, sleep=lambda s: None)


def stored_case(store, status="stored"):
    cid = store.create_case()
    store.add_upload(cid, "T1", "BRATS_x_t1.nii.gz", b"fake scan bytes")
    store.set_status(cid, status)
    return cid


def test_fresh_case_is_kept(store, clock):
    cid = stored_case(store)
    clock.advance(hours=23, minutes=59)
    rep = store.sweep()
    assert rep.kept == [cid] and not rep.deleted and (store.root / cid).is_dir()


def test_expired_case_is_removed(store, clock):
    cid = stored_case(store)
    clock.advance(hours=24)
    rep = store.sweep()
    assert rep.deleted == [(cid, "expired")] and not (store.root / cid).exists()
    log = (store.root / AUDIT_DIR / DELETION_LOG).read_text().splitlines()
    assert json.loads(log[-1]) == {"time": "2026-10-18T09:00:00+00:00", "case_id": cid, "reason": "expired"}


def test_sweep_removes_only_expired(store, clock):
    old = stored_case(store)
    clock.advance(hours=20)
    new = stored_case(store)
    clock.advance(hours=5)
    rep = store.sweep()
    assert rep.deleted == [(old, "expired")] and rep.kept == [new]


def test_restart_still_deletes_expired(tmp_path, clock):
    first = UploadStore(tmp_path / "uploads", clock=clock)
    cid = stored_case(first)
    del first                                               # "process ends"; nothing in memory survives
    clock.advance(days=3)                                   # machine was off for days
    second = UploadStore(tmp_path / "uploads", clock=clock)
    sweeper = BackgroundSweeper(second, interval_seconds=3600)
    rep = sweeper.start()                                   # first sweep runs at startup
    sweeper.stop()
    assert rep.deleted == [(cid, "expired")] and not (tmp_path / "uploads" / cid).exists()


def test_dry_run_deletes_nothing(store, clock):
    cid = stored_case(store)
    clock.advance(hours=30)
    rep = store.sweep(dry_run=True)
    assert rep.would_delete == [(cid, "expired")] and not rep.deleted and (store.root / cid).is_dir()


@pytest.mark.parametrize("hours", [24.01, 25, 48, 0, -1])
def test_retention_above_24h_or_non_positive_rejected(tmp_path, hours):
    with pytest.raises(ValueError):
        UploadStore(tmp_path / "u", retention_hours=hours)


def test_shorter_retention_and_hard_cap_on_disk(store, clock):
    cid = stored_case(store)
    meta_path = store.root / cid / MARKER_NAME
    meta = json.loads(meta_path.read_text())
    meta["expires_at"] = (T0 + timedelta(days=30)).isoformat()      # tampered metadata
    meta_path.write_text(json.dumps(meta))
    clock.advance(hours=24)
    assert store.sweep().deleted == [(cid, "expired")]               # hard 24 h maximum still applies


@pytest.mark.parametrize("name", ["../../evil.nii.gz", "..\\evil.nii.gz", "/etc/x.nii.gz", "C:evil.nii.gz",
                                  "C:\\x\\y.nii.gz", "a/b.nii.gz", ".hidden.nii.gz", "~x.nii.gz", "x\x00.nii.gz",
                                  "x;rm -rf.nii.gz", "scan.dcm", ""])
def test_unsafe_upload_names_refused(store, name):
    cid = store.create_case()
    with pytest.raises(UnsafeNameError):
        store.add_upload(cid, "T1", name, b"x")
    assert sorted(p.name for p in (store.root / cid).iterdir()) == [MARKER_NAME]


def test_stored_name_never_uses_user_name(store):
    cid = store.create_case()
    p = store.add_upload(cid, "T1Gd", "Doe John 1970 (scan).nii.gz", io.BytesIO(b"abc"))
    assert p.name == "T1Gd.nii.gz" and p.parent == store.root / cid


@pytest.mark.parametrize("bad", ["..", "../x", "0" * 31, "g" * 32, "/" + "a" * 31, "A" * 32])
def test_path_traversal_case_ids_refused(store, bad):
    with pytest.raises(UnsafeNameError):
        store.delete_case(bad)


def test_oversized_upload_leaves_no_partial_file(tmp_path, clock):
    s = UploadStore(tmp_path / "u", clock=clock, max_upload_bytes=10)
    cid = s.create_case()
    with pytest.raises(UploadTooLargeError):
        s.add_upload(cid, "T1", "a_t1.nii.gz", io.BytesIO(b"x" * 100))
    assert not (s.root / cid / "T1.nii.gz").exists()


def _symlink_or_skip(target, link, is_dir=True):
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        if sys.platform == "win32" and is_dir:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))      # junction needs no admin rights
        else:
            pytest.skip("symbolic links not permitted on this system")


def test_symlinked_case_folder_never_followed(store, tmp_path, clock):
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "keep.txt").write_text("must survive")
    (outside / MARKER_NAME).write_text("{}")                   # even a marker inside the target must not matter
    link = store.root / ("a" * 32)
    _symlink_or_skip(outside, link)
    clock.advance(days=5)
    rep = store.sweep()
    assert (outside / "keep.txt").read_text() == "must survive"
    assert any(name == "a" * 32 for name, _ in rep.skipped)
    with pytest.raises(UnsafeNameError):
        store.delete_case("a" * 32)
    assert (outside / "keep.txt").exists()


def test_link_inside_case_removed_without_touching_target(store, tmp_path, clock):
    outside = tmp_path / "precious2"
    outside.mkdir()
    (outside / "keep.txt").write_text("must survive")
    cid = stored_case(store)
    _symlink_or_skip(outside, store.root / cid / "sneaky")
    clock.advance(hours=25)
    assert store.sweep().deleted == [(cid, "expired")]
    assert (outside / "keep.txt").read_text() == "must survive" and not (store.root / cid).exists()


def test_folders_without_marker_never_deleted(store, clock):
    foreign = store.root / ("b" * 32)
    foreign.mkdir()
    (foreign / "data.nii.gz").write_bytes(b"x")
    other = store.root / "notes"
    other.mkdir()
    clock.advance(days=10)
    rep = store.sweep()
    assert foreign.exists() and other.exists() and not rep.deleted
    assert {n for n, _ in rep.skipped} == {"b" * 32, "notes"}
    with pytest.raises(UnsafeNameError):
        store.delete_case("b" * 32)


def test_rejected_upload_removed_immediately(store):
    cid = stored_case(store, status="receiving")
    assert store.reject_case(cid, "NOT_GZIP") is True
    assert not (store.root / cid).exists()
    assert "rejected:NOT_GZIP" in (store.root / AUDIT_DIR / DELETION_LOG).read_text()


def test_incomplete_and_failed_cases_removed_at_next_sweep(store, clock):
    receiving = stored_case(store, status="receiving")
    failed = stored_case(store, status="failed")
    processing = stored_case(store, status="processing")
    assert dict(store.sweep().deleted) == {failed: "failed"}           # in-progress ones get a grace period
    clock.advance(hours=1)
    assert dict(store.sweep().deleted) == {receiving: "incomplete:receiving", processing: "incomplete:processing"}


def test_corrupt_metadata_removed(store):
    cid = stored_case(store)
    (store.root / cid / MARKER_NAME).write_text("{not json")
    assert store.sweep().deleted == [(cid, "corrupt-metadata")]


def test_delete_now(store):
    cid = stored_case(store)
    assert store.delete_case(cid) is True and not (store.root / cid).exists()
    assert store.delete_case(cid) is False


def test_case_tempdir_inside_case(store):
    cid = stored_case(store)
    with store.case_tempdir(cid) as d:
        assert store.root / cid in d.parents


def test_library_temp_redirected_into_uploads_root(store, clock, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)
    for var in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(var, os.environ.get(var, ""))
    scratch = redirect_library_temp(store)
    fd, name = tempfile.mkstemp()
    os.close(fd)
    assert Path(name).parent == scratch
    future = (clock.now + timedelta(days=2)).timestamp()
    os.utime(name, (future - 86400 * 3, future - 86400 * 3))
    clock.advance(days=2)
    assert any(r == "expired-temp" for _, r in store.sweep().deleted) and not Path(name).exists()


def test_locked_file_is_retried_and_reported(store, clock, monkeypatch):
    cid = stored_case(store)
    clock.advance(hours=25)
    real_unlink, calls = os.unlink, []

    def flaky(path, *a, **k):
        calls.append(path)
        if len(calls) <= 2:
            raise PermissionError("locked")
        return real_unlink(path, *a, **k)
    monkeypatch.setattr(os, "unlink", flaky)
    assert store.sweep().deleted == [(cid, "expired")] and len(calls) >= 3

    cid2 = stored_case(store)
    clock.advance(hours=25)
    monkeypatch.setattr(os, "unlink", lambda *a, **k: (_ for _ in ()).throw(PermissionError("locked")))
    rep = store.sweep()
    assert rep.failures and rep.failures[0][0] == cid2 and not rep.ok      # reported, not ignored


# ------------------------------------------------------------------ root guards

def test_refuses_dataset_root_and_dangerous_roots(tmp_path):
    ds = tmp_path / "Task01_BrainTumour"
    ds.mkdir()
    for root in (ds, tmp_path):                                       # equals / contains the dataset
        with pytest.raises(UnsafeRootError):
            UploadStore(root, dataset_root=ds)
    with pytest.raises(UnsafeRootError):
        UploadStore(ds / "uploads", dataset_root=ds)                   # inside the dataset
    with pytest.raises(UnsafeRootError):
        UploadStore("")
    with pytest.raises(UnsafeRootError):
        UploadStore(Path(Path.cwd().anchor))
    with pytest.raises(UnsafeRootError):
        UploadStore(Path.home())
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(UnsafeRootError):
        UploadStore(repo.parent, repo_root=repo)                       # contains the repository


@pytest.mark.parametrize("path", [
    "C:/Users/me/OneDrive/uploads", "C:/Users/me/OneDrive - Contoso/x", "/home/me/Dropbox/scans",
    "/Users/me/Library/Mobile Documents/com~apple~CloudDocs/u", "/Users/me/Library/CloudStorage/GoogleDrive-a/u",
    "G:/My Drive/uploads", "/home/me/Nextcloud/u", "/Users/me/iCloud Drive/u", "D:/Box/u",
])
def test_cloud_synced_paths_detected(path):
    assert cloud_sync_reasons(path)


@pytest.mark.parametrize("path", ["/srv/neuroscope/uploads", "D:/scratch/uploads", "/home/me/boxes/u",
                                  "/data/megabytes/u"])
def test_ordinary_paths_not_flagged(path, monkeypatch):
    for var in ("OneDrive", "OneDriveConsumer", "OneDriveCommercial"):
        monkeypatch.delenv(var, raising=False)
    assert not cloud_sync_reasons(path)


def test_cloud_synced_root_refused_without_override(tmp_path):
    root = tmp_path / "Dropbox" / "uploads"
    with pytest.raises(CloudSyncedRootError):
        UploadStore(root)
    assert UploadStore(root, allow_cloud_synced=True).root == root.resolve()
