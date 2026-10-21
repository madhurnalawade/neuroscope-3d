"""Local-only storage of user uploads with automatic deletion (at most 24 hours).

Layout under ONE uploads root:
    <root>/<32 hex case id>/.neuroscope_case.json   marker + metadata (created_at, expires_at, status)
    <root>/<case id>/T1.nii.gz ...                  uploaded scans, stored under fixed names (never user names)
    <root>/<case id>/tmp/                           per-case temporary files
    <root>/_scratch/                                process-wide temp dir for libraries (redirect_library_temp)
    <root>/_audit/deletions.log                     one line per deletion: time, case id, reason (no scan data)

Expiry is decided only from timestamps stored on disk, so a restart (or a machine that was off for days)
still deletes everything that has expired at the first sweep. Folders without the marker are never deleted.
This module is for NEW uploads only; it refuses to run on or around the development dataset.
"""
import contextlib
import json
import logging
import os
import re
import secrets
import stat
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path, PureWindowsPath
from typing import Callable, List, Optional, Tuple

from .guards import CloudSyncedRootError, check_uploads_root, cloud_sync_reasons

MARKER_NAME = ".neuroscope_case.json"
MARKER_TAG = "neuroscope-upload-case-v1"
SCRATCH_DIR = "_scratch"
SCRATCH_MARKER = ".neuroscope_scratch"
AUDIT_DIR = "_audit"
DELETION_LOG = "deletions.log"
MAX_RETENTION_HOURS = 24.0
DEFAULT_RETENTION_HOURS = 24.0
INCOMPLETE_GRACE = timedelta(minutes=30)   # 'receiving'/'processing' older than this = crashed or abandoned
MAX_UPLOAD_BYTES = 1_000_000_000
STATUSES = ("receiving", "stored", "processing", "done", "failed", "rejected")
REMOVE_NOW = ("failed", "rejected")
CASE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
ROLE_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
DERIVED_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}$")
UPLOAD_NAME_RE = re.compile(r"^[A-Za-z0-9 ._()+-]{1,255}$")
_REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
REPO_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("neuroscope.storage")


class UnsafeNameError(ValueError):
    """An uploaded file name or identifier was rejected (traversal, absolute path, odd characters...)."""


class UploadTooLargeError(ValueError):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def check_upload_name(name: str) -> str:
    """Validate a user-supplied file name and return its NIfTI extension. The name itself is never used on disk."""
    if not isinstance(name, str) or not name:
        raise UnsafeNameError("Empty file name.")
    if any(ord(ch) < 32 for ch in name) or "/" in name or "\\" in name:
        raise UnsafeNameError("File names may not contain path separators or control characters.")
    if name in (".", "..") or name.startswith(("~", ".")) or PureWindowsPath(name).drive or Path(name).is_absolute():
        raise UnsafeNameError("File names may not be paths, hidden names or start with '~' or '.'.")
    if not UPLOAD_NAME_RE.match(name):
        raise UnsafeNameError("File name contains characters outside A-Z a-z 0-9 space . _ - ( ) +.")
    low = name.lower()
    for ext in (".nii.gz", ".nii"):
        if low.endswith(ext):
            return ext
    raise UnsafeNameError("Only .nii and .nii.gz files are accepted.")


def _is_link(st: os.stat_result) -> bool:
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & _REPARSE)


def _private_dir(path: Path):
    path.mkdir(mode=0o700, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700)       # POSIX: owner only. Windows: only the read-only flag is affected.


def _open_new_private(path: Path):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    return os.fdopen(os.open(str(path), flags, 0o600), "wb")


@dataclass
class CaseInfo:
    case_id: str
    status: str
    created_at: Optional[datetime]
    expires_at: Optional[datetime]
    problem: Optional[str] = None        # set when the marker is unreadable


@dataclass
class SweepReport:
    dry_run: bool
    deleted: List[Tuple[str, str]] = field(default_factory=list)         # (case id, reason)
    would_delete: List[Tuple[str, str]] = field(default_factory=list)
    kept: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)          # (entry name, why it was not touched)
    failures: List[Tuple[str, str]] = field(default_factory=list)         # (case id, error class)

    @property
    def ok(self) -> bool:
        return not self.failures


class UploadStore:
    """All access to uploaded files goes through this class."""

    def __init__(self, uploads_root, *, retention_hours: float = DEFAULT_RETENTION_HOURS,
                 clock: Callable[[], datetime] = utcnow, dataset_root=None, repo_root=REPO_ROOT,
                 allow_cloud_synced: bool = False, incomplete_grace: timedelta = INCOMPLETE_GRACE,
                 max_upload_bytes: int = MAX_UPLOAD_BYTES, sleep: Callable[[float], None] = time.sleep,
                 create: bool = True):
        retention_hours = float(retention_hours)
        if not (0 < retention_hours <= MAX_RETENTION_HOURS):
            raise ValueError(f"Retention must be more than 0 and at most {MAX_RETENTION_HOURS:g} hours "
                             f"(got {retention_hours:g}).")
        self.root = check_uploads_root(uploads_root, dataset_root=dataset_root, repo_root=repo_root)
        reasons = cloud_sync_reasons(self.root)
        if reasons and not allow_cloud_synced:
            raise CloudSyncedRootError(
                "The uploads root appears to be inside a cloud-synced folder (" + "; ".join(reasons) + "). "
                "A sync client would copy uploaded scans to the cloud. Choose a local, unsynced folder, or pass "
                "the explicit override only if you have verified that this folder is NOT synced.")
        if reasons:
            log.warning("!!! UPLOADS ROOT MAY BE CLOUD-SYNCED (%s). Override given; scans may leave this machine. !!!",
                        "; ".join(reasons))
        self.retention = timedelta(hours=retention_hours)
        self.clock = clock
        self.incomplete_grace = incomplete_grace
        self.max_upload_bytes = max_upload_bytes
        self._sleep = sleep
        self._lock = threading.RLock()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            _private_dir(self.root)
            _private_dir(self.root / AUDIT_DIR)

    # ------------------------------------------------------------------ paths

    def case_path(self, case_id: str) -> Path:
        """Path of a case folder, refusing anything that is not a direct, real (non-link) child of the root."""
        if not isinstance(case_id, str) or not CASE_ID_RE.match(case_id):
            raise UnsafeNameError("Invalid case id.")
        path = self.root / case_id
        if path.exists() or path.is_symlink():
            st = os.lstat(path)
            if _is_link(st):
                raise UnsafeNameError("Case folder is a link; refusing to use it.")
            if path.resolve().parent != self.root:
                raise UnsafeNameError("Case folder resolves outside the uploads root.")
        return path

    def _marker(self, case_id: str) -> Path:
        return self.case_path(case_id) / MARKER_NAME

    # ------------------------------------------------------------------ metadata

    def _write_meta(self, case_dir: Path, meta: dict):
        tmp = case_dir / (MARKER_NAME + ".tmp")
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        with _open_new_private(tmp) as f:
            f.write(json.dumps(meta, indent=1).encode("utf-8"))
        os.replace(tmp, case_dir / MARKER_NAME)

    def read_meta(self, case_id: str) -> dict:
        meta = json.loads(self._marker(case_id).read_text(encoding="utf-8"))
        if meta.get("marker") != MARKER_TAG or meta.get("case_id") != case_id:
            raise ValueError("marker does not belong to this case")
        return meta

    def info(self, case_id: str) -> CaseInfo:
        try:
            meta = self.read_meta(case_id)
            return CaseInfo(case_id, meta["status"], datetime.fromisoformat(meta["created_at"]),
                            datetime.fromisoformat(meta["expires_at"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return CaseInfo(case_id, "unknown", None, None, problem=type(exc).__name__)

    def set_status(self, case_id: str, status: str):
        if status not in STATUSES:
            raise ValueError(f"Unknown status {status!r}.")
        with self._lock:
            meta = self.read_meta(case_id)
            meta["status"] = status
            meta["updated_at"] = _iso(self.clock())
            self._write_meta(self.case_path(case_id), meta)

    # ------------------------------------------------------------------ writing

    def create_case(self) -> str:
        """Create a new case folder with a random id and its marker (status 'receiving')."""
        now = self.clock()
        with self._lock:
            for _ in range(5):
                case_id = secrets.token_hex(16)
                path = self.root / case_id
                try:
                    path.mkdir(mode=0o700)
                    break
                except FileExistsError:
                    continue
            else:
                raise RuntimeError("Could not allocate a case id.")
            with contextlib.suppress(OSError):
                os.chmod(path, 0o700)
            self._write_meta(path, {
                "marker": MARKER_TAG, "case_id": case_id, "status": "receiving",
                "created_at": _iso(now), "expires_at": _iso(now + self.retention),
                "retention_hours": self.retention.total_seconds() / 3600, "files": [],
            })
        log.info("created case=%s expires_at=%s", case_id, _iso(now + self.retention))
        return case_id

    def add_upload(self, case_id: str, role: str, original_name: str, source) -> Path:
        """Store one uploaded scan as '<role><ext>' (e.g. 'T1Gd.nii.gz'). ``source`` is bytes or a binary
        file object. The user's file name is validated but not used on disk."""
        ext = check_upload_name(original_name)
        if not ROLE_RE.match(role or ""):
            raise UnsafeNameError("Invalid role.")
        return self._write_file(case_id, role + ext, source)

    def write_derived(self, case_id: str, name: str, source) -> Path:
        """Store a file derived from the upload (mask, mesh, thumbnail, report) inside the case folder."""
        if not DERIVED_RE.match(name or "") or ".." in name:
            raise UnsafeNameError("Invalid derived file name.")
        return self._write_file(case_id, name, source)

    def _write_file(self, case_id: str, name: str, source) -> Path:
        case_dir = self.case_path(case_id)
        if not (case_dir / MARKER_NAME).is_file():
            raise UnsafeNameError("Unknown case.")
        dest = case_dir / name
        written = 0
        try:
            with _open_new_private(dest) as out:
                chunks = [source] if isinstance(source, (bytes, bytearray)) else iter(lambda: source.read(1 << 20), b"")
                for chunk in chunks:
                    written += len(chunk)
                    if written > self.max_upload_bytes:
                        raise UploadTooLargeError(f"Upload exceeds {self.max_upload_bytes / 1e9:.2f} GB.")
                    out.write(chunk)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(dest)                     # never keep a partial file
            raise
        with self._lock:
            meta = self.read_meta(case_id)
            meta["files"] = sorted(set(meta.get("files", [])) | {name})
            self._write_meta(case_dir, meta)
        return dest

    @contextlib.contextmanager
    def case_tempdir(self, case_id: str):
        """Temporary directory inside the case folder, removed on exit and covered by the case's expiry."""
        base = self.case_path(case_id) / "tmp"
        _private_dir(base)
        with tempfile.TemporaryDirectory(dir=str(base)) as d:
            yield Path(d)

    # ------------------------------------------------------------------ deletion

    def delete_case(self, case_id: str, reason: str = "requested") -> bool:
        """Delete one case folder now (for the API: 'delete my data'). Returns True if it was removed."""
        with self._lock:
            path = self.case_path(case_id)
            if not path.exists():
                return False
            if not (path / MARKER_NAME).is_file():
                raise UnsafeNameError("Folder has no NeuroScope marker; refusing to delete it.")
            self._remove_tree(path)
            self._log_deletion(case_id, reason)
            return True

    def reject_case(self, case_id: str, error_code: str = "validation") -> bool:
        """Remove a rejected upload immediately, as soon as the rejection is returned."""
        safe_code = re.sub(r"[^A-Z0-9_]", "", str(error_code).upper())[:40] or "VALIDATION"
        return self.delete_case(case_id, reason=f"rejected:{safe_code}")

    def _decide(self, case_id: str, now: datetime) -> Optional[str]:
        """Reason to delete this case now, or None to keep it."""
        try:
            meta = self.read_meta(case_id)
            created = datetime.fromisoformat(meta["created_at"])
            expires = min(datetime.fromisoformat(meta["expires_at"]), created + timedelta(hours=MAX_RETENTION_HOURS))
            status = meta.get("status")
        except (OSError, ValueError, KeyError, TypeError):
            return "corrupt-metadata"
        if created.tzinfo is None or expires.tzinfo is None:
            return "corrupt-metadata"
        if created > now + timedelta(minutes=5):
            return "created-in-future"
        if now >= expires:
            return "expired"
        if status in REMOVE_NOW:
            return status
        if status in ("receiving", "processing") and now - created > self.incomplete_grace:
            return f"incomplete:{status}"
        if status not in STATUSES:
            return "unknown-status"
        return None

    def sweep(self, dry_run: bool = False) -> SweepReport:
        """Remove every expired, rejected, failed, abandoned or corrupt case folder. Never touches folders
        without the marker, links, or anything outside the uploads root."""
        report = SweepReport(dry_run=dry_run)
        now = self.clock()
        with self._lock:
            if not self.root.is_dir():
                return report
            for entry in sorted(os.scandir(self.root), key=lambda e: e.name):
                name = entry.name
                if name in (AUDIT_DIR, SCRATCH_DIR):
                    continue
                st = entry.stat(follow_symlinks=False)
                if _is_link(st):
                    report.skipped.append((name, "link (not followed, not deleted)"))
                    continue
                if not stat.S_ISDIR(st.st_mode) or not CASE_ID_RE.match(name):
                    report.skipped.append((name, "not a NeuroScope case folder"))
                    continue
                if not (Path(entry.path) / MARKER_NAME).is_file():
                    report.skipped.append((name, "no marker file"))
                    continue
                reason = self._decide(name, now)
                if reason is None:
                    report.kept.append(name)
                elif dry_run:
                    report.would_delete.append((name, reason))
                else:
                    try:
                        self._remove_tree(Path(entry.path))
                        self._log_deletion(name, reason)
                        report.deleted.append((name, reason))
                    except OSError as exc:
                        report.failures.append((name, type(exc).__name__))
                        log.error("FAILED to delete case=%s reason=%s error=%s; will retry at next sweep",
                                  name, reason, type(exc).__name__)
            self._sweep_scratch(now, report, dry_run)
        return report

    def _sweep_scratch(self, now: datetime, report: SweepReport, dry_run: bool):
        scratch = self.root / SCRATCH_DIR
        if not (scratch / SCRATCH_MARKER).is_file() or _is_link(os.lstat(scratch)):
            return
        for entry in os.scandir(scratch):
            if entry.name == SCRATCH_MARKER:
                continue
            st = entry.stat(follow_symlinks=False)
            age = now - datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
            if age < self.retention:
                continue
            label = f"{SCRATCH_DIR}/{entry.name[:40]}"
            if dry_run:
                report.would_delete.append((label, "expired-temp"))
                continue
            try:
                self._remove_path(Path(entry.path), st)
                self._log_deletion(label, "expired-temp")
                report.deleted.append((label, "expired-temp"))
            except OSError as exc:
                report.failures.append((label, type(exc).__name__))

    def _remove_tree(self, path: Path):
        st = os.lstat(path)
        if _is_link(st) or path.resolve().parent != self.root:
            raise PermissionError("refusing to delete outside the uploads root or through a link")
        self._remove_path(path, st)

    def _remove_path(self, path: Path, st: os.stat_result):
        if _is_link(st):
            self._retry(_unlink_link, path)                 # remove the link itself, never its target
        elif stat.S_ISDIR(st.st_mode):
            for child in os.scandir(path):
                self._remove_path(Path(child.path), child.stat(follow_symlinks=False))
            self._retry(os.rmdir, path)
        else:
            self._retry(os.unlink, path)

    def _retry(self, fn, path: Path, attempts: int = 5):
        delay = 0.1
        for i in range(attempts):
            try:
                fn(path)
                return
            except FileNotFoundError:
                return
            except PermissionError:
                if i == attempts - 1:
                    raise
                with contextlib.suppress(OSError):          # read-only flag (Windows) blocks deletion
                    os.chmod(path, stat.S_IWRITE | stat.S_IREAD | (stat.S_IEXEC if path.is_dir() else 0))
            except OSError:
                if i == attempts - 1:
                    raise
            self._sleep(delay)                              # locked file (e.g. open in another program)
            delay *= 3

    def _log_deletion(self, case_id: str, reason: str):
        line = json.dumps({"time": _iso(self.clock()), "case_id": case_id, "reason": reason})
        try:
            with open(self.root / AUDIT_DIR / DELETION_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            log.error("could not write deletion log (%s)", type(exc).__name__)
        log.info("deleted case=%s reason=%s", case_id, reason)

    def list_cases(self) -> List[CaseInfo]:
        out = []
        for entry in sorted(os.scandir(self.root), key=lambda e: e.name):
            if CASE_ID_RE.match(entry.name) and not _is_link(entry.stat(follow_symlinks=False)):
                out.append(self.info(entry.name))
        return out


def _unlink_link(path: Path):
    try:
        os.unlink(path)
    except (IsADirectoryError, PermissionError):
        if sys.platform == "win32":
            os.rmdir(path)                                  # directory symlink or junction on Windows
        else:
            raise


class BackgroundSweeper:
    """Runs ``store.sweep()`` once at start (synchronously) and then every ``interval_seconds`` in a daemon
    thread. Deletion does not depend on this thread: expiry is on disk and the next start sweeps again."""

    def __init__(self, store: UploadStore, interval_seconds: float = 600.0):
        self.store = store
        self.interval = interval_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_report: Optional[SweepReport] = None

    def _run_once(self):
        try:
            self.last_report = self.store.sweep()
            if self.last_report.failures:
                log.error("sweep left %d case(s) undeleted; see errors above", len(self.last_report.failures))
        except Exception as exc:  # noqa: BLE001 - keep the thread alive, but report loudly
            log.error("upload sweep failed (%s)", type(exc).__name__)

    def start(self) -> SweepReport:
        self._run_once()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, name="neuroscope-upload-sweeper", daemon=True)
            self._thread.start()
        return self.last_report

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._run_once()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


def redirect_library_temp(store: UploadStore) -> Path:
    """Make Python's tempfile module (and child processes via TMPDIR/TEMP/TMP) use <root>/_scratch, so
    library temp files are covered by the same cleanup. Returns the scratch path."""
    scratch = store.root / SCRATCH_DIR
    _private_dir(scratch)
    marker = scratch / SCRATCH_MARKER
    if not marker.exists():
        marker.write_text(MARKER_TAG + "\n", encoding="utf-8")
    tempfile.tempdir = str(scratch)
    for var in ("TMPDIR", "TEMP", "TMP"):
        os.environ[var] = str(scratch)
    return scratch
