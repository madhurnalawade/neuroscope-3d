"""
Validated study repository for NeuroScope 3D.
Keeps studies in memory, mirrors each record to <study_dir>/study.json so studies survive a server
restart, and deletes every upload once it is older than the TTL (default 23 h, always < 24 h):
lazily on access, on startup, and from a background sweeper every 10 minutes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from app.core.exceptions import StudyExpiredError, StudyNotFoundError
from app.core.security import BASE_STUDY_DIR, cleanup_study_dir

DEFAULT_STUDY_TTL_SECONDS = float(os.environ.get("NEUROSCOPE_STUDY_TTL_SECONDS", 23 * 3600))
SWEEP_INTERVAL_SECONDS = 600
RECORD_FILENAME = "study.json"


@dataclass
class ValidatedStudy:
    """Represents a validated multi-sequence MRI study ready for segmentation."""
    study_id: str
    study_dir: Path
    modalities: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    validated: bool = True
    created_timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = DEFAULT_STUDY_TTL_SECONDS

    @property
    def created_at(self) -> str:
        return datetime.fromtimestamp(self.created_timestamp, tz=timezone.utc).isoformat()

    @property
    def expires_at(self) -> str:
        return datetime.fromtimestamp(self.created_timestamp + self.ttl_seconds, tz=timezone.utc).isoformat()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_timestamp) > self.ttl_seconds


class StudyStore:
    """Thread-safe store of validated studies."""

    def __init__(self, ttl_seconds: float = DEFAULT_STUDY_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._studies: Dict[str, ValidatedStudy] = {}
        self._lock = threading.Lock()
        self._sweeper: Optional[threading.Thread] = None

    def register_study(
        self,
        study_id: str,
        study_dir: Path,
        modalities: Dict[str, Dict[str, Any]],
        metadata: Dict[str, Any],
        ttl_seconds: Optional[float] = None,
    ) -> ValidatedStudy:
        """Register a successfully validated MRI study."""
        self.clean_expired_studies()
        study = ValidatedStudy(
            study_id=study_id,
            study_dir=study_dir,
            modalities=modalities,
            metadata=metadata,
            validated=True,
            created_timestamp=time.time(),
            ttl_seconds=ttl_seconds or self.ttl_seconds,
        )
        with self._lock:
            self._studies[study_id] = study
        self.persist(study)
        return study

    def persist(self, study: ValidatedStudy):
        """Write the study record next to its files so it can be reloaded after a restart."""
        record = {
            "study_id": study.study_id,
            "modalities": study.modalities,
            "metadata": study.metadata,
            "validated": study.validated,
            "created_timestamp": study.created_timestamp,
            "ttl_seconds": study.ttl_seconds,
        }
        try:
            (Path(study.study_dir) / RECORD_FILENAME).write_text(json.dumps(record), encoding="utf-8")
        except OSError:
            pass

    def get_validated_study(self, study_id: str) -> ValidatedStudy:
        """
        Retrieve validated study by study_id.
        Raises StudyNotFoundError if not found, or StudyExpiredError if expired.
        """
        with self._lock:
            study = self._studies.get(study_id)

        if not study:
            raise StudyNotFoundError(study_id=study_id)

        if study.is_expired:
            self.delete_study(study_id)
            raise StudyExpiredError(study_id=study_id)

        return study

    def is_valid_study(self, study_id: str) -> bool:
        """Check if study exists, is validated, and has not expired."""
        try:
            study = self.get_validated_study(study_id)
            return study.validated and not study.is_expired
        except Exception:
            return False

    def list_studies(self) -> List[ValidatedStudy]:
        """Unexpired studies, soonest deletion first (expired ones are deleted on the way)."""
        self.clean_expired_studies()
        with self._lock:
            studies = list(self._studies.values())
        return sorted(studies, key=lambda s: s.created_timestamp + s.ttl_seconds)

    def delete_study(self, study_id: str):
        """Remove a study and delete its temporary files from disk."""
        with self._lock:
            study = self._studies.pop(study_id, None)
        if study and study.study_dir:
            cleanup_study_dir(study.study_dir)

    def clean_expired_studies(self):
        """Clean up all studies that have exceeded their TTL."""
        expired_ids = []
        with self._lock:
            for s_id, study in list(self._studies.items()):
                if study.is_expired:
                    expired_ids.append((s_id, study.study_dir))
                    del self._studies[s_id]

        for _, s_dir in expired_ids:
            if s_dir:
                cleanup_study_dir(s_dir)

    def sweep_disk(self):
        """Delete expired or unregistered upload directories; reload unexpired ones not in memory."""
        if not BASE_STUDY_DIR.exists():
            return
        now = time.time()
        for d in BASE_STUDY_DIR.iterdir():
            if not d.is_dir():
                continue
            with self._lock:
                known = d.name in self._studies
            if known:
                continue
            rec_path = d / RECORD_FILENAME
            try:
                rec = json.loads(rec_path.read_text(encoding="utf-8"))
                study = ValidatedStudy(
                    study_id=rec["study_id"],
                    study_dir=d,
                    modalities=rec["modalities"],
                    metadata=rec["metadata"],
                    validated=rec.get("validated", True),
                    created_timestamp=float(rec["created_timestamp"]),
                    ttl_seconds=min(float(rec.get("ttl_seconds", self.ttl_seconds)), self.ttl_seconds),
                )
            except (OSError, ValueError, KeyError):
                # No usable record: an interrupted upload or foreign directory. Delete once past the TTL
                # (or after 1 h for half-written uploads) based on modification time.
                if now - d.stat().st_mtime > min(self.ttl_seconds, 3600):
                    cleanup_study_dir(d)
                continue
            if study.is_expired:
                cleanup_study_dir(d)
            else:
                with self._lock:
                    self._studies[study.study_id] = study

    def start_sweeper(self):
        """Run sweep_disk + clean_expired_studies now and then every SWEEP_INTERVAL_SECONDS."""
        if self._sweeper and self._sweeper.is_alive():
            return

        def loop():
            while True:
                try:
                    self.clean_expired_studies()
                    self.sweep_disk()
                except Exception:
                    pass
                time.sleep(SWEEP_INTERVAL_SECONDS)

        self._sweeper = threading.Thread(target=loop, name="study-sweeper", daemon=True)
        self._sweeper.start()

    def clear(self):
        """Clear all registered studies and their files (primarily for test tearDown)."""
        with self._lock:
            studies = list(self._studies.values())
            self._studies.clear()
        for study in studies:
            cleanup_study_dir(study.study_dir)


# Global singleton store
study_store = StudyStore()
