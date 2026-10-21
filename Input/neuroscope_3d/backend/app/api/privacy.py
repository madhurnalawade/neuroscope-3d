"""Retention overview for the home page's Privacy and Security section: what is stored and when it is deleted.

GET  /api/privacy/retention       every stored study, without its ID or file names (so the page never
                                  reveals other people's uploads): short reference, kind, file count, size,
                                  expiry.
POST /api/privacy/retention/mine  the file names inside the studies whose full IDs the browser sends in the
                                  body (it only knows the IDs of studies it uploaded itself).
"""

import time
from pathlib import Path
from typing import List

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.study_store import SWEEP_INTERVAL_SECONDS, ValidatedStudy, study_store

router = APIRouter(prefix="/api/privacy", tags=["Privacy"])
MAX_IDS = 50


def _ref(study_id: str) -> str:
    """First 6 hex digits of the random part: enough to tell studies apart, useless for opening one."""
    return study_id.rsplit("_", 1)[-1][:6]


def _files(study: ValidatedStudy) -> List[Path]:
    d = Path(study.study_dir)
    return sorted((p for p in d.rglob("*") if p.is_file()), key=lambda p: str(p)) if d.is_dir() else []


def _summary(study: ValidatedStudy) -> dict:
    files = _files(study)
    return {
        "ref": _ref(study.study_id),
        "kind": "precomputed sample" if study.study_id.startswith("backup_") else "upload",
        "created_at": study.created_at,
        "expires_at": study.expires_at,
        "expires_ts": study.created_timestamp + study.ttl_seconds,
        "n_files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
    }


@router.get("/retention")
def retention():
    return {
        "now_ts": time.time(),
        "ttl_hours": round(study_store.ttl_seconds / 3600, 2),
        "sweep_minutes": round(SWEEP_INTERVAL_SECONDS / 60),
        "studies": [_summary(s) for s in study_store.list_studies()],
    }


class MineRequest(BaseModel):
    study_ids: List[str] = Field(default_factory=list, max_length=MAX_IDS)


@router.post("/retention/mine")
def retention_mine(req: MineRequest):
    live = {s.study_id: s for s in study_store.list_studies()}
    out = []
    for sid in dict.fromkeys(req.study_ids):
        s = live.get(sid)
        if not s:
            continue
        root = Path(s.study_dir)
        out.append({"ref": _ref(sid), "files": [{"name": p.relative_to(root).as_posix(), "bytes": p.stat().st_size}
                                                for p in _files(s)]})
    return {"studies": out, "live_ids": [sid for sid in req.study_ids if sid in live]}
