"""Load the precomputed, clearly labelled backup case (only for when the live pipeline cannot be demonstrated)."""

import json
import shutil

from fastapi import APIRouter
from fastapi.responses import JSONResponse, RedirectResponse

from app.core.paths import REPO_ROOT
from app.core.security import create_temp_study_dir, generate_secure_id
from app.services.study_store import study_store

router = APIRouter(tags=["Backup case"])
BACKUP_DIR = REPO_ROOT / "backup_case"
MODALITY_OF = {"t1": "T1", "t1gd": "T1Gd", "t2": "T2", "flair": "FLAIR"}


def load_backup_study() -> str:
    info = json.loads((BACKUP_DIR / "backup.json").read_text(encoding="utf-8"))
    record = json.loads((BACKUP_DIR / "prediction_meta.json").read_text(encoding="utf-8"))
    study_id = generate_secure_id(prefix="backup_")
    study_dir = create_temp_study_dir(study_id)
    modalities = {}
    for seq, name in info["files"].items():
        shutil.copy2(BACKUP_DIR / name, study_dir / name)
        modalities[MODALITY_OF[seq]] = {"status": "valid", "filename": name, "path": str(study_dir / name),
                                        "shape": record["shape"], "zooms": record["spacing_mm"]}
    for name in ("prediction.nii.gz", "prediction_meta.json", "expert.nii.gz", "uncertainty_tta.nii.gz"):
        if (BACKUP_DIR / name).exists():
            shutil.copy2(BACKUP_DIR / name, study_dir / name)
    metadata = {"shape": record["shape"], "voxel_spacing_mm": record["spacing_mm"], "prediction": record,
                "backup": info, "suitability": {"warnings": [], "refusals": []}}
    study_store.register_study(study_id, study_dir, modalities, metadata)
    return study_id


@router.get("/backup", summary="Open the PRECOMPUTED BACKUP case in the viewer (not a live run)")
def open_backup():
    if not (BACKUP_DIR / "backup.json").exists():
        return JSONResponse(status_code=404, content={
            "status": "error", "error_code": "NO_BACKUP_CASE",
            "message": "No backup case prepared. Run: python scripts/make_backup_case.py demo_cases/BRATS_001"})
    return RedirectResponse(f"/viewer/{load_backup_study()}", status_code=303)
