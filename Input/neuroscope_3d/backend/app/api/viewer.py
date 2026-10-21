"""Viewer, expert-mask and report endpoints."""

from pathlib import Path

from typing import Optional

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from app.core.exceptions import NIfTIValidationError
from app.core.security import sanitize_filename
from app.services import report, viewer_data
from app.services.nifti_validator import NIfTIValidator

router = APIRouter(tags=["Viewer & Report"])
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _error(e: NIfTIValidationError):
    return JSONResponse(status_code=e.status_code, content=e.to_dict())


@router.get("/viewer/{study_id}", response_class=HTMLResponse, include_in_schema=False)
async def viewer_page(study_id: str):
    return FileResponse(str(STATIC_DIR / "viewer.html"))


@router.get("/api/viewer/{study_id}/meta", summary="Geometry, volumes, agreement and provenance for the viewer")
def viewer_meta(study_id: str):
    try:
        return viewer_data.meta(viewer_data.get_study(study_id))
    except NIfTIValidationError as e:
        return _error(e)


@router.get("/api/viewer/{study_id}/volume/{name}", summary="uint8 volume (t1|t1gd|t2|flair|pred|expert), RAS, x fastest")
def viewer_volume(study_id: str, name: str):
    try:
        data = viewer_data.volume_bytes(viewer_data.get_study(study_id), name)
    except NIfTIValidationError as e:
        return _error(e)
    return Response(content=data, media_type="application/octet-stream")


@router.get("/api/viewer/{study_id}/meshes", summary="Brain and region surface meshes in world millimetres")
def viewer_meshes(study_id: str):
    try:
        return viewer_data.meshes(viewer_data.get_study(study_id))
    except NIfTIValidationError as e:
        return _error(e)


@router.get("/api/viewer/{study_id}/stl",
            summary="Binary STL of 3D layers for printing (comma-separated mesh keys, default brain), in mm")
def viewer_stl(study_id: str, layers: str = "brain"):
    keys = [k.strip() for k in layers.split(",") if k.strip()]
    try:
        stl = viewer_data.stl_bytes(viewer_data.get_study(study_id), keys)
    except NIfTIValidationError as e:
        return _error(e)
    name = "_".join(keys or ["brain"]) if len(keys) <= 3 else "layers"
    return Response(stl, media_type="model/stl",
                    headers={"Content-Disposition": f'attachment; filename="voxels_{study_id}_{name}.stl"'})


@router.post("/api/viewer/{study_id}/expert", summary="Attach an optional expert mask (BraTS codes 0/1/2/4)")
def upload_expert_mask(study_id: str, file: UploadFile = File(...)):
    try:
        study = viewer_data.get_study(study_id)
        name = sanitize_filename(file.filename or "expert.nii.gz")
        dest = Path(study.study_dir) / f"upload_{name}"
        NIfTIValidator().save_upload_stream(file.file, dest, file.filename or name, modality="expert")
        try:
            viewer_data.store_expert_mask(study, dest)
        finally:
            dest.unlink(missing_ok=True)
        return {"status": "success", "study_id": study_id, "message": "Expert mask attached."}
    except NIfTIValidationError as e:
        return _error(e)


@router.delete("/api/viewer/{study_id}/expert", summary="Remove the expert mask from this study")
def delete_expert_mask(study_id: str):
    try:
        viewer_data.remove_expert_mask(viewer_data.get_study(study_id))
        return {"status": "success", "study_id": study_id, "message": "Expert mask removed."}
    except NIfTIValidationError as e:
        return _error(e)


class Readings(BaseModel):
    WT: Optional[float] = None
    TC: Optional[float] = None
    ET: Optional[float] = None


@router.post("/api/viewer/{study_id}/readings",
             summary="Doctor's volume readings in mL (WT/TC/ET); compared by volume difference only, never Dice")
def save_readings(study_id: str, body: Readings):
    try:
        study = viewer_data.get_study(study_id)
        return {"status": "success", "study_id": study_id,
                "readings": viewer_data.store_readings(study, body.model_dump())}
    except NIfTIValidationError as e:
        return _error(e)


@router.delete("/api/viewer/{study_id}/readings", summary="Remove the doctor's volume readings from this study")
def delete_readings(study_id: str):
    try:
        viewer_data.clear_readings(viewer_data.get_study(study_id))
        return {"status": "success", "study_id": study_id}
    except NIfTIValidationError as e:
        return _error(e)


@router.post("/api/viewer/{study_id}/ablation",
             summary="Modality ablation: re-run with each sequence zeroed (not clinical evidence); cached per study")
def run_ablation(study_id: str):
    from app.services.segmentation_engine import _GPU_LOCK
    from app.services.study_store import study_store
    from brainseg.inference import load_sequences, modality_ablation

    try:
        study = _require_prediction(study_id)
    except NIfTIValidationError as e:
        return _error(e)
    if not study.metadata.get("ablation"):
        files = {k: study.modalities[v]["path"] for k, v in viewer_data.SEQUENCE_KEYS.items()}
        stack, ref = load_sequences(files)
        with _GPU_LOCK:
            study.metadata["ablation"] = modality_ablation(stack, ref.affine, ref.header.get_zooms()[:3])
        study_store.persist(study)
    return study.metadata["ablation"]


def _require_prediction(study_id: str):
    study = viewer_data.get_study(study_id)
    if viewer_data._labels(study, "pred") is None:
        raise NIfTIValidationError("Run segmentation for this study before generating a report.",
                                   "PREDICTION_NOT_FOUND", status_code=404)
    return study


@router.get("/report/{study_id}", response_class=HTMLResponse, summary="Printable HTML report")
def report_page(study_id: str):
    try:
        return HTMLResponse(report.render_html(_require_prediction(study_id)))
    except NIfTIValidationError as e:
        return _error(e)


@router.get("/api/report/{study_id}.pdf", summary="PDF report")
def report_pdf(study_id: str):
    try:
        pdf = report.render_pdf(_require_prediction(study_id))
    except NIfTIValidationError as e:
        return _error(e)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="voxels_report_{study_id}.pdf"'})
