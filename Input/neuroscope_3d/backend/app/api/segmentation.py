"""
Segmentation API endpoints for NeuroScope 3D.
Receives validated study requests and runs MONAI SegResNet 3D pipeline.
"""

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from app.api.upload import validate_mri_study
from app.core.exceptions import NIfTIValidationError, StudyNotFoundError
from app.models.schemas import SegmentationRunRequest, SegmentationRunResponse
from app.services.segmentation_engine import segmentation_pipeline
from app.services.study_store import study_store

router = APIRouter(prefix="/api/segmentation", tags=["Segmentation Pipeline"])


@router.post(
    "/predict",
    summary="One call: multipart t1, t1gd, t2, flair (+ optional expert) -> NIfTI label map (BraTS 1/2/4)",
    response_class=FileResponse,
)
async def predict_endpoint(request: Request):
    """Validates the four files, runs inference and returns prediction.nii.gz on the input grid and affine.
    The study ID, viewer and report URLs are returned in X-Study-Id / X-Viewer-Url / X-Report-Url headers."""
    validated = await validate_mri_study(request)
    if isinstance(validated, JSONResponse):
        return validated
    study_id = validated["study_id"]
    try:
        await run_in_threadpool(segmentation_pipeline.run_segmentation, study_id)
    except NIfTIValidationError as e:
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "study_id": study_id,
                                                      "message": f"Segmentation pipeline failed: {e}"})
    return FileResponse(
        str(segmentation_pipeline.prediction_path(study_id)), media_type="application/gzip",
        filename=f"{study_id}_prediction.nii.gz",
        headers={"X-Study-Id": study_id, "X-Viewer-Url": f"/viewer/{study_id}", "X-Report-Url": f"/report/{study_id}"},
    )


@router.post(
    "/run",
    response_model=SegmentationRunResponse,
    summary="Run MONAI SegResNet brain tumor segmentation on validated MRI study",
)
def run_segmentation_endpoint(request: SegmentationRunRequest):
    """
    Triggers segmentation inference.
    Strictly verifies on the server that study_id corresponds to a valid, unexpired study.
    Rejects invalid, missing, or expired studies.
    """
    study_id = request.study_id
    try:
        # Independent validation check
        result = segmentation_pipeline.run_segmentation(study_id)
        return result
    except NIfTIValidationError as e:
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Segmentation pipeline failed: {str(e)}",
                "study_id": study_id,
            },
        )


@router.get(
    "/prediction/{study_id}",
    summary="Download the NIfTI label map (BraTS codes 1/2/4, input grid and affine)",
)
async def download_prediction(study_id: str):
    try:
        path = segmentation_pipeline.prediction_path(study_id)
    except NIfTIValidationError as e:
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"status": "error", "error_code": "PREDICTION_NOT_FOUND",
                     "message": "Run /api/segmentation/run for this study first.", "study_id": study_id},
        )
    return FileResponse(str(path), media_type="application/gzip", filename=f"{study_id}_prediction.nii.gz")


@router.get(
    "/study/{study_id}",
    summary="Retrieve validation status and metadata for a study",
)
async def get_study_status(study_id: str):
    """Check if a study is registered and ready for segmentation."""
    try:
        study = study_store.get_validated_study(study_id)
        return {
            "status": "success",
            "study_id": study.study_id,
            "validated": study.validated,
            "created_at": study.created_at,
            "expires_at": study.expires_at,
            "metadata": study.metadata,
            "modalities": {
                k: {"status": v["status"], "filename": v["filename"], "shape": v["shape"]}
                for k, v in study.modalities.items()
            },
        }
    except NIfTIValidationError as e:
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
