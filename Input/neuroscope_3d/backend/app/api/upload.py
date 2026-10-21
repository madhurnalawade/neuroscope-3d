"""
Upload and validation API endpoints for NIfTI MRI sequences.
Handles multipart/form-data streaming, individual slot validation,
and complete four-modality study verification.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile, status
from fastapi.responses import JSONResponse

import app.core.paths  # noqa: F401
from brainseg.inference import sha256

from app.core.exceptions import (
    DuplicateModalityError,
    FileTooLargeError,
    InvalidExtensionError,
    InvalidNIfTIFormatError,
    MissingModalityError,
    NIfTIValidationError,
)
from app.core.security import (
    cleanup_study_dir,
    create_temp_study_dir,
    generate_secure_id,
    sanitize_filename,
)
from app.models.schemas import (
    SingleFileValidationResponse,
    StudyValidationErrorResponse,
    StudyValidationSuccessResponse,
)
from app.services.modality_detector import (
    VALID_MODALITIES,
    detect_modality_from_filename,
    normalize_modality_name,
)
from app.services.nifti_validator import NIfTIValidator
from app.services.study_store import study_store
from app.services.suitability_check import check_study_suitability
from app.services.viewer_data import store_expert_mask

EXPERT_FIELDS = ("expert", "expert_mask", "mask", "seg", "label", "labels")

router = APIRouter(prefix="/api/upload", tags=["Upload & Validation"])
validator = NIfTIValidator()


@router.post(
    "/validate-single",
    response_model=SingleFileValidationResponse,
    summary="Validate an individual NIfTI file for immediate UI feedback",
)
async def validate_single_file_endpoint(
    file: UploadFile = File(...),
    modality: Optional[str] = Form(None),
):
    """
    Validates a single uploaded NIfTI file and returns dimensional and spatial metadata.
    Does not register a study for segmentation.
    """
    orig_name = file.filename or "unknown.nii.gz"
    clean_name = sanitize_filename(orig_name)
    norm_mod = normalize_modality_name(modality) if modality else detect_modality_from_filename(clean_name)

    temp_id = generate_secure_id(prefix="single_")
    temp_dir = create_temp_study_dir(temp_id)
    dest_path = temp_dir / clean_name

    try:
        total_bytes = validator.save_upload_stream(
            file_stream=file.file,
            dest_path=dest_path,
            filename=orig_name,
            modality=norm_mod,
        )

        info = validator.validate_single_file(
            file_path=dest_path,
            original_filename=orig_name,
            modality=norm_mod,
            file_size_bytes=total_bytes,
        )

        return SingleFileValidationResponse(
            valid=True,
            filename=orig_name,
            modality=norm_mod,
            file_size_mb=round(total_bytes / (1024 * 1024), 2),
            shape=list(info.shape),
            zooms=[round(z, 4) for z in info.zooms],
            orientation=list(info.orientation),
        )
    except NIfTIValidationError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "valid": False,
                "filename": orig_name,
                "modality": norm_mod,
                "file_size_mb": round((dest_path.stat().st_size if dest_path.exists() else 0) / (1024 * 1024), 2),
                "error_code": e.error_code,
                "message": e.message,
            },
        )
    finally:
        cleanup_study_dir(temp_dir)


@router.post(
    "/validate",
    summary="Validate complete 4-sequence MRI study (T1, T1Gd, T2, FLAIR)",
)
async def validate_mri_study(request: Request):
    """
    Accepts four MRI sequences mapped to T1, T1Gd, T2, and FLAIR.
    Supports:
    1. Named form fields: t1, t1gd (or t1ce), t2, flair
    2. Multi-file uploads: files + modalities
    3. Automatic modality detection when 4 files are uploaded
    """
    form = await request.form()
    study_id = generate_secure_id(prefix="brats_")
    study_dir = create_temp_study_dir(study_id)

    staged_files: Dict[str, Tuple[Path, str, int]] = {}
    modality_counts: Dict[str, int] = {}

    try:
        uploads_to_process: List[Tuple[Optional[str], Any]] = []

        field_map = {
            "t1": "T1",
            "t1gd": "T1Gd",
            "t1ce": "T1Gd",
            "t1c": "T1Gd",
            "t1post": "T1Gd",
            "t2": "T2",
            "flair": "FLAIR",
        }

        expert_upload = None
        for form_key, upload_item in form.multi_items():
            key_lower = form_key.lower()
            # Duck-typing check for UploadFile (supports both Starlette and FastAPI UploadFile)
            if hasattr(upload_item, "file") and hasattr(upload_item, "filename"):
                if key_lower in EXPERT_FIELDS:
                    expert_upload = upload_item
                elif key_lower in field_map:
                    uploads_to_process.append((field_map[key_lower], upload_item))
                elif key_lower in ("files", "file", "scans", "images"):
                    uploads_to_process.append((None, upload_item))
                else:
                    norm = normalize_modality_name(form_key)
                    uploads_to_process.append((norm, upload_item))

        # Check explicit modalities array if provided
        modalities_list = form.getlist("modalities")
        if not modalities_list and "modalities" in form:
            val = str(form.get("modalities"))
            modalities_list = [m.strip() for m in val.split(",") if m.strip()]

        if modalities_list:
            unassigned_idx = 0
            new_uploads = []
            for mod, ufile in uploads_to_process:
                if mod is None and unassigned_idx < len(modalities_list):
                    norm = normalize_modality_name(modalities_list[unassigned_idx])
                    new_uploads.append((norm, ufile))
                    unassigned_idx += 1
                else:
                    new_uploads.append((mod, ufile))
            uploads_to_process = new_uploads

        # If any upload still has no modality, attempt automatic recognition from filename
        resolved_uploads: List[Tuple[str, Any]] = []
        for mod, ufile in uploads_to_process:
            filename = getattr(ufile, "filename", "") or ""
            target_mod = mod or detect_modality_from_filename(filename)
            if not target_mod:
                raise NIfTIValidationError(
                    message=(
                        f"Could not automatically determine MRI sequence modality for file '{filename}'. "
                        f"Please manually select the modality (T1, T1Gd, T2, or FLAIR)."
                    ),
                    error_code="AMBIGUOUS_MODALITY",
                    details={"filename": filename},
                )
            resolved_uploads.append((target_mod, ufile))

        if len(resolved_uploads) < 4:
            provided_mods = [m for m, _ in resolved_uploads]
            missing = [m for m in VALID_MODALITIES if m not in provided_mods]
            raise MissingModalityError(missing_modalities=missing)

        # Check duplicate assignments
        for target_mod, ufile in resolved_uploads:
            modality_counts[target_mod] = modality_counts.get(target_mod, 0) + 1
            if modality_counts[target_mod] > 1:
                raise DuplicateModalityError(modality=target_mod, count=modality_counts[target_mod])

        # Check missing modalities
        missing = [m for m in VALID_MODALITIES if m not in modality_counts]
        if missing:
            raise MissingModalityError(missing_modalities=missing)

        # Stream uploads to disk in isolated temporary study directory
        for target_mod, ufile in resolved_uploads:
            orig_name = getattr(ufile, "filename", "") or f"{target_mod}.nii.gz"
            clean_name = sanitize_filename(orig_name)
            dest_filename = f"{target_mod}_{clean_name}"
            dest_path = study_dir / dest_filename

            written_bytes = validator.save_upload_stream(
                file_stream=ufile.file,
                dest_path=dest_path,
                filename=orig_name,
                modality=target_mod,
            )
            staged_files[target_mod] = (dest_path, orig_name, written_bytes)

        return _finalize_study(study_id, study_dir, staged_files, expert_upload)

    except NIfTIValidationError as e:
        cleanup_study_dir(study_dir)
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
    except Exception as e:
        cleanup_study_dir(study_dir)
        return _internal_error(e)


@router.post(
    "/validate-4d",
    summary="Validate ONE 4-channel NIfTI (X, Y, Z, 4): split into T1, T1Gd, T2 and FLAIR, then validate",
)
async def validate_4d_study(
    file: UploadFile = File(..., description="4D NIfTI with the four sequences stacked on the last axis"),
    channel_order: str = Form(
        "FLAIR,T1,T1Gd,T2",
        description="Sequence held in channels 0..3 (default: Medical Segmentation Decathlon order)",
    ),
    expert: Optional[UploadFile] = File(None, description="Optional expert mask, BraTS codes 0/1/2/4"),
):
    """Accepts a single 4-channel study file and treats each channel exactly like an uploaded sequence."""
    study_id = generate_secure_id(prefix="brats_")
    study_dir = create_temp_study_dir(study_id)
    try:
        order = [normalize_modality_name(c) for c in channel_order.split(",")]
        if len(order) != 4 or sorted(o or "" for o in order) != sorted(VALID_MODALITIES):
            raise NIfTIValidationError(
                f"channel_order must name T1, T1Gd, T2 and FLAIR exactly once (got '{channel_order}').",
                "INVALID_CHANNEL_ORDER",
            )
        orig = file.filename or "study.nii.gz"
        source = study_dir / f"source_{sanitize_filename(orig)}"
        validator.save_upload_stream(file.file, source, orig, modality="4D study")
        try:
            img = nib.load(str(source), mmap=False)
            shape = tuple(img.shape)
        except Exception as e:  # noqa: BLE001
            raise InvalidNIfTIFormatError(filename=orig, reason=str(e), modality="4D study")
        if len(shape) != 4 or shape[3] != 4:
            raise NIfTIValidationError(
                f"'{orig}' has shape {shape}. A single-file study must be 4D with exactly four channels "
                "(X, Y, Z, 4). Upload four 3D sequence files instead.",
                "NOT_FOUR_CHANNEL_STUDY",
                details={"shape": list(shape), "filename": orig},
            )
        try:
            data = np.asarray(img.dataobj)
        except Exception as e:  # noqa: BLE001
            raise InvalidNIfTIFormatError(filename=orig, reason=f"Could not read voxel data: {e}", modality="4D study")

        stem = sanitize_filename(orig)
        for ext in (".nii.gz", ".nii"):
            if stem.lower().endswith(ext):
                stem = stem[: -len(ext)]
        staged: Dict[str, Tuple[Path, str, int]] = {}
        for i, mod in enumerate(order):
            name = f"{stem}_ch{i}_{mod}.nii.gz"
            out = study_dir / f"{mod}_{name}"
            header = img.header.copy()
            channel = nib.Nifti1Image(np.ascontiguousarray(data[..., i]), img.affine, header)
            channel.set_qform(img.get_qform(), code=int(img.header["qform_code"]) or 1)
            channel.set_sform(img.get_sform(), code=int(img.header["sform_code"]) or 1)
            nib.save(channel, str(out))
            staged[mod] = (out, name, out.stat().st_size)
        source_info = {"file": orig, "sha256": sha256(source), "shape": list(shape),
                       "channel_order": order, "size_bytes": source.stat().st_size}
        del data, img
        source.unlink(missing_ok=True)
        return _finalize_study(study_id, study_dir, staged, expert, source_4d=source_info)

    except NIfTIValidationError as e:
        cleanup_study_dir(study_dir)
        return JSONResponse(status_code=e.status_code, content=e.to_dict())
    except Exception as e:
        cleanup_study_dir(study_dir)
        return _internal_error(e)


def _internal_error(e: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "valid": False,
            "error_code": "INTERNAL_VALIDATION_ERROR",
            "message": f"Unexpected server error during validation: {str(e)}",
            "details": {"error": str(e)},
            "ready_for_segmentation": False,
        },
    )


def _finalize_study(study_id, study_dir, staged_files, expert_upload=None, source_4d=None) -> Dict[str, Any]:
    """Shared tail of both upload paths: cross-sequence validation, suitability, registration, expert mask."""
    valid, validated_seqs, metadata, duration = validator.validate_study(staged_files)
    metadata["suitability"] = check_study_suitability(validated_seqs)
    if source_4d:
        metadata["source_4d"] = source_4d

    reg_modalities: Dict[str, Dict[str, Any]] = {}
    for mod, info in validated_seqs.items():
        reg_modalities[mod] = {
            "status": "valid",
            "filename": info.filename,
            "path": str(info.saved_path),
            "size_bytes": info.file_size_bytes,
            "shape": list(info.shape),
            "zooms": list(info.zooms),
        }

    study = study_store.register_study(
        study_id=study_id,
        study_dir=study_dir,
        modalities=reg_modalities,
        metadata=metadata,
    )

    # Optional expert mask (BraTS codes 0/1/2/4) on the same grid; an invalid mask rejects the study
    if expert_upload is not None:
        orig = getattr(expert_upload, "filename", "") or "expert.nii.gz"
        dest = study_dir / f"upload_{sanitize_filename(orig)}"
        try:
            validator.save_upload_stream(expert_upload.file, dest, orig, modality="expert")
            store_expert_mask(study, dest)
        except NIfTIValidationError:
            study_store.delete_study(study_id)
            raise
        finally:
            dest.unlink(missing_ok=True)

    return {
        "status": "success",
        "valid": True,
        "message": "All four MRI sequences validated successfully."
        + (f" {len(metadata['suitability']['warnings'])} suitability warning(s)."
           if metadata["suitability"]["warnings"] else ""),
        "warnings": metadata["suitability"]["warnings"],
        "study_id": study_id,
        "modalities": {m: "valid" for m in VALID_MODALITIES},
        "sequence_files": {m: v["filename"] for m, v in reg_modalities.items()},
        "metadata": metadata,
        "expert_mask": expert_upload is not None,
        "ready_for_segmentation": True,
    }
