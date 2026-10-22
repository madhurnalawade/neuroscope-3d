"""
Data for the 3D / linked-slice viewer and the report.

All viewer arrays are reoriented to the closest canonical RAS voxel order (axis 0 -> patient Right,
axis 1 -> Anterior, axis 2 -> Superior) so orientation labels are fixed. `canonical_affine` maps those
voxel indices to world millimetres; `original_affine` is the uploaded affine, so the client can convert
a canonical index back to the uploaded voxel index: ijk_orig = inv(original_affine) @ canonical_affine @ ijk.
"""

import base64
import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes

import app.core.paths  # noqa: F401
from app.core.exceptions import NIfTIValidationError
from app.services.study_store import ValidatedStudy, study_store

from brainseg.constants import DISCLAIMER, LABEL_NAMES, MODEL_INFO, PREPROCESSING, REGIONS
from brainseg.inference import UNCERTAINTY_FILENAME
from brainseg.labels import region_masks, region_volumes

SEQUENCE_KEYS = {"t1": "T1", "t1gd": "T1Gd", "t2": "T2", "flair": "FLAIR"}
EXPERT_FILENAME = "expert.nii.gz"
PREDICTION_FILENAME = "prediction.nii.gz"
CACHE_DIR = "viewer_cache"

MESH_STYLE = {
    "brain": {"label": "Brain surface", "color": "#d8d0c0", "opacity": 0.18},
    "NCR": {"label": "NCR (1) necrotic core", "color": "#d6452c", "opacity": 0.95},
    "ED": {"label": "ED (2) oedema", "color": "#78a870", "opacity": 0.45},
    "ET": {"label": "ET (4) enhancing", "color": "#e8b030", "opacity": 0.9},
}
LABEL_CODES = {"NCR": 1, "ED": 2, "ET": 4}


# ---------------------------------------------------------------- geometry

def _ref_image(study: ValidatedStudy) -> nib.Nifti1Image:
    return nib.load(study.modalities["T1Gd"]["path"], mmap=False)


def _canonical(study: ValidatedStudy):
    ref = _ref_image(study)
    ornt = io_orientation(ref.affine)
    canon_affine = ref.affine @ inv_ornt_aff(ornt, ref.shape[:3])
    return ref, ornt, canon_affine


def _reorient(arr: np.ndarray, ornt) -> np.ndarray:
    return np.ascontiguousarray(apply_orientation(arr, ornt))


def _cache(study: ValidatedStudy) -> Path:
    d = Path(study.study_dir) / CACHE_DIR
    d.mkdir(exist_ok=True)
    return d


def _labels(study: ValidatedStudy, which: str) -> Optional[np.ndarray]:
    path = Path(study.study_dir) / (PREDICTION_FILENAME if which == "pred" else EXPERT_FILENAME)
    if not path.exists():
        return None
    return np.asarray(nib.load(str(path), mmap=False).dataobj).astype(np.uint8)


# ---------------------------------------------------------------- expert mask

def validate_expert_mask(path: Path, ref: nib.Nifti1Image) -> np.ndarray:
    """Expert masks must be readable 3D maps on the study grid with BraTS codes 0/1/2/4 only."""
    try:
        img = nib.load(str(path), mmap=False)
        data = np.asarray(img.dataobj)
    except Exception as e:  # noqa: BLE001
        raise NIfTIValidationError(f"Expert mask is not a readable NIfTI file: {e}", "INVALID_EXPERT_MASK")
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise NIfTIValidationError(f"Expert mask must be 3D, got shape {data.shape}.", "INVALID_EXPERT_MASK")
    if data.shape != ref.shape[:3]:
        raise NIfTIValidationError(
            f"Expert mask shape {data.shape} does not match the MRI sequences {ref.shape[:3]}.",
            "EXPERT_MASK_SHAPE_MISMATCH",
        )
    # checked one property at a time so the message says what differs; the affine check covers the rest
    o_mask, o_ref = "".join(nib.aff2axcodes(img.affine)), "".join(nib.aff2axcodes(ref.affine))
    if o_mask != o_ref:
        raise NIfTIValidationError(
            f"Expert mask orientation {o_mask} does not match the MRI sequences ({o_ref}).",
            "EXPERT_MASK_ORIENTATION_MISMATCH", details={"expert": o_mask, "mri": o_ref})
    z_mask = [round(float(z), 4) for z in img.header.get_zooms()[:3]]
    z_ref = [round(float(z), 4) for z in ref.header.get_zooms()[:3]]
    if not np.allclose(z_mask, z_ref, rtol=1e-3, atol=1e-3):
        raise NIfTIValidationError(
            f"Expert mask voxel spacing {z_mask} mm does not match the MRI sequences ({z_ref} mm).",
            "EXPERT_MASK_SPACING_MISMATCH", details={"expert": z_mask, "mri": z_ref})
    if not np.allclose(img.affine, ref.affine, rtol=1e-3, atol=1e-3):
        shift = float(np.linalg.norm(img.affine[:3, 3] - ref.affine[:3, 3]))
        raise NIfTIValidationError(
            "Expert mask is not aligned with the MRI sequences (same grid size, orientation and spacing, but its "
            f"position or axes differ; origin offset {shift:.2f} mm). Resample it onto the MRI grid first.",
            "EXPERT_MASK_AFFINE_MISMATCH", details={"origin_offset_mm": round(shift, 3)})
    if not np.all(np.isfinite(data)) or not np.allclose(data, np.round(data)):
        raise NIfTIValidationError("Expert mask must contain integer label codes.", "INVALID_EXPERT_MASK")
    values = set(np.unique(data).astype(int).tolist())
    if not values <= {0, 1, 2, 4}:
        hint = ""
        if 3 in values and 4 not in values:
            hint = (" It looks like Medical Segmentation Decathlon coding (1 oedema, 2 non-enhancing, 3 enhancing);"
                    " convert it with scripts/convert_expert_labels.py first.")
        raise NIfTIValidationError(
            f"Expert mask contains codes {sorted(values)}; only BraTS codes 0, 1 (NCR), 2 (ED), 4 (ET) are accepted.{hint}",
            "INVALID_EXPERT_LABELS",
            details={"values": sorted(values)},
        )
    return data.astype(np.uint8)


def store_expert_mask(study: ValidatedStudy, upload_path: Path):
    ref = _ref_image(study)
    data = validate_expert_mask(upload_path, ref)
    pred = _labels(study, "pred")
    if pred is not None and pred.any() and np.array_equal(pred, data):
        # the model's own output is not ground truth; comparing it with itself would report a perfect score
        raise NIfTIValidationError(
            "This mask is voxel-for-voxel identical to the model's prediction for this study, so it cannot serve as "
            "an independent expert reference. Upload a mask drawn by an expert.",
            "EXPERT_MASK_IS_PREDICTION")
    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    nib.save(nib.Nifti1Image(data, ref.affine, header), str(Path(study.study_dir) / EXPERT_FILENAME))
    for f in _cache(study).glob("*"):
        if f.name.startswith(("expert", "meshes")):
            f.unlink()
    study.metadata["expert_mask"] = {"filename": upload_path.name.split("_", 1)[-1]}
    study_store.persist(study)


def remove_expert_mask(study: ValidatedStudy):
    (Path(study.study_dir) / EXPERT_FILENAME).unlink(missing_ok=True)
    for f in _cache(study).glob("*"):
        if f.name.startswith(("expert", "meshes")):
            f.unlink()
    study.metadata.pop("expert_mask", None)
    study_store.persist(study)


# ---------------------------------------------------------------- doctor's volume readings

READING_REGIONS = ("WT", "TC", "ET")
READING_MAX_ML = 2000.0                                 # larger than any adult brain; catches unit mistakes (mm³)


def store_readings(study: ValidatedStudy, volumes_ml: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and store volumes (mL) entered from a doctor's reading. WT, TC and/or ET."""
    vals = {}
    for r in READING_REGIONS:
        v = volumes_ml.get(r)
        if v is None:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v):
            raise NIfTIValidationError(f"{r} must be a number of millilitres.", "INVALID_READING")
        if v < 0 or v > READING_MAX_ML:
            raise NIfTIValidationError(
                f"{r} = {v:g} mL is outside 0 to {READING_MAX_ML:.0f} mL. Enter millilitres (1 mL = 1,000 mm³).",
                "INVALID_READING")
        vals[r] = float(v)
    if not vals:
        raise NIfTIValidationError("Enter at least one of WT, TC or ET in mL.", "INVALID_READING")
    # the regions are nested (ET inside TC inside WT), so the volumes must be too
    for inner, outer in (("TC", "WT"), ("ET", "TC"), ("ET", "WT")):
        if inner in vals and outer in vals and vals[inner] > vals[outer] + 1e-9:
            raise NIfTIValidationError(
                f"{inner} ({vals[inner]:g} mL) cannot be larger than {outer} ({vals[outer]:g} mL): ET is part of TC "
                "and TC is part of WT.", "INCONSISTENT_READING")
    study.metadata["doctor_readings"] = {
        "volumes_mL": vals,
        "entered_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    study_store.persist(study)
    return study.metadata["doctor_readings"]


def clear_readings(study: ValidatedStudy):
    study.metadata.pop("doctor_readings", None)
    study_store.persist(study)


def readings_comparison(model_volumes: Optional[Dict[str, Any]], readings: Optional[Dict[str, Any]]):
    """Model volume minus the entered volume, in mL and as a percentage of the entered volume. Volumes alone
    cannot give Dice, sensitivity or any overlap score: those need a voxel-wise reference."""
    if not model_volumes or not readings:
        return None
    rows = {}
    for r in READING_REGIONS:
        if r not in readings["volumes_mL"]:
            continue
        doc, model = float(readings["volumes_mL"][r]), float(model_volumes[r]["mL"])
        diff = model - doc
        rows[r] = {"model_mL": model, "doctor_mL": doc, "difference_mL": diff, "abs_difference_mL": abs(diff),
                   "percent_difference": None if doc == 0 else 100.0 * diff / doc}
    return rows


def expert_notes(pred: Optional[np.ndarray], expert: Optional[np.ndarray]):
    """Plain-language notes on empty regions, so undefined or trivial scores are not misread."""
    if expert is None:
        return []
    notes = []
    if not expert.any():
        notes.append("The expert mask is empty (no tumour labelled). Sensitivity and HD95 are undefined where the "
                     "expert marked nothing; Dice and IoU are 1.0 only where the model also predicted nothing.")
    r_e = region_masks(expert)
    r_p = region_masks(pred) if pred is not None else None
    for name in READING_REGIONS:
        if expert.any() and not r_e[name].any():
            notes.append(f"The expert marked no {name} voxels, so sensitivity for {name} is undefined.")
        if r_p is not None and not r_p[name].any():
            notes.append(f"The model predicted no {name} voxels, so precision for {name} is undefined.")
    return notes


# ---------------------------------------------------------------- volumes for slices

def volume_bytes(study: ValidatedStudy, name: str) -> bytes:
    """uint8 voxels in canonical RAS order, x fastest (index = x + X*(y + Y*z))."""
    cache = _cache(study) / f"{name}.u8"
    if cache.exists():
        return cache.read_bytes()
    _, ornt, _ = _canonical(study)
    if name in SEQUENCE_KEYS:
        data = np.asarray(nib.load(study.modalities[SEQUENCE_KEYS[name]]["path"], mmap=False).dataobj, dtype=np.float32)
        nz = data[data != 0]
        lo, hi = (np.percentile(nz, [0.5, 99.5]) if nz.size else (0.0, 1.0))
        scaled = np.clip((data - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        scaled[data == 0] = 0
        arr = scaled.astype(np.uint8)
    elif name == "uncert":
        path = Path(study.study_dir) / UNCERTAINTY_FILENAME
        if not path.exists():
            raise NIfTIValidationError("No uncertainty map for this study.", "NOT_AVAILABLE", status_code=404)
        arr = np.asarray(nib.load(str(path), mmap=False).dataobj).astype(np.uint8)
    elif name in ("pred", "expert"):
        arr = _labels(study, name)
        if arr is None:
            raise NIfTIValidationError(f"No {name} label map for this study.", "NOT_AVAILABLE", status_code=404)
    else:
        raise NIfTIValidationError(f"Unknown volume '{name}'.", "NOT_AVAILABLE", status_code=404)
    out = _reorient(arr, ornt).tobytes(order="F")
    cache.write_bytes(out)
    return out


# ---------------------------------------------------------------- meshes

def _surface(mask: np.ndarray, canon_affine: np.ndarray, sigma: float, step: int):
    """Closed iso-surface of a canonical-order mask as (world-mm vertices, faces), or None if too small.
    Shared by the viewer meshes and the print export (app/services/print_export.py)."""
    if mask.sum() < 8:
        return None
    field = gaussian_filter(np.pad(mask.astype(np.float32), 2), sigma)
    if field.max() < 0.5:
        return None
    verts, faces, _, _ = marching_cubes(field, level=0.5, step_size=step, allow_degenerate=False)
    verts -= 2  # undo padding
    return (np.c_[verts, np.ones(len(verts))] @ canon_affine.T)[:, :3], faces


def _mesh(mask: np.ndarray, canon_affine: np.ndarray, sigma: float, step: int) -> Optional[Dict[str, str]]:
    surf = _surface(mask, canon_affine, sigma, step)
    if surf is None:
        return None
    world, faces = surf[0].astype(np.float32), surf[1]
    return {
        "vertices": base64.b64encode(world.tobytes()).decode(),
        "faces": base64.b64encode(faces.astype(np.uint32).tobytes()).decode(),
        "n_vertices": int(len(world)),
        "n_faces": int(len(faces)),
    }


def brain_mask(study: ValidatedStudy, ornt) -> np.ndarray:
    """Voxels non-zero in any sequence (the inputs are skull-stripped), in canonical order."""
    seqs = [np.asarray(nib.load(study.modalities[m]["path"], mmap=False).dataobj) != 0 for m in SEQUENCE_KEYS.values()]
    return _reorient(np.logical_or.reduce(seqs).astype(np.uint8), ornt)


def meshes(study: ValidatedStudy) -> Dict[str, Any]:
    cache = _cache(study) / "meshes.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    _, ornt, canon_affine = _canonical(study)
    brain = brain_mask(study, ornt)
    out = {"brain": {**MESH_STYLE["brain"], "source": "brain", **(_mesh(brain, canon_affine, 1.5, 2) or {})}}
    for source in ("pred", "expert"):
        lab = _labels(study, source)
        if lab is None:
            continue
        lab = _reorient(lab, ornt)
        for name, code in LABEL_CODES.items():
            m = _mesh(lab == code, canon_affine, 0.6, 1)
            if m:
                style = dict(MESH_STYLE[name])
                if source == "expert":
                    style["label"] = "Expert " + style["label"]
                    style["opacity"] = 0.35
                out[f"{source}_{name}"] = {**style, "source": source, "code": code, **m}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


# ---------------------------------------------------------------- STL export (3D printing)

STL_HEADER = b"Voxels research prototype - model-predicted surfaces in mm - not for clinical use"[:80].ljust(80, b" ")
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])  # 50 bytes per triangle


def _triangles(mesh: Dict[str, Any]) -> np.ndarray:
    """(n, 3, 3) triangles of one cached mesh, wound so the normals point outward."""
    verts = np.frombuffer(base64.b64decode(mesh["vertices"]), np.float32).reshape(-1, 3)
    faces = np.frombuffer(base64.b64decode(mesh["faces"]), np.uint32).reshape(-1, 3)
    tri = verts[faces].astype(np.float64)
    # signed volume of a closed surface is positive when the winding is outward; the affine can mirror it
    if np.einsum("ij,ij->", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) < 0:
        tri = tri[:, ::-1]
    return tri


def stl_bytes(study: ValidatedStudy, layers) -> bytes:
    """Binary STL of the chosen 3D layers (keys of `meshes`), in millimetres, shifted so the model sits at the
    origin. Surfaces are closed (marching cubes on a padded mask), so each layer is a printable solid."""
    data = meshes(study)
    available = [k for k, v in data.items() if v.get("vertices")]
    layers = list(dict.fromkeys(layers)) or ["brain"]
    unknown = [k for k in layers if k not in available]
    if unknown:
        raise NIfTIValidationError(
            f"No 3D surface for {', '.join(unknown)}. Available: {', '.join(available) or 'none'}.",
            "NOT_AVAILABLE", details={"available": available}, status_code=404)
    tri = np.concatenate([_triangles(data[k]) for k in layers])
    tri -= tri.reshape(-1, 3).min(axis=0)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), STL_RECORD)
    rec["normal"], rec["v"] = normals, tri
    return STL_HEADER + np.uint32(len(tri)).tobytes() + rec.tobytes()


# ---------------------------------------------------------------- metrics + meta

def _agreement(pred: np.ndarray, expert: np.ndarray, spacing) -> Dict[str, Any]:
    from brainseg.metrics import case_metrics
    return case_metrics(pred, expert, spacing)


def meta(study: ValidatedStudy) -> Dict[str, Any]:
    ref, ornt, canon_affine = _canonical(study)
    canon_shape = [0, 0, 0]
    for in_axis, (out_axis, _) in enumerate(ornt):
        canon_shape[int(out_axis)] = int(ref.shape[in_axis])
    spacing = [float(z) for z in ref.header.get_zooms()[:3]]
    canon_spacing = [float(np.linalg.norm(canon_affine[:3, i])) for i in range(3)]
    pred, expert = _labels(study, "pred"), _labels(study, "expert")
    record = study.metadata.get("prediction", {})
    result: Dict[str, Any] = {
        "study_id": study.study_id,
        "created_at": study.created_at,
        "expires_at": study.expires_at,
        "canonical_shape": canon_shape,
        "canonical_affine": canon_affine.tolist(),
        "canonical_spacing_mm": canon_spacing,
        "original_shape": [int(s) for s in ref.shape[:3]],
        "original_affine": ref.affine.tolist(),
        "original_orientation": "".join(nib.aff2axcodes(ref.affine)),
        "spacing_mm": spacing,
        "voxel_volume_mm3": float(np.prod(spacing)),
        "sequences": {k: study.modalities[v]["filename"] for k, v in SEQUENCE_KEYS.items()},
        "label_names": {str(k): v for k, v in LABEL_NAMES.items()},
        "regions": {k: list(v) for k, v in REGIONS.items()},
        "has_prediction": pred is not None,
        "has_expert": expert is not None,
        "volumes": {"model": region_volumes(pred, spacing) if pred is not None else None,
                    "expert": region_volumes(expert, spacing) if expert is not None else None},
        "agreement": _agreement(pred, expert, spacing) if pred is not None and expert is not None else None,
        "expert_mask": study.metadata.get("expert_mask"),
        "expert_notes": expert_notes(pred, expert),
        "doctor_readings": study.metadata.get("doctor_readings"),
        "readings_comparison": readings_comparison(region_volumes(pred, spacing) if pred is not None else None,
                                                   study.metadata.get("doctor_readings")),
        "suitability": study.metadata.get("suitability", {}),
        "backup": study.metadata.get("backup"),
        "source_4d": study.metadata.get("source_4d"),
        "uncertainty": record.get("uncertainty"),
        "has_uncertainty": (Path(study.study_dir) / UNCERTAINTY_FILENAME).exists(),
        "ablation": study.metadata.get("ablation"),
        "provenance": record.get("provenance"),
        "model": MODEL_INFO,
        "preprocessing": PREPROCESSING,
        "disclaimer": DISCLAIMER,
    }
    if pred is not None and pred.any():
        c = np.argwhere(_reorient(pred, ornt) > 0).mean(axis=0)
        result["lesion_centroid_canonical"] = [int(round(v)) for v in c]
    return result


def get_study(study_id: str) -> ValidatedStudy:
    return study_store.get_validated_study(study_id)
