"""
MONAI SegResNet Brain MRI Segmentation Engine for NeuroScope 3D.
Delegates inference to the shared `brainseg` pipeline (repository root), which loads the validated
sequences in protocol order [T1Gd, T1, T2, FLAIR], restores the prediction to the input grid and
affine, and writes a NIfTI label map in BraTS codes (1 = NCR, 2 = ED, 4 = ET).
"""

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import app.core.paths  # noqa: F401  (makes brainseg importable)
from app.core.exceptions import NIfTIValidationError
from app.services.study_store import study_store

from brainseg.constants import CHECKPOINT, DISCLAIMER, MODEL_INFO, MODEL_INPUT_ORDER

PROTOCOL_CHANNEL_ORDER = ["T1Gd", "T1", "T2", "FLAIR"]
# Upload module modality names -> shared brainseg sequence names
MODALITY_TO_SEQUENCE = {"T1": "t1", "T1Gd": "t1gd", "T2": "t2", "FLAIR": "flair"}
PREDICTION_FILENAME = "prediction.nii.gz"
METADATA_FILENAME = "prediction_meta.json"
_GPU_LOCK = threading.Lock()


class SegmentationEngine:
    """
    Executes 3D brain tumor segmentation on validated MRI studies.
    Strictly verifies study registration and checks segmentation dependencies.
    """

    def __init__(self, weights_path: Path = CHECKPOINT):
        self.weights_path = Path(weights_path)

    def check_dependencies(self) -> Tuple[bool, List[str]]:
        """
        Check for required inference dependencies and pre-trained model weights.
        Returns (is_ready, list_of_missing_components).
        """
        missing = []
        try:
            import torch  # noqa: F401
        except ImportError:
            missing.append("PyTorch (torch)")

        try:
            import monai  # noqa: F401
        except ImportError:
            missing.append("MONAI (monai)")

        if not self.weights_path.exists():
            missing.append(f"Model weights ({self.weights_path.name} at {self.weights_path})")

        return (len(missing) == 0), missing

    def prediction_path(self, study_id: str) -> Path:
        return study_store.get_validated_study(study_id).study_dir / PREDICTION_FILENAME

    def run_segmentation(self, study_id: str) -> Dict[str, Any]:
        """
        Execute segmentation pipeline on a validated study.
        1. Verifies study exists, is validated, and is unexpired via study_store.
        2. Verifies presence of all 4 required sequences.
        3. Runs brainseg inference and saves the BraTS-coded label map in the study directory.
        4. Reports exact missing components if dependencies/weights are not installed.
        """
        # 1. Verification of study
        study = study_store.get_validated_study(study_id)
        if not study.validated:
            raise NIfTIValidationError(
                message=f"Study '{study_id}' has not passed validation and cannot be segmented.",
                error_code="UNVALIDATED_STUDY",
                status_code=400,
            )

        # 2. Modality presence verification
        for mod in PROTOCOL_CHANNEL_ORDER:
            if mod not in study.modalities or study.modalities[mod].get("status") != "valid":
                raise NIfTIValidationError(
                    message=f"Study is missing valid sequence for modality '{mod}'.",
                    error_code="INCOMPLETE_STUDY",
                    status_code=400,
                )

        # 3. Check dependencies & weights
        is_ready, missing = self.check_dependencies()
        if not is_ready:
            missing_str = ", ".join(missing)
            raise RuntimeError(
                f"Segmentation dependencies or model weights are unavailable. "
                f"Missing components: {missing_str}. "
                f"Upload and validation succeeded, but inference requires these dependencies."
            )

        # 4. Run the shared pipeline on the uploaded files
        from brainseg.inference import load_sequences, run_case

        files = {MODALITY_TO_SEQUENCE[m]: Path(study.modalities[m]["path"]) for m in PROTOCOL_CHANNEL_ORDER}
        assert set(files) == set(MODEL_INPUT_ORDER)
        stack, ref = load_sequences(files)
        out_path = study.study_dir / PREDICTION_FILENAME
        with _GPU_LOCK:  # one inference at a time on the GPU
            record = run_case(stack, ref, out_path, files)
        for f in (Path(study.study_dir) / "viewer_cache").glob("*"):
            if f.name.startswith(("pred", "meshes", "uncert")):
                f.unlink()
        (study.study_dir / METADATA_FILENAME).write_text(json.dumps(record, indent=2), encoding="utf-8")
        study.metadata["prediction"] = record
        study_store.persist(study)

        vols = record["estimated_volumes"]
        sub_regions = {
            "whole_tumor": {"label": "Whole Tumor (WT)", "codes": [1, 2, 4], **_metric(vols["WT"])},
            "tumor_core": {"label": "Tumor Core (TC)", "codes": [1, 4], **_metric(vols["TC"])},
            "enhancing_tumor": {"label": "Enhancing Tumor (ET)", "codes": [4], **_metric(vols["ET"])},
        }

        return {
            "status": "success",
            "study_id": study_id,
            "message": "Suspected tumour regions predicted by the model. Volumes are estimates.",
            "model": f"{MODEL_INFO['name']} v{MODEL_INFO['version']}",
            "input_channel_order": PROTOCOL_CHANNEL_ORDER,
            "label_codes": record["label_codes"],
            "prediction_url": f"/api/segmentation/prediction/{study_id}",
            "sub_regions": sub_regions,
            "whole_tumor_volume_cm3": sub_regions["whole_tumor"]["volume_cm3"],
            "tumor_core_volume_cm3": sub_regions["tumor_core"]["volume_cm3"],
            "enhancing_tumor_volume_cm3": sub_regions["enhancing_tumor"]["volume_cm3"],
            "estimated_volumes": vols,
            "shape": record["shape"],
            "spacing_mm": record["spacing_mm"],
            "affine": record["affine"],
            "provenance": record["provenance"],
            "uncertainty": record.get("uncertainty"),
            "warnings": study.metadata.get("suitability", {}).get("warnings", []),
            "viewer_url": f"/viewer/{study_id}",
            "report_url": f"/report/{study_id}",
            "disclaimer": DISCLAIMER,
            "status_code": 200,
        }


def _metric(v: Dict[str, Any]) -> Dict[str, Any]:
    # 1 cm3 == 1 mL
    return {"volume_cm3": round(v["mL"], 2), "voxel_count": v["voxels"]}


# Global singleton pipeline
segmentation_pipeline = SegmentationEngine()
