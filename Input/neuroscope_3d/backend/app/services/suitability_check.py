"""Study-level suitability check (skull-stripped brain MRI?) run after structural validation."""
from typing import Any, Dict

import nibabel as nib
import numpy as np

import app.core.paths  # noqa: F401
from app.core.exceptions import NIfTIValidationError

from brainseg.suitability import assess


def check_study_suitability(validated_seqs: Dict[str, Any]) -> Dict[str, Any]:
    """Raise NIfTIValidationError (422) when the study is clearly unsuitable; otherwise return warnings."""
    volumes = {mod: np.asarray(nib.load(str(info.saved_path), mmap=False).dataobj) for mod, info in validated_seqs.items()}
    spacing = next(iter(validated_seqs.values())).zooms
    result = assess(volumes, spacing)
    if result["refusals"]:
        raise NIfTIValidationError(
            message="Input refused: " + " ".join(result["refusals"]),
            error_code="UNSUITABLE_INPUT",
            details=result,
            status_code=422,
        )
    return result
