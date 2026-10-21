"""Error codes and the structured issue type returned by every check (no exceptions for bad input)."""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# per file
FILE_NOT_FOUND = "FILE_NOT_FOUND"
NOT_A_FILE = "NOT_A_FILE"
EMPTY_FILE = "EMPTY_FILE"
FILE_TOO_LARGE = "FILE_TOO_LARGE"
BAD_EXTENSION = "BAD_EXTENSION"
NOT_GZIP = "NOT_GZIP"                # named .nii.gz but not gzip data
CORRUPT_GZIP = "CORRUPT_GZIP"        # gzip header present but stream truncated or damaged
NOT_NIFTI = "NOT_NIFTI"
UNREADABLE = "UNREADABLE"
NOT_3D = "NOT_3D"
DATA_TOO_LARGE = "DATA_TOO_LARGE"    # header promises more voxel data than allowed
NON_FINITE = "NON_FINITE"
CONSTANT = "CONSTANT"
BAD_SPACING = "BAD_SPACING"
BAD_AFFINE = "BAD_AFFINE"
BAD_LABELS = "BAD_LABELS"            # expert mask with values outside the BraTS 2021 codes

# set level
UNKNOWN_SEQUENCE = "UNKNOWN_SEQUENCE"
AMBIGUOUS_SEQUENCE = "AMBIGUOUS_SEQUENCE"
DUPLICATE_SEQUENCE = "DUPLICATE_SEQUENCE"
MISSING_SEQUENCE = "MISSING_SEQUENCE"
SHAPE_MISMATCH = "SHAPE_MISMATCH"
AFFINE_MISMATCH = "AFFINE_MISMATCH"
SPACING_MISMATCH = "SPACING_MISMATCH"


@dataclass
class Issue:
    code: str
    message: str            # human readable; contains only sanitised short file names, never voxel data
    file: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"


def issue(code: str, message: str, file: Optional[str] = None, **details) -> Issue:
    return Issue(code=code, message=message, file=file, details=details)
