"""
Automatic MRI modality detection from filenames with token-boundary matching.
Recognizes T1, T1Gd (T1ce/T1c/T1post), T2, and FLAIR without false-positive overlaps.
"""

import os
import re
from typing import Optional

# Standard modality identifiers
MODALITY_T1 = "T1"
MODALITY_T1GD = "T1Gd"
MODALITY_T2 = "T2"
MODALITY_FLAIR = "FLAIR"

VALID_MODALITIES = [MODALITY_T1, MODALITY_T1GD, MODALITY_T2, MODALITY_FLAIR]

# Regex patterns matching exact token boundaries (\b, _, -, or string start/end)
# Order matters: more specific patterns (T1Gd, FLAIR) must be checked before T1 and T2!

# 1. T1Gd / T1ce patterns:
# matches t1ce, t1gd, t1c, t1post, t1_ce, t1-gd, t1+c, t1_post, etc.
T1GD_PATTERN = re.compile(
    r"(?:^|[_\-\.\s])(t1(?:ce|gd|c|post|\+c|\+gd|\-ce|\-gd|\-post|_ce|_gd|_post))(?:$|[_\-\.\s])",
    re.IGNORECASE,
)

# 2. FLAIR patterns:
# matches flair, t2flair, t2_flair, t2-flair
FLAIR_PATTERN = re.compile(
    r"(?:^|[_\-\.\s])(t2[\-_]?flair|flair)(?:$|[_\-\.\s])",
    re.IGNORECASE,
)

# 3. T2 patterns:
# matches t2, t2w, t2_weighted, t2star, but NOT t2flair (checked after FLAIR)
T2_PATTERN = re.compile(
    r"(?:^|[_\-\.\s])(t2(?:w|_w|\-w|weighted|_weighted|\*|star)?)(?:$|[_\-\.\s])",
    re.IGNORECASE,
)

# 4. T1 patterns:
# matches t1, t1w, t1_weighted, but NOT t1ce/t1gd/t1c (checked after T1Gd)
T1_PATTERN = re.compile(
    r"(?:^|[_\-\.\s])(t1(?:w|_w|\-w|weighted|_weighted)?)(?:$|[_\-\.\s])",
    re.IGNORECASE,
)


def detect_modality_from_filename(filename: str) -> Optional[str]:
    """
    Detect the MRI sequence modality from a filename.
    Returns 'T1', 'T1Gd', 'T2', 'FLAIR', or None if ambiguous/unrecognized.
    """
    if not filename:
        return None

    # Strip directory path and common NIfTI extensions
    name = os.path.basename(filename)
    lower = name.lower()
    for ext in (".nii.gz", ".nii", ".gz"):
        if lower.endswith(ext):
            name = name[: -len(ext)]
            break

    # 1. Check for T1Gd (T1ce / T1c / T1post) first to avoid greedy T1 match
    if T1GD_PATTERN.search(name):
        return MODALITY_T1GD

    # 2. Check for FLAIR (including t2flair) to avoid greedy T2 match
    if FLAIR_PATTERN.search(name):
        return MODALITY_FLAIR

    # 3. Check for T2 (t2, t2w)
    if T2_PATTERN.search(name):
        return MODALITY_T2

    # 4. Check for T1 (t1, t1w)
    if T1_PATTERN.search(name):
        return MODALITY_T1

    return None


def normalize_modality_name(modality: str) -> Optional[str]:
    """
    Normalize user or form input modality string to canonical form:
    'T1', 'T1Gd', 'T2', 'FLAIR'.
    """
    if not modality:
        return None

    clean = modality.strip().upper()
    if clean in ("T1", "T1W"):
        return MODALITY_T1
    if clean in ("T1GD", "T1CE", "T1C", "T1POST", "T1_CE", "T1_GD", "T1-CE", "T1-GD"):
        return MODALITY_T1GD
    if clean in ("T2", "T2W"):
        return MODALITY_T2
    if clean in ("FLAIR", "T2FLAIR", "T2_FLAIR"):
        return MODALITY_FLAIR

    return None
