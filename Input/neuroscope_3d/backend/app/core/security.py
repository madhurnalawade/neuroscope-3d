"""
Security utilities: filename sanitization, secure identifier generation,
and isolated temporary workspace management.
"""

import os
import re
import secrets
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Union

BASE_STUDY_DIR = Path(
    os.environ.get("NEUROSCOPE_STUDY_DIR", Path(tempfile.gettempdir()) / "neuroscope_studies")
).resolve()


def sanitize_filename(filename: str) -> str:
    """
    Sanitize an uploaded filename to prevent directory traversal, injection,
    and control-character attacks.
    """
    if not filename:
        return "unnamed_scan.nii.gz"

    # Remove null bytes and carriage returns
    clean = filename.replace("\x00", "").replace("\r", "").replace("\n", "")

    # Extract base filename ignoring any directory paths
    clean = os.path.basename(clean.replace("\\", "/"))

    # Remove dangerous directory traversal tokens
    clean = re.sub(r"\.\.+[/]?", "", clean)

    # Determine extension
    lower = clean.lower()
    ext = ""
    if lower.endswith(".nii.gz"):
        ext = ".nii.gz"
        stem = clean[:-7]
    elif lower.endswith(".nii"):
        ext = ".nii"
        stem = clean[:-4]
    else:
        # Unknown extension
        parts = clean.rsplit(".", 1)
        stem = parts[0]
        ext = ("." + parts[1]) if len(parts) > 1 else ""

    # Sanitize stem
    stem_clean = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", stem)
    stem_clean = stem_clean[:128] or "mri_sequence"

    return stem_clean + ext


def generate_secure_id(prefix: str = "study_") -> str:
    """Generate a high-entropy, cryptographically secure identifier."""
    token = secrets.token_hex(16)
    return f"{prefix}{token}"


def get_study_directory(study_id: str) -> Path:
    """Return the absolute path for an isolated study directory."""
    # Validate study_id contains only alphanumeric characters and underscores
    if not re.match(r"^[a-zA-Z0-9_\-]+$", study_id):
        raise ValueError(f"Invalid characters in study_id: {study_id}")
    return BASE_STUDY_DIR / study_id


def create_temp_study_dir(study_id: str) -> Path:
    """Create an isolated temporary directory for the uploaded study."""
    study_dir = get_study_directory(study_id)
    study_dir.mkdir(parents=True, exist_ok=True)
    return study_dir


def cleanup_study_dir(study_dir: Union[str, Path]):
    """Safely remove a temporary study directory and all its contents."""
    try:
        path = Path(study_dir)
        if path.exists() and path.is_dir() and str(BASE_STUDY_DIR) in str(path.resolve()):
            shutil.rmtree(path, ignore_errors=True)
    except Exception:
        pass
