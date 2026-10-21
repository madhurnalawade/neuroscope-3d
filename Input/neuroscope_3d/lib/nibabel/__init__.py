"""
Production-quality lightweight, fully standard NIfTI-1 / NIfTI-2 module compatible with nibabel.
Provides nibabel API surface: load, save, Nifti1Image, Nifti1Header, orientations, affines.
"""

from .nifti1 import Nifti1Image, Nifti1Header, load, save
from . import orientations
from . import affines
from .filebasedimages import ImageFileError
from .spatialimages import HeaderDataError

__all__ = [
    "load",
    "save",
    "Nifti1Image",
    "Nifti1Header",
    "orientations",
    "affines",
    "ImageFileError",
    "HeaderDataError",
]
