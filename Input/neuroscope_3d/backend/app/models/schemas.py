"""
Pydantic schemas for NIfTI validation and segmentation requests/responses.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class StudyMetadata(BaseModel):
    shape: List[int] = Field(..., description="Voxel array dimensions [x, y, z]")
    voxel_spacing_mm: List[float] = Field(..., description="Voxel spacing in millimeters [dx, dy, dz]")
    orientation: List[str] = Field(..., description="Anatomical axis orientation codes e.g. ['R', 'A', 'S']")
    affine_consistent: bool = Field(True, description="Whether full 4x4 affine matrices match within tolerance")
    all_finite: bool = Field(True, description="Whether all voxels contain finite values (no NaN or Inf)")
    all_non_constant: bool = Field(True, description="Whether all sequences have non-constant intensity variation")
    total_size_bytes: Optional[int] = Field(None, description="Total size of all 4 uploaded sequences in bytes")
    total_size_mb: Optional[float] = Field(None, description="Total size in Megabytes")
    validation_duration_seconds: Optional[float] = Field(None, description="Time taken to validate in seconds")


class StudyValidationSuccessResponse(BaseModel):
    status: str = "success"
    valid: bool = True
    message: str = "All four MRI sequences validated successfully."
    study_id: str
    modalities: Dict[str, str] = Field(
        default_factory=lambda: {
            "T1": "valid",
            "T1Gd": "valid",
            "T2": "valid",
            "FLAIR": "valid",
        }
    )
    metadata: StudyMetadata
    ready_for_segmentation: bool = True


class StudyValidationErrorResponse(BaseModel):
    status: str = "error"
    valid: bool = False
    error_code: str
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    ready_for_segmentation: bool = False


class SingleFileValidationResponse(BaseModel):
    valid: bool
    filename: str
    modality: Optional[str] = None
    file_size_mb: float
    error_code: Optional[str] = None
    message: Optional[str] = None
    shape: Optional[List[int]] = None
    zooms: Optional[List[float]] = None
    orientation: Optional[List[str]] = None


class SegmentationRunRequest(BaseModel):
    study_id: str = Field(..., description="The validated study identifier")


class TumorSubRegionMetrics(BaseModel):
    label: Optional[str] = None
    codes: Optional[List[int]] = None
    volume_cm3: float
    voxel_count: int
    mean_intensity: Optional[float] = None


class SegmentationRunResponse(BaseModel):
    status: str = "success"
    study_id: str
    message: str = "Suspected tumour regions predicted by the model. Volumes are estimates."
    model: str
    input_channel_order: List[str] = ["T1Gd", "T1", "T2", "FLAIR"]
    label_codes: Dict[str, str] = Field(..., description="BraTS codes in the prediction NIfTI")
    prediction_url: str = Field(..., description="Download URL of the NIfTI label map (input grid/affine)")
    sub_regions: Dict[str, TumorSubRegionMetrics]
    whole_tumor_volume_cm3: float
    tumor_core_volume_cm3: float
    enhancing_tumor_volume_cm3: float
    estimated_volumes: Dict[str, Dict[str, Any]] = Field(
        ..., description="Per label/region: voxels, voxel_volume_mm3, mm3, mL (voxel count x voxel volume)"
    )
    shape: List[int]
    spacing_mm: List[float]
    affine: List[List[float]]
    provenance: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = Field(None, description="Test-time-augmentation agreement summary")
    warnings: List[str] = Field(default_factory=list, description="Input suitability warnings")
    viewer_url: str
    report_url: str
    disclaimer: str
    status_code: int = 200
