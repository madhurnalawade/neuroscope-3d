# NeuroScope 3D — NIfTI Upload & Validation Module

Production-quality medical imaging upload, validation, and segmentation module for NeuroScope 3D, designed for the BraTS 2021 multi-parametric brain MRI challenge dataset.

## Features

- **Four Required Modalities**: T1, T1Gd (T1ce), T2, and FLAIR. Supports `.nii` and `.nii.gz` files.
- **Automatic Modality Recognition**: Case-insensitive token-boundary filename parsing recognizing BraTS conventions (`_t1`, `_t1ce`, `_t2`, `_flair`) and clinical aliases (`t1gd`, `t1c`, `t1post`, `t2w`, `t2flair`).
- **All 11 Validation Rules Enforced**:
  1. Strict extension check (`.nii`, `.nii.gz`)
  2. 1 GiB streaming size limit per file with immediate partial cleanup
  3. Format integrity verification and voxel payload validation
  4. Exactly 3D volume verification (no 2D slices or 4D series)
  5. 100% full-volume finite-value validation (no NaN or ±Inf)
  6. Non-constant intensity check (rejects all-zero or repeated intensities)
  7. Cross-sequence shape consistency
  8. Full 4×4 affine matrix congruence via `numpy.allclose(rtol=1e-3, atol=1e-3)`
  9. Spatial voxel zoom check and non-singular affine check (supports anisotropic scans like 1×1×2 mm)
  10. Required modalities enforcement (flags missing or duplicate assignments)
  11. Secure temporary storage, filename sanitization, and decompression protection
- **MONAI SegResNet Pipeline Integration**: Loads validated channels in strict protocol order (`[T1Gd, T1, T2, FLAIR]`) and computes Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) volumetrics in cm³.
- **Clean Responsive UI**: Four upload slots with drag-and-drop, visual status badges (Gray/Blue/Yellow/Green/Red), live single-file validation, validation summary card, and medical safety disclaimer.

## Quick Start

```bash
# 1. Run all 33 automated tests
./run_tests.sh

# 2. Start the FastAPI server and web app
./start_server.sh
```

- Open `http://localhost:8000/` for the web interface.
- Open `http://localhost:8000/docs` for the interactive Swagger API documentation.

## Test Suite

The test suite covers:
- **8 Mandatory Judges' Failure Pack Scenarios** (`tests/test_judges_failure_pack.py`):
  1. Fake NIfTI (text file renamed to .nii.gz)
  2. Corrupt Gzip archive
  3. Missing modality (e.g. FLAIR omitted)
  4. Shape mismatch across sequences
  5. 2D image rejection
  6. All-zero constant image rejection
  7. NaN / infinite values rejection
  8. Oversized file (> 1 GiB streaming cutoff)
- **20 Extended Validation Scenarios** (`tests/test_additional_validation.py`):
  1. Valid BraTS study
  2. .nii upload
  3. .nii.gz upload
  4. Mixed .nii and .nii.gz
  5. Duplicate modalities
  6. Wrong extensions (.dcm, .txt)
  7. Positive infinity
  8. Negative infinity
  9. Constant nonzero image
  10. Affine mismatch
  11. Orientation mismatch (LPS vs RAS)
  12. Different voxel spacing
  13. Valid anisotropic spacing (1×1×2 mm)
  14. Singular affine
  15. Truncated NIfTI payload
  16. Corrupt voxel data with valid header
  17. Malicious filename sanitization
  18. Interrupted upload cleanup
  19. Expired study ID rejection
  20. Invalid study ID segmentation rejection
- **5 FastAPI API Integration Tests** (`tests/test_api_integration.py`):
  End-to-end multipart form upload, validation summary, health check, and segmentation run.

## Medical Safety Notice

Research and teaching prototype only. Not for diagnosis or clinical decision-making. Use only public, de-identified research data.
