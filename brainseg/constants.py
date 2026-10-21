"""Shared interface constants. Every workstream should import these rather than hard-coding values."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT / "models" / "brats_mri_segmentation"
CHECKPOINT = BUNDLE_DIR / "models" / "model.pt"

# Canonical sequence names used across the app (upload form, API, pipeline).
SEQUENCES = ("t1", "t1gd", "t2", "flair")

# Channel order the pretrained MONAI bundle expects (configs/metadata.json: T1c, T1, T2, FLAIR).
MODEL_INPUT_ORDER = ("t1gd", "t1", "t2", "flair")

# Channel order inside the Medical Segmentation Decathlon Task01 4D images
# (dataset.json "modality": FLAIR, T1w, t1gd, T2w).
MSD_CHANNEL_ORDER = ("flair", "t1", "t1gd", "t2")

# BraTS label codes for the output label map. Category identifiers, not severity scores.
BACKGROUND, NCR, ED, ET = 0, 1, 2, 4
LABEL_NAMES = {NCR: "NCR (necrotic core)", ED: "ED (oedema)", ET: "ET (enhancing tumour)"}

# Medical Segmentation Decathlon Task01 label codes -> BraTS codes.
# MSD: 1 = edema, 2 = non-enhancing tumour, 3 = enhancing tumour.
MSD_TO_BRATS = {0: BACKGROUND, 1: ED, 2: NCR, 3: ET}

# Grouped regions derived from BraTS codes.
REGIONS = {
    "WT": (NCR, ED, ET),  # whole tumour
    "TC": (NCR, ET),      # tumour core
    "ET": (ET,),          # enhancing tumour
}

MODEL_INFO = {
    "name": "MONAI brats_mri_segmentation (SegResNet)",
    "version": "0.5.4",
    "licence": "Apache-2.0 (MONAI Consortium)",
    "training_data": "BraTS 2018 (285 cases; bundle authors' own split)",
    "source": "https://huggingface.co/MONAI/brats_mri_segmentation",
}

PREPROCESSING = (
    "Stack sequences as [T1Gd, T1, T2, FLAIR]; reorient to RAS; resample to 1 mm isotropic (bilinear); "
    "z-score intensity normalisation per channel over non-zero voxels; sliding-window inference with AMP; "
    "sigmoid; invert resampling/orientation back to the input grid (bilinear on probabilities); "
    "threshold 0.5; TC/WT/ET channels -> BraTS codes (ET=4, TC without ET=1, WT without TC=2). "
    "Uncertainty: 3 extra passes on the input mirrored along each axis (label map from the plain pass)."
)

LIMITATIONS = [
    "Research and teaching prototype; not clinically validated and not approved for any clinical use. It does not "
    "provide diagnosis, grading, prognosis, treatment or surgical/radiotherapy planning.",
    "The pretrained model was trained by the MONAI team on BraTS 2018 (adult glioma, pre-operative, skull-stripped, "
    "co-registered, 1 mm). It may fail on other tumour types (metastases, meningioma, paediatric tumours), "
    "post-operative scans, non-skull-stripped or non-co-registered images, unusual contrast, motion or other artefacts.",
    "Evaluation cases come from the Medical Segmentation Decathlon Task01, which derives from BraTS 2016/2017; most "
    "are likely included in the model's BraTS 2018 training data, so reported Dice/sensitivity are optimistic.",
    "Enhancing tumour (ET) and necrotic core (NCR) are small, low-contrast regions and are the least reliable; "
    "ET can be missed entirely or predicted where an expert would mark none (in our evaluation, non-enhancing "
    "tumours such as BRATS_175 and BRATS_169 received spurious ET/TC predictions).",
    "Images coarser than 1 mm are resampled to 1 mm; thick-slice data reduces accuracy and volume precision.",
    "Volumes are estimates (voxel count x voxel volume) of the model's predicted regions, not measurements of "
    "the tumour, and inherit all segmentation errors.",
    "Input suitability checks are simple heuristics (background fraction, tissue volume, mask agreement, size, "
    "shape, texture); passing them does not guarantee the scan is appropriate.",
    "The test-time-augmentation uncertainty only measures how the prediction changes when the scan is mirrored; a "
    "confidently wrong region can be unanimous across passes. Modality ablation is not clinical evidence.",
]

DISCLAIMER = (
    "Research and teaching prototype. Outputs are suspected tumour regions predicted by the model and "
    "estimated volumes. Not a diagnosis; not clinically validated; not for clinical use."
)
