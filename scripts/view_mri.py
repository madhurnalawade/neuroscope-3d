"""Show the four MRI sequences of one case, optionally with a BraTS label map (1/2/4) overlaid.

    python scripts/view_mri.py BRATS_001
    python scripts/view_mri.py BRATS_001 --labels outputs/BRATS_001/prediction.nii.gz
"""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "dataset"
METADATA_PATH = DATA_DIR / "dataset.json"

# BraTS codes: 1 NCR (red), 2 ED (green), 4 ET (yellow); index = code
LABEL_CMAP = ListedColormap(["none", "red", "limegreen", "none", "yellow"])


def find_image(case_id):
    for folder in ("imagesTr", "images.tr", "imagesTs"):
        path = DATA_DIR / folder / f"{case_id}.nii.gz"
        if path.exists() and path.stat().st_size > 1024:  # skip macOS "._" stubs
            return path
    raise FileNotFoundError(f"MRI file not found for {case_id} under {DATA_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id", nargs="?", default="BRATS_001")
    parser.add_argument("--labels", type=Path, help="BraTS-coded label map to overlay")
    parser.add_argument("--slice", type=int, help="Axial slice index (default: middle)")
    args = parser.parse_args()

    # Read dataset metadata to identify MRI channels
    modalities = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            modalities = json.load(f).get("modality", {})

    image_path = find_image(args.case_id)
    image = nib.load(str(image_path))
    volume = image.get_fdata(dtype=np.float32)

    print("MRI file:", image_path.name)
    print("Volume shape:", volume.shape)
    print("Voxel spacing:", image.header.get_zooms())
    print("Intensity range:", np.nanmin(volume), "to", np.nanmax(volume))

    # Expect a 3D volume with four MRI channels
    if volume.ndim != 4 or volume.shape[3] != 4:
        raise ValueError(f"Expected MRI data shaped (X, Y, Z, 4), got {volume.shape}")

    labels = None
    if args.labels:
        labels = np.asarray(nib.load(str(args.labels)).dataobj)
        if labels.shape != volume.shape[:3]:
            raise ValueError(f"Label map shape {labels.shape} does not match image {volume.shape[:3]}")
        print("Label codes present:", np.unique(labels).tolist())

    slice_idx = args.slice if args.slice is not None else volume.shape[2] // 2

    # Display all four MRI sequences
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for channel, ax in enumerate(axes):
        ax.imshow(np.rot90(volume[:, :, slice_idx, channel]), cmap="gray")
        if labels is not None:
            ax.imshow(np.rot90(labels[:, :, slice_idx]), cmap=LABEL_CMAP, vmin=0, vmax=4,
                      alpha=0.4, interpolation="nearest")
        ax.set_title(modalities.get(str(channel), f"Channel {channel}"))
        ax.axis("off")

    title = f"{image_path.name} — slice {slice_idx}"
    if labels is not None:
        title += "  |  red NCR (1), green ED (2), yellow ET (4) — suspected regions predicted by the model"
    fig.suptitle(title)
    fig.text(0.5, 0.01, "Research prototype. Not a diagnosis.", ha="center", fontsize=9)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
