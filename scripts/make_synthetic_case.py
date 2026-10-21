"""Build a synthetic 4-sequence phantom from an expert label map, for pipeline plumbing tests only.

The phantom is NOT real MRI. Its only purpose is to exercise loading, channel ordering, resampling and
grid restoration while the real images are unavailable. Writes two variants:
  <out>/ras_1mm/   identity affine, 1 mm, same grid as the label map
  <out>/lps_aniso/ flipped L/R and A/P, 1 x 1 x 1.5 mm, offset origin (tests reorientation + resampling)
plus <out>/msd4d.nii.gz in Decathlon channel order (FLAIR, T1w, T1Gd, T2w).
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainseg.constants import MSD_CHANNEL_ORDER  # noqa: E402

# Rough relative intensities per (background brain, ED, NCR, ET) for each sequence.
CONTRAST = {
    "t1":    (0.60, 0.50, 0.30, 0.50),
    "t1gd":  (0.60, 0.55, 0.30, 1.30),
    "t2":    (0.50, 1.00, 1.20, 0.80),
    "flair": (0.50, 1.10, 0.70, 0.90),
}


def main(label_path: Path, out: Path):
    lab_img = nib.load(str(label_path))
    lab = np.asarray(lab_img.dataobj)  # MSD codes: 1 edema, 2 non-enhancing, 3 enhancing
    rng = np.random.default_rng(0)
    xx, yy, zz = np.meshgrid(*[np.linspace(-1, 1, n) for n in lab.shape], indexing="ij")
    brain = (xx / 0.72) ** 2 + (yy / 0.85) ** 2 + ((zz - 0.05) / 0.85) ** 2 <= 1
    brain |= lab > 0

    vols = {}
    for seq, (b, ed, ncr, et) in CONTRAST.items():
        v = np.where(brain, b, 0.0)
        v[lab == 1], v[lab == 2], v[lab == 3] = ed, ncr, et
        v = gaussian_filter(v, 1.0) + rng.normal(0, 0.03, v.shape) * brain
        vols[seq] = (np.clip(v, 0, None) * 1000 * brain).astype(np.float32)

    d = out / "ras_1mm"
    d.mkdir(parents=True, exist_ok=True)
    for seq, v in vols.items():
        nib.save(nib.Nifti1Image(v, lab_img.affine), str(d / f"{seq}.nii.gz"))
    nib.save(lab_img, str(d / "expert_msd_codes.nii.gz"))

    msd = np.stack([vols[s] for s in MSD_CHANNEL_ORDER], axis=-1)
    nib.save(nib.Nifti1Image(msd, lab_img.affine), str(out / "msd4d.nii.gz"))

    # Variant with a different orientation, anisotropic spacing and a translated origin.
    d = out / "lps_aniso"
    d.mkdir(parents=True, exist_ok=True)
    aff = np.diag([-1.0, -1.0, 1.5, 1.0])
    aff[:3, 3] = (120.0, 130.0, -80.0)
    for seq, v in vols.items():
        w = zoom(v[::-1, ::-1, :], (1, 1, 1 / 1.5), order=1).astype(np.float32)
        nib.save(nib.Nifti1Image(w, aff), str(d / f"{seq}.nii.gz"))
    print(f"Synthetic phantom written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
