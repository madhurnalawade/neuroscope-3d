"""Split a Decathlon Task01 4D case into four 3D sequence files + a BraTS-coded expert mask, ready to upload.

    python scripts/prepare_demo_case.py data/Task01_BrainTumour BRATS_001 demo_cases

Writes demo_cases/BRATS_001/BRATS_001_{t1,t1ce,t2,flair}.nii.gz and BRATS_001_seg.nii.gz (codes 1/2/4).
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainseg.constants import MSD_CHANNEL_ORDER  # noqa: E402
from brainseg.labels import msd_to_brats  # noqa: E402

FILE_SUFFIX = {"t1": "t1", "t1gd": "t1ce", "t2": "t2", "flair": "flair"}  # BraTS naming


def main(data_dir: Path, case: str, out_root: Path):
    img = nib.load(str(data_dir / "imagesTr" / f"{case}.nii.gz"), mmap=False)
    data = np.asarray(img.dataobj, dtype=np.float32)
    out = out_root / case
    out.mkdir(parents=True, exist_ok=True)
    for ch, seq in enumerate(MSD_CHANNEL_ORDER):
        nib.save(nib.Nifti1Image(data[..., ch], img.affine), str(out / f"{case}_{FILE_SUFFIX[seq]}.nii.gz"))
    lab_path = data_dir / "labelsTr" / f"{case}.nii.gz"
    if lab_path.exists():
        lab = nib.load(str(lab_path), mmap=False)
        nib.save(nib.Nifti1Image(msd_to_brats(np.asarray(lab.dataobj)), lab.affine), str(out / f"{case}_seg.nii.gz"))
    print(f"Wrote {sorted(p.name for p in out.iterdir())} to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
