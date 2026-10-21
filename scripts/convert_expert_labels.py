"""Convert Decathlon Task01 expert label maps (1=edema, 2=non-enhancing, 3=enhancing) to BraTS codes (1/2/4).

    python scripts/convert_expert_labels.py dataset/labelsTr outputs/expert_brats
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainseg.labels import msd_to_brats  # noqa: E402


def main(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(f for f in src_dir.glob("*.nii*") if not f.name.startswith("._"))
    for f in files:
        img = nib.load(str(f))
        msd = np.asarray(img.dataobj)
        brats = msd_to_brats(msd)
        # Check the mapping voxel-for-voxel: edema 1->2, non-enhancing 2->1, enhancing 3->4.
        for src, dst in ((1, 2), (2, 1), (3, 4)):
            assert ((msd == src) == (brats == dst)).all()
        header = img.header.copy()
        header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(brats, img.affine, header), str(dst_dir / f.name))
        counts = {int(k): int(v) for k, v in zip(*np.unique(brats, return_counts=True))}
        print(f"{f.name}: {counts}")
    print(f"Converted {len(files)} label maps -> {dst_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
