"""Upload four sequences to a running server, run segmentation, download and check the label map.

    python tests/http_end_to_end.py http://127.0.0.1:8000 T1.nii.gz T1Gd.nii.gz T2.nii.gz FLAIR.nii.gz [expert_brats.nii.gz]
"""
import gzip
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import requests


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def main(base, t1, t1gd, t2, flair, expert=None):
    t0 = time.perf_counter()
    files = [(k, (Path(p).name, open(p, "rb"))) for k, p in (("t1", t1), ("t1gd", t1gd), ("t2", t2), ("flair", flair))]
    up = requests.post(f"{base}/api/upload/validate", files=files)
    t_up = time.perf_counter() - t0
    print(f"upload+validate: HTTP {up.status_code} in {t_up:.1f} s")
    up.raise_for_status()
    study_id = up.json()["study_id"]

    seg = requests.post(f"{base}/api/segmentation/run", json={"study_id": study_id})
    t_seg = time.perf_counter() - t0
    print(f"segmentation:    HTTP {seg.status_code} at {t_seg:.1f} s")
    seg.raise_for_status()
    body = seg.json()

    dl = requests.get(base + body["prediction_url"])
    dl.raise_for_status()
    pred = nib.Nifti1Image.from_bytes(gzip.decompress(dl.content))
    lab = np.asarray(pred.dataobj)
    ref = nib.load(t1)
    print(f"download:        HTTP {dl.status_code} at {time.perf_counter() - t0:.1f} s total")

    assert pred.shape == ref.shape, (pred.shape, ref.shape)
    assert np.allclose(pred.affine, ref.affine)
    assert lab.dtype == np.uint8 and set(np.unique(lab).tolist()) <= {0, 1, 2, 4}
    print(f"label map: shape {pred.shape}, affine matches input, dtype uint8, codes {np.unique(lab).tolist()}")
    for name, v in body["estimated_volumes"].items():
        print(f"  {name:>3}: {v['voxels']:>7} voxels x {v['voxel_volume_mm3']:.3f} mm3 = {v['mL']:7.2f} mL (estimated)")
    print("provenance:", {k: v for k, v in body["provenance"].items() if k not in ("inputs", "model", "preprocessing")})

    if expert:
        e = np.asarray(nib.load(expert).dataobj)
        for name, codes in (("WT", (1, 2, 4)), ("TC", (1, 4)), ("ET", (4,))):
            print(f"  Dice {name} vs expert: {dice(np.isin(lab, codes), np.isin(e, codes)):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
