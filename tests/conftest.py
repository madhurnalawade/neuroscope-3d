"""Shared fixtures for the neuroscope tests. Every test runs with the network guard on and fails if any
connection or DNS lookup was attempted. All data is synthetic and written under pytest's tmp_path."""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neuroscope.security import netguard  # noqa: E402

SHAPE = (12, 10, 8)
AFFINE = np.array([[-1.0, 0, 0, 90], [0, 1.0, 0, -126], [0, 0, 1.0, -72], [0, 0, 0, 1]])


@pytest.fixture(autouse=True)
def no_network(request):
    """Block the network for every test; fail the test if anything tried to connect."""
    if "allow_attempts" in request.keywords:
        with netguard.network_guard():
            yield
        return
    with netguard.network_guard() as attempts:
        yield
    assert not attempts, f"network access attempted: {attempts}"


def pytest_configure(config):
    config.addinivalue_line("markers", "allow_attempts: test deliberately triggers the network guard")


def volume(seed=0, shape=SHAPE, dtype=np.float32):
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 1000).astype(dtype)


def save(path, data, affine=AFFINE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(path))
    return path


@pytest.fixture
def four_files(tmp_path):
    """Valid synthetic study: four sequence files named by sequence."""
    names = {"FLAIR": "case_flair.nii.gz", "T1": "case_t1.nii.gz", "T1Gd": "case_t1gd.nii.gz", "T2": "case_t2.nii.gz"}
    return {seq: save(tmp_path / "in" / name, volume(i)) for i, (seq, name) in enumerate(names.items())}


def make_msd(root: Path, modality=None, labels=None, n_cases=3, n_channels=4, sidecars=True):
    """Tiny Decathlon-style dataset: 4D images (X, Y, Z, C), 3D masks in Decathlon codes 0-3, dataset.json."""
    modality = modality or {"0": "FLAIR", "1": "T1w", "2": "t1gd", "3": "T2w"}
    labels = labels or {"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"}
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    training = []
    for k in range(1, n_cases + 1):
        cid = f"BRATS_{k:03d}"
        # channel c is filled with values 100*(c+1) + noise, so tests can tell channels apart
        img = np.stack([volume(k * 10 + c) + 100 * (c + 1) for c in range(n_channels)], axis=-1)
        save(root / "imagesTr" / f"{cid}.nii.gz", img)
        mask = np.zeros(SHAPE, np.uint8)
        mask[2:6, 2:6, 2:6] = 1
        mask[3:5, 3:5, 3:5] = 2
        mask[4, 4, 4] = 3
        save(root / "labelsTr" / f"{cid}.nii.gz", mask)
        training.append({"image": f"./imagesTr/{cid}.nii.gz", "label": f"./labelsTr/{cid}.nii.gz"})
        if sidecars:
            (root / "imagesTr" / f"._{cid}.nii.gz").write_bytes(b"\x00\x05\x16\x07Mac OS X")
            (root / "labelsTr" / f"._{cid}.nii.gz").write_bytes(b"\x00\x05\x16\x07Mac OS X")
    save(root / "imagesTs" / "BRATS_900.nii.gz", np.stack([volume(900 + c) for c in range(n_channels)], -1))
    (root / "dataset.json").write_text(json.dumps({
        "name": "synthetic", "modality": modality, "labels": labels, "numTraining": n_cases,
        "training": training, "test": ["./imagesTs/BRATS_900.nii.gz"],
    }), encoding="utf-8")
    return root
