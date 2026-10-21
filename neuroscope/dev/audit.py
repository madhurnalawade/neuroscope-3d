"""Streaming audit of every labelled Decathlon case: one case in memory at a time, failures collected."""
import datetime
import time

import nibabel as nib
import numpy as np

from ..security.hygiene import describe_exception
from .msd import read_dataset


def audit_case(info, entry) -> list:
    """Return a list of problems for one case (empty list = passed)."""
    problems = []
    img = nib.load(str(info.root / entry.image), mmap=False)
    lab = nib.load(str(info.root / entry.label), mmap=False)
    n_ch = len(info.channels)
    if img.ndim != 4 or img.shape[3] != n_ch:
        problems.append(f"image shape {img.shape} is not (X, Y, Z, {n_ch})")
        return problems
    if tuple(lab.shape) != tuple(img.shape[:3]):
        problems.append(f"mask shape {lab.shape} != image spatial shape {img.shape[:3]}")
    if not np.allclose(lab.affine, img.affine, rtol=0, atol=1e-3):
        problems.append(f"mask affine differs from image affine (max {np.abs(lab.affine - img.affine).max():.3g})")
    for idx, seq in sorted(info.channels.items()):
        a = np.asarray(img.dataobj[..., idx])
        if a.dtype.kind in "fc" and not np.all(np.isfinite(a)):
            problems.append(f"channel {idx} ({seq}) has non-finite values")
        if a.min() == a.max():
            problems.append(f"channel {idx} ({seq}) is constant")
        del a
    m = np.asarray(lab.dataobj)
    if m.dtype.kind in "fc" and not np.all(np.isfinite(m)):
        problems.append("mask has non-finite values")
    values = sorted(int(v) for v in np.unique(m) if np.isfinite(v))
    extra = [v for v in values if v not in info.label_map]
    if extra:
        problems.append(f"mask values {extra} not in dataset.json labels {sorted(info.label_map)}")
    return problems


def audit_dataset(dataset_root, progress=print, limit=None) -> dict:
    info = read_dataset(dataset_root)
    cases = [c for c in info.cases if c.label]
    if limit:
        cases = cases[:limit]
    started = time.monotonic()
    results, failures = [], []
    for i, c in enumerate(cases, 1):
        try:
            problems = audit_case(info, c)
        except Exception as exc:  # noqa: BLE001 - record and continue with the next case
            problems = [f"unreadable ({describe_exception(exc)})"]
        results.append({"case_id": c.case_id, "ok": not problems, "problems": problems})
        if problems:
            failures.append(c.case_id)
        if progress:
            elapsed = time.monotonic() - started
            eta = elapsed / i * (len(cases) - i)
            progress(f"[{i}/{len(cases)}] {c.case_id} {'ok' if not problems else 'FAIL'}  "
                     f"elapsed {elapsed:.0f}s  eta {eta:.0f}s")
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_root_name": info.root.name,
        "checked": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "failed_case_ids": failures,
        "checks": ["readable", "image is 4D with one channel per dataset.json modality", "mask shape == image "
                   "spatial shape", "mask affine == image affine (1e-3 mm)", "finite values", "no constant channel",
                   "mask values within dataset.json labels"],
        "results": results,
    }
