"""Reproduce the judges' automated test pack against a running server, on a real case.

    python tests/judge_scenarios.py http://127.0.0.1:8000 demo_cases/BRATS_002

Scenarios (from the problem statement's scoring table): volume cube, anisotropic, orientation (same and
mirrored anatomy), reproducibility, linked-view coordinate mapping, input sanity (not skull-stripped, not a
brain). The failure pack is covered by Input/neuroscope_3d/backend/tests/test_judges_failure_pack.py.
"""
import gzip
import io
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import requests

SEQ_SUFFIX = {"t1": "_t1", "t1gd": "_t1ce", "t2": "_t2", "flair": "_flair"}
RESULTS = []


def check(name, ok, detail):
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def load_case(case_dir: Path):
    imgs = {s: nib.load(str(next(case_dir.glob(f"*{suf}.nii.gz"))), mmap=False) for s, suf in SEQ_SUFFIX.items()}
    return {s: np.asarray(i.dataobj, dtype=np.float32) for s, i in imgs.items()}, imgs["t1"].affine


def to_bytes(arr, affine):
    return gzip.compress(nib.Nifti1Image(arr, affine).to_bytes(), compresslevel=1)


def run(base, vols, affine, expert=None, tag="case"):
    files = [(s, (f"{tag}{SEQ_SUFFIX[s]}.nii.gz", to_bytes(v, affine), "application/gzip")) for s, v in vols.items()]
    if expert is not None:
        files.append(("expert", (f"{tag}_seg.nii.gz", to_bytes(expert.astype(np.uint8), affine), "application/gzip")))
    up = requests.post(f"{base}/api/upload/validate", files=files)
    if up.status_code != 200:
        return up.json(), None, None
    sid = up.json()["study_id"]
    seg = requests.post(f"{base}/api/segmentation/run", json={"study_id": sid}).json()
    pred = nib.Nifti1Image.from_bytes(gzip.decompress(requests.get(base + seg["prediction_url"]).content))
    meta = requests.get(f"{base}/api/viewer/{sid}/meta").json()
    return up.json(), np.asarray(pred.dataobj), meta


def world_x_side(labels, affine, brain):
    """Sign of (lesion centroid - brain centroid) along world x (RAS: + = patient right)."""
    lc = affine @ np.r_[np.argwhere(labels > 0).mean(0), 1]
    bc = affine @ np.r_[np.argwhere(brain).mean(0), 1]
    return "R" if lc[0] > bc[0] else "L"


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2 * (a & b).sum() / s


def main(base, case_dir):
    vols, aff = load_case(Path(case_dir))
    brain = vols["flair"] > 0

    # Baseline + reproducibility
    _, p1, m1 = run(base, vols, aff, tag="base")
    _, p2, _ = run(base, vols, aff, tag="base2")
    wt1 = m1["volumes"]["model"]["WT"]["mL"]
    d = dice(p1 > 0, p2 > 0)
    check("Reproducibility (same case twice)", d >= 0.99, f"WT Dice between runs {d:.4f} (need >= 0.99)")

    # Anisotropic: every second axial slice removed, spacing 1 x 1 x 2
    aff2 = aff.copy(); aff2[:3, 2] *= 2
    half = {s: v[:, :, ::2] for s, v in vols.items()}
    _, ph, mh = run(base, half, aff2, tag="aniso")
    wth = mh["volumes"]["model"]["WT"]["mL"]
    check("Anisotropic (every 2nd axial slice removed)", abs(wth - wt1) / wt1 <= 0.10,
          f"WT {wt1:.2f} mL vs {wth:.2f} mL ({100 * (wth - wt1) / wt1:+.1f}%, need within 10%)")

    # Volume: expert mask = 20x20x20 cube labelled 2 at 1 x 1 x 2 mm
    cube = np.zeros(half["t1"].shape, np.uint8)
    cube[100:120, 100:120, 30:50] = 2
    up, _, mc = run(base, half, aff2, expert=cube, tag="cube")
    ev = mc["volumes"]["expert"]
    check("Volume cube (expert WT 16.0 mL, TC 0, ET 0)",
          abs(ev["WT"]["mL"] - 16.0) <= 0.16 and ev["TC"]["mL"] == 0 and ev["ET"]["mL"] == 0,
          f"expert WT {ev['WT']['mL']:.3f} mL, TC {ev['TC']['mL']}, ET {ev['ET']['mL']} "
          f"({ev['WT']['voxels']} voxels x {ev['WT']['voxel_volume_mm3']} mm3)")

    # Orientation, same anatomy: voxels flipped L-R and affine changed to match
    X = vols["t1"].shape[0]
    flip = np.eye(4); flip[0, 0] = -1; flip[0, 3] = X - 1
    aff_f = aff @ flip
    flipped = {s: v[::-1].copy() for s, v in vols.items()}
    _, pf, mf = run(base, flipped, aff_f, tag="flipped")
    wtf = mf["volumes"]["model"]["WT"]["mL"]
    s0, sf = world_x_side(p1, aff, brain), world_x_side(pf, aff_f, brain[::-1])
    check("Orientation, same anatomy (flipped voxels + matching affine)",
          s0 == sf and abs(wtf - wt1) / wt1 <= 0.02,
          f"side {s0} -> {sf}; WT {wt1:.2f} vs {wtf:.2f} mL ({100 * (wtf - wt1) / wt1:+.2f}%, need within 2%)")

    # Orientation, mirrored anatomy: voxels flipped, affine unchanged
    _, pm, mm = run(base, flipped, aff, tag="mirrored")
    sm = world_x_side(pm, aff, brain[::-1])
    check("Orientation, mirrored anatomy (flipped voxels, same affine)", sm != s0, f"side {s0} -> {sm} (must change)")

    # Linked views: a marker at a known uploaded voxel maps to one canonical voxel and back (what the viewer does)
    A, O = np.array(mf["canonical_affine"]), np.array(mf["original_affine"])
    marker = np.array([60, 150, 70, 1.0])
    canon = np.linalg.inv(A) @ O @ marker
    back = np.linalg.inv(O) @ A @ np.round(canon)
    check("Linked views coordinate round-trip (flipped case)", np.abs(back - marker).max() <= 2,
          f"uploaded voxel {marker[:3].astype(int)} -> canonical {np.round(canon[:3]).astype(int)} -> {np.round(back[:3]).astype(int)}")

    # Input sanity: not skull-stripped (skull shell + scanner noise background)
    rng = np.random.default_rng(0)
    g = np.meshgrid(*[np.linspace(-1, 1, n) for n in vols["t1"].shape], indexing="ij")
    r = np.sqrt((g[0] / .9) ** 2 + (g[1] / .95) ** 2 + (g[2] / 1.0) ** 2)
    shell = (r > .82) & (r < .95)
    noisy = {s: np.where(brain, v, 0) + shell * np.percentile(v[brain], 90) + rng.normal(40, 10, v.shape).clip(1)
             for s, v in vols.items()}
    up, _, _ = run(base, noisy, aff, tag="skull")
    refused = up.get("error_code") == "UNSUITABLE_INPUT"
    check("Input sanity: not skull-stripped", refused or bool(up.get("warnings")),
          f"{'refused' if refused else 'warnings: ' + str(up.get('warnings'))}: {up.get('message', '')[:140]}")

    # Input sanity: not a brain (a smooth cylinder phantom with a zero background, e.g. a limb-like object)
    cyl = ((g[0] / .35) ** 2 + (g[1] / .35) ** 2 <= 1).astype(np.float32)
    notbrain = {s: (cyl * (300 + 50 * k) + rng.normal(0, 5, cyl.shape) * cyl).astype(np.float32)
                for k, s in enumerate(vols)}
    up, _, _ = run(base, notbrain, aff, tag="cylinder")
    refused = up.get("error_code") == "UNSUITABLE_INPUT"
    check("Input sanity: not a brain", refused or bool(up.get("warnings")),
          f"{'refused' if refused else 'warnings: ' + str(up.get('warnings'))}: {up.get('message', '')[:140]}")

    print(f"\n{sum(ok for _, ok, _ in RESULTS)}/{len(RESULTS)} scenarios passed")
    return all(ok for _, ok, _ in RESULTS)


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
