"""End-to-end checks of the model pipeline on the synthetic phantom (plumbing only, not accuracy).

    python scripts/make_synthetic_case.py dataset/labelsTr/BRATS_001.nii.gz outputs/synthetic_BRATS_001
    python tests/check_pipeline.py outputs/synthetic_BRATS_001 outputs/expert_brats/BRATS_001.nii.gz
"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from brainseg.constants import SEQUENCES  # noqa: E402
from brainseg.inference import load_msd_4d, load_sequences, run_case  # noqa: E402
from brainseg.labels import region_masks  # noqa: E402


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * (a & b).sum() / s


def check_output(pred_path, input_ref):
    out = nib.load(str(pred_path))
    lab = np.asarray(out.dataobj)
    assert out.shape == input_ref.shape, (out.shape, input_ref.shape)
    assert np.allclose(out.affine, input_ref.affine), "affine not preserved"
    assert out.get_data_dtype() == np.uint8
    assert set(np.unique(lab)) <= {0, 1, 2, 4}, np.unique(lab)
    m = region_masks(lab)
    assert not (m["ET"] & ~m["TC"]).any() and not (m["TC"] & ~m["WT"]).any(), "regions not nested"
    return lab


def main(case_dir: Path, expert_path: Path):
    expert = np.asarray(nib.load(str(expert_path)).dataobj)
    em = region_masks(expert)
    results = {}

    # 1. Four separate files on the label grid.
    files = {s: case_dir / "ras_1mm" / f"{s}.nii.gz" for s in SEQUENCES}
    stack, ref = load_sequences(files)
    run_case(stack, ref, case_dir / "check" / "ras.nii.gz", files)
    lab_ras = check_output(case_dir / "check" / "ras.nii.gz", ref)
    results["four files, RAS 1 mm"] = lab_ras

    # 2. Same case as a Decathlon 4D file: must give the identical label map.
    stack4, ref4 = load_msd_4d(case_dir / "msd4d.nii.gz")
    assert np.array_equal(stack4, stack), "MSD channel reordering differs from four-file order"
    run_case(stack4, ref4, case_dir / "check" / "msd4d.nii.gz", {"msd4d": case_dir / "msd4d.nii.gz"})
    lab_4d = check_output(case_dir / "check" / "msd4d.nii.gz", ref4)
    assert np.array_equal(lab_4d, lab_ras), "4D and four-file predictions differ"

    # 3. Flipped orientation, 1.5 mm slices, shifted origin: output must sit on that grid.
    files = {s: case_dir / "lps_aniso" / f"{s}.nii.gz" for s in SEQUENCES}
    stack, ref = load_sequences(files)
    run_case(stack, ref, case_dir / "check" / "lps_aniso.nii.gz", files)
    lab_lps = check_output(case_dir / "check" / "lps_aniso.nii.gz", ref)
    # Map it back onto the 1 mm RAS grid to compare in the same space.
    from scipy.ndimage import zoom
    back = zoom(lab_lps[::-1, ::-1, :], (1, 1, expert.shape[2] / lab_lps.shape[2]), order=0)
    results["four files, LPS 1x1x1.5 mm (compared after mapping back)"] = back

    print("All structural checks passed: grid, affine, uint8, codes {0,1,2,4}, nesting, 4D == four files.")
    print("Dice vs expert mask on the SYNTHETIC phantom (sanity check only, not model accuracy):")
    for name, lab in results.items():
        pm = region_masks(lab)
        print(f"  {name}: " + ", ".join(f"{r} {dice(pm[r], em[r]):.3f}" for r in ("WT", "TC", "ET")))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
