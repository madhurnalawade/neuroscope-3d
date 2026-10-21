"""DEVELOPMENT ONLY. Export one Decathlon 4D case as four separate 3D NIfTI files + a BraTS-coded seg file,
then reload them through the production four-file loader and verify shape, affine, spacing and every
voxel value match exactly. Use the exported files to test the same path the judges will use.

  python scripts/export_msd_case.py --dataset-root <DATASET_ROOT> --case BRATS_001 --output-dir <OUTPUT_DIR>

Channel names come from dataset.json. No rotation, flip or resampling is applied.
Exit code 0 = exported and verified, 1 = failed.
"""
import sys
from pathlib import Path

from _cli import base_parser, finish, require


def main(argv=None) -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--dataset-root", type=Path, help="Decathlon Task01_BrainTumour folder (read only)")
    ap.add_argument("--case", required=True, help="case id, e.g. BRATS_001")
    ap.add_argument("--output-dir", type=Path, help="where to write the five files (not inside the dataset)")
    ap.add_argument("--overwrite", action="store_true", help="replace files from an earlier export")
    args = finish(ap, argv)
    require(ap, args, "dataset_root", "output_dir")

    from neuroscope.dev.export import export_case, verify_export
    from neuroscope.dev.msd import DatasetError, load_msd_case, read_dataset
    try:
        info = read_dataset(args.dataset_root)
        case = load_msd_case(info, args.case)
        paths = export_case(info, case, args.case, args.output_dir, overwrite=args.overwrite)
    except (DatasetError, OSError, ValueError) as exc:
        print(f"FAILED: {exc if isinstance(exc, DatasetError) else type(exc).__name__}")
        return 1
    print("Channel order read from dataset.json: "
          + ", ".join(f"{i}={s}" for i, s in sorted(info.channels.items())))
    print("Label mapping (dataset.json name -> BraTS 2021 code): "
          + ", ".join(f"{k} '{info.labels[str(k)]}' -> {v}" for k, v in sorted(info.label_map.items())))
    for key, p in paths.items():
        print(f"  wrote {key:<5} {p.name}")
    problems = verify_export(case, paths)
    if problems:
        print("VERIFY FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"VERIFIED: reloaded through the four-file loader; shape {case.shape}, spacing {case.spacing} mm, "
          f"orientation {''.join(case.orientation)}, affine and all voxel values identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
