"""DEVELOPMENT ONLY. Build a JSON manifest of the real Decathlon cases from headers only.

  python scripts/build_manifest.py --dataset-root <DATASET_ROOT> --output-dir <OUTPUT_DIR> [--include-unlabelled]

Writes <OUTPUT_DIR>/msd_manifest.json (paths relative to the dataset root) and prints the counts.
"._" macOS sidecar files are ignored and counted separately.
"""
import sys
from pathlib import Path

from _cli import base_parser, finish, require


def main(argv=None) -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--dataset-root", type=Path, help="Decathlon Task01_BrainTumour folder (read only)")
    ap.add_argument("--output-dir", type=Path, help="folder for msd_manifest.json (not inside the dataset)")
    ap.add_argument("--include-unlabelled", action="store_true", help="also list imagesTs cases (no labels)")
    args = finish(ap, argv)
    require(ap, args, "dataset_root", "output_dir")

    from neuroscope.dev.export import check_output_dir
    from neuroscope.dev.manifest import build_manifest, write_json
    from neuroscope.dev.msd import DatasetError
    try:
        out = check_output_dir(args.output_dir, args.dataset_root)
        man = build_manifest(args.dataset_root, require_labels=not args.include_unlabelled,
                             progress=lambda i, n, cid: (i % 50 == 0 or i == n) and print(f"  headers {i}/{n}",
                                                                                          flush=True))
    except (DatasetError, OSError, ValueError, KeyError) as exc:
        print(f"FAILED: {exc if isinstance(exc, DatasetError) else type(exc).__name__}")
        return 1
    print()
    path = write_json(man, out, "msd_manifest.json")
    c = man["counts"]
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        print(f"{sub:<9} raw entries {c[sub]['raw_entries']:>4}  '._' sidecars ignored "
              f"{c[sub]['macos_sidecars_ignored']:>4}  real NIfTI files {c[sub]['nifti_files']:>4}")
    print(f"dataset.json: {c['dataset_json']['training_listed']} training, {c['dataset_json']['test_listed']} test")
    for k, v in c["unmatched"].items():
        print(f"{k}: {len(v)}" + (f"  {v[:10]}{' ...' if len(v) > 10 else ''}" if v else ""))
    print(f"Cases in manifest: {c['manifest_cases']}")
    print(f"Channels (from dataset.json): {man['channels']}")
    print(f"Wrote {path.name} to the output directory.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
