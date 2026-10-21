"""DEVELOPMENT ONLY. Audit every labelled Decathlon case, one at a time, and write all failures to
<OUTPUT_DIR>/msd_audit.json instead of stopping at the first one.

  python scripts/audit_dataset.py --dataset-root <DATASET_ROOT> --output-dir <OUTPUT_DIR> [--limit N]

Checks: readable, 4D with one channel per dataset.json modality, mask shape == image spatial shape,
affines match, finite values, no constant channel, mask values within dataset.json labels.
This is a developer log, not a user-facing report. Exit code 0 = all passed, 1 = failures found.
"""
import sys
from pathlib import Path

from _cli import base_parser, finish, require


def main(argv=None) -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--dataset-root", type=Path, help="Decathlon Task01_BrainTumour folder (read only)")
    ap.add_argument("--output-dir", type=Path, help="folder for msd_audit.json (not inside the dataset)")
    ap.add_argument("--limit", type=int, default=None, help="audit only the first N labelled cases")
    args = finish(ap, argv)
    require(ap, args, "dataset_root", "output_dir")

    from neuroscope.dev.audit import audit_dataset
    from neuroscope.dev.export import check_output_dir
    from neuroscope.dev.manifest import write_json
    from neuroscope.dev.msd import DatasetError
    try:
        out = check_output_dir(args.output_dir, args.dataset_root)
        rep = audit_dataset(args.dataset_root, progress=lambda line: print(line, flush=True), limit=args.limit)
    except (DatasetError, OSError, ValueError, KeyError) as exc:
        print(f"FAILED: {exc if isinstance(exc, DatasetError) else type(exc).__name__}")
        return 1
    print()
    write_json(rep, out, "msd_audit.json")
    print(f"Checked {rep['checked']}: {rep['passed']} passed, {rep['failed']} failed.")
    for r in rep["results"]:
        if not r["ok"]:
            print(f"  {r['case_id']}: " + "; ".join(r["problems"]))
    print("Wrote msd_audit.json to the output directory.")
    return 0 if rep["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
