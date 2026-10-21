"""Load and validate one case and print a summary (shape, spacing, orientation, intensity ranges, hashes)
or a clear error. Nothing is written and nothing is sent anywhere.

Four files (production path):
  python scripts/load_case.py --t1 <T1> --t1gd <T1GD> --t2 <T2> --flair <FLAIR> [--seg <SEG>]
  python scripts/load_case.py <FILE1> <FILE2> <FILE3> <FILE4>          (sequence inferred from file names)
Decathlon case (development only):
  python scripts/load_case.py --dataset-root <DATASET_ROOT> --msd-case BRATS_001

Exit code 0 = valid, 1 = rejected, 2 = usage error.
"""
import json
import sys

from _cli import base_parser, finish


def main(argv=None) -> int:
    ap = base_parser(__doc__)
    ap.add_argument("files", nargs="*", help="sequence files whose sequence is inferred from the file name")
    for seq in ("t1", "t1gd", "t2", "flair"):
        ap.add_argument(f"--{seq}", help=f"explicit {seq.upper()} file (always wins over file-name inference)")
    ap.add_argument("--seg", help="optional expert mask in BraTS 2021 codes (0, 1 NCR, 2 ED, 4 ET)")
    ap.add_argument("--max-gb", type=float, default=1.0, help="per-file size limit in GB (default 1.0)")
    ap.add_argument("--dataset-root", help="Decathlon Task01 folder (development only, with --msd-case)")
    ap.add_argument("--msd-case", help="case id such as BRATS_001 (development only)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = finish(ap, argv)

    if args.msd_case:
        if not args.dataset_root:
            ap.error("--msd-case needs --dataset-root (or dataset_root in --config)")
        if args.files or any(getattr(args, s) for s in ("t1", "t1gd", "t2", "flair", "seg")):
            ap.error("use either four files or --dataset-root/--msd-case, not both")
        from neuroscope.dev.msd import DatasetError, load_msd_case, read_dataset   # dev only, imported lazily
        try:
            case = load_msd_case(read_dataset(args.dataset_root), args.msd_case)
        except (DatasetError, OSError, ValueError) as exc:
            print(f"REJECTED: {exc if isinstance(exc, DatasetError) else type(exc).__name__}")
            return 1
        print("DEVELOPMENT DATASET CASE (not an upload)")
    else:
        from neuroscope.io import load_four_files
        explicit = {s: getattr(args, s) for s in ("t1", "t1gd", "t2", "flair") if getattr(args, s)}
        res = load_four_files(explicit=explicit, unlabelled=args.files, seg=args.seg,
                              max_bytes=int(args.max_gb * 1e9))
        if not res.ok:
            print(f"REJECTED ({len(res.issues)} problem(s)); no result was produced:")
            for issue in res.issues:
                print(f"  - {issue}")
            return 1
        case = res.case

    summary = case.summary()
    if args.json:
        print(json.dumps(summary, indent=1))
        return 0
    print(f"OK  shape {tuple(summary['shape'])}  spacing {summary['spacing_mm']} mm  orientation {summary['orientation']}")
    for seq, s in summary["sequences"].items():
        print(f"  {seq:<5} {s['file']:<40} {s['dtype']:<8} range [{s['min']}, {s['max']}]  "
              f"p1-p99(nonzero) {s['p1_p99_nonzero']}  sha256 {s['sha256'][:16]}...")
    if "expert_mask" in summary:
        m = summary["expert_mask"]
        print(f"  mask  {m['file']:<40} voxels per BraTS code {m['voxels_per_code']}  sha256 {m['sha256'][:16]}...")
    for note in summary.get("notes", []):
        print(f"  note: {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
