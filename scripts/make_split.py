"""DEVELOPMENT ONLY. Reproducible patient-level split (train / dev / held-out) of the labelled Decathlon
cases with a fixed seed. Saves <OUTPUT_DIR>/msd_split.json and prints it.

  python scripts/make_split.py --dataset-root <DATASET_ROOT> --output-dir <OUTPUT_DIR> [--heldout 20] [--dev 20] [--seed 20261017]

Never train or tune on the held-out cases.
"""
import json
import sys
from pathlib import Path

from _cli import base_parser, finish, require


def main(argv=None) -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--dataset-root", type=Path, help="Decathlon Task01_BrainTumour folder (read only)")
    ap.add_argument("--output-dir", type=Path, help="folder for msd_split.json (not inside the dataset)")
    ap.add_argument("--heldout", type=int, default=20, help="held-out labelled cases (at least 10; default 20)")
    ap.add_argument("--dev", type=int, default=20, help="dev cases for tuning choices (default 20)")
    ap.add_argument("--seed", type=int, default=20261017, help="random seed (default 20261017)")
    args = finish(ap, argv)
    require(ap, args, "dataset_root", "output_dir")

    from neuroscope.dev.export import check_output_dir
    from neuroscope.dev.manifest import write_json
    from neuroscope.dev.msd import DatasetError
    from neuroscope.dev.split import make_split
    try:
        out = check_output_dir(args.output_dir, args.dataset_root)
        split = make_split(args.dataset_root, n_heldout=args.heldout, n_dev=args.dev, seed=args.seed)
    except (DatasetError, OSError, ValueError) as exc:
        print(f"FAILED: {exc}")
        return 1
    write_json(split, out, "msd_split.json")
    print(json.dumps({k: v for k, v in split.items() if k != "train"}, indent=1))
    print(f"train: {len(split['train'])} case IDs (full list in msd_split.json)")
    print("\nREMINDER: " + split["pretraining_overlap_reminder"])
    print("CAVEAT: " + split["patient_level_caveat"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
