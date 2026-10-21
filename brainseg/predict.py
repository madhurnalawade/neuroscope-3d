"""Command-line entry point: four NIfTI sequences in, BraTS label map (1/2/4) on the input grid out.

    python -m brainseg.predict --t1 T1.nii.gz --t1gd T1Gd.nii.gz --t2 T2.nii.gz --flair FLAIR.nii.gz \
        --out outputs/case/prediction.nii.gz
    python -m brainseg.predict --msd4d dataset/imagesTr/BRATS_001.nii.gz --out outputs/BRATS_001/prediction.nii.gz
"""
import argparse
import json
import sys
from pathlib import Path

from .constants import DISCLAIMER, SEQUENCES
from .inference import load_msd_4d, load_sequences, run_case


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in SEQUENCES:
        p.add_argument(f"--{s}", type=Path, help=f"{s.upper()} NIfTI file")
    p.add_argument("--msd4d", type=Path, help="Decathlon Task01 4D image instead of four files")
    p.add_argument("--out", type=Path, required=True, help="Output label map (.nii.gz)")
    p.add_argument("--meta", type=Path, help="Output metadata JSON (default: next to --out)")
    p.add_argument("--no-tta", action="store_true", help="Skip the 3 extra test-time-augmentation passes "
                   "(uncertainty only; the label map is identical either way)")
    args = p.parse_args(argv)

    if args.msd4d:
        files = {"msd4d": args.msd4d}
        stack, ref = load_msd_4d(args.msd4d)
    else:
        files = {s: getattr(args, s) for s in SEQUENCES}
        missing = [s for s, f in files.items() if f is None]
        if missing:
            p.error(f"missing --{' --'.join(missing)} (or use --msd4d)")
        stack, ref = load_sequences(files)

    record = run_case(stack, ref, args.out, files, tta=not args.no_tta)
    meta_path = args.meta or args.out.with_name(args.out.name.split(".")[0] + "_meta.json")
    meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(DISCLAIMER)
    print(f"Prediction: {args.out}  ({record['provenance']['device']}, {record['provenance']['inference_seconds']} s)")
    for name, v in record["estimated_volumes"].items():
        print(f"  {name:>3}: {v['voxels']:>8} voxels x {v['voxel_volume_mm3']:.3f} mm3 = {v['mL']:.2f} mL (estimated)")
    print(f"Metadata:   {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
