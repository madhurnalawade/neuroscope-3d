"""Precompute the clearly labelled BACKUP case (used only if the live pipeline cannot be shown).

    python scripts/make_backup_case.py demo_cases/BRATS_001

Runs the normal pipeline once on the four sequence files in the folder (and keeps the *_seg.nii.gz expert mask
if present) and stores inputs + outputs in backup_case/. The app serves it at /backup with a
"PRECOMPUTED BACKUP CASE" banner on every screen. It is never used for judges' uploads.
"""
import datetime
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brainseg.inference import load_sequences, run_case  # noqa: E402

SUFFIX = {"t1": "_t1.nii.gz", "t1gd": "_t1ce.nii.gz", "t2": "_t2.nii.gz", "flair": "_flair.nii.gz"}


def main(case_dir: Path):
    out = ROOT / "backup_case"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()
    files = {}
    for seq, suf in SUFFIX.items():
        src = next(case_dir.glob(f"*{suf}"))
        files[seq] = out / src.name
        shutil.copy2(src, files[seq])
    seg = next(case_dir.glob("*_seg.nii.gz"), None)
    if seg:
        shutil.copy2(seg, out / "expert.nii.gz")
    stack, ref = load_sequences(files)
    record = run_case(stack, ref, out / "prediction.nii.gz", files)
    (out / "prediction_meta.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out / "backup.json").write_text(json.dumps({
        "label": "PRECOMPUTED BACKUP CASE",
        "case": f"{case_dir.name} (Medical Segmentation Decathlon Task01, public research data)",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": {k: v.name for k, v in files.items()},
        "expert_mask": seg is not None,
    }, indent=2), encoding="utf-8")
    print(f"Backup case written to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
