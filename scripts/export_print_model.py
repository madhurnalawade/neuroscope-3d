"""Print-ready clear brain shell: a clear brain with a tumour cavity plus separate nested tumour parts
(edema_green, enhancing_yellow, necrotic_red) as one 3MF with named, coloured parts, per-part STLs, a preview
PNG and a summary table. Separate from the viewer's layered STL export, which is unchanged.

Usage:
    python scripts/export_print_model.py --backup --no-network
    python scripts/export_print_model.py --backup --mode split_halves --scale 1.2 --no-network
    python scripts/export_print_model.py --study-id <ID> --source expert --out outputs/print/my_case

Defaults come from the config block in app/services/print_export.py (PrintConfig).
Research and teaching prototype; model-predicted regions; not for clinical use.
"""
import sys
from dataclasses import replace
from pathlib import Path

from _cli import REPO_ROOT, base_parser, finish

sys.path.insert(0, str(REPO_ROOT / "Input" / "neuroscope_3d" / "backend"))


def main(argv=None) -> int:
    ap = base_parser(__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--backup", action="store_true", help="use the precomputed backup case (backup_case/)")
    src.add_argument("--study-id", help="an uploaded study that is still on disk")
    ap.add_argument("--output-dir", "--out", dest="output_dir", type=Path,
                    help="output folder (default outputs/print/<case>_<mode>)")
    ap.add_argument("--mode", choices=("solid_clear", "split_halves"))
    ap.add_argument("--scale", type=float)
    ap.add_argument("--clearance", type=float, help="mm gap around nested parts")
    ap.add_argument("--smooth-iterations", type=int)
    ap.add_argument("--source", choices=("pred", "expert"), help="model prediction (default) or expert mask")
    ap.add_argument("--build-volume", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--no-marker", action="store_true", help="leave out the raised L on the left side")
    args = finish(ap, argv)

    try:
        from app.services import print_export
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    from app.services.study_store import study_store

    cfg = print_export.CONFIG
    overrides = {"MODE": args.mode, "SCALE": args.scale, "CLEARANCE": args.clearance,
                 "SMOOTH_ITERATIONS": args.smooth_iterations, "SOURCE": args.source,
                 "BUILD_VOLUME": tuple(args.build_volume) if args.build_volume else None}
    cfg = replace(cfg, **{k: v for k, v in overrides.items() if v is not None}, MARKER=not args.no_marker)

    temp_study = None
    try:
        if args.backup:
            from app.api.backup import BACKUP_DIR, load_backup_study
            if not (BACKUP_DIR / "backup.json").exists():
                print("error: no backup case. Run: python scripts/make_backup_case.py demo_cases/BRATS_001",
                      file=sys.stderr)
                return 2
            temp_study = load_backup_study()
            study = study_store.get_validated_study(temp_study)
            case = "backup"
        else:
            study = study_store.get_validated_study(args.study_id)
            case = args.study_id
        out = args.output_dir or REPO_ROOT / "outputs" / "print" / f"{case}_{cfg.MODE}"
        print(f"Building print parts ({cfg.MODE}, SCALE {cfg.SCALE:g}, source {cfg.SOURCE}) ...", flush=True)
        result = print_export.export_study(study, out, cfg, case_name=case)
    except print_export.PrintExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 - study lookup errors carry their own message
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        if temp_study:
            study_store.delete_study(temp_study)

    print(result["text"])
    print(f"\nWrote {result['out_dir']} (3MF, stl/, preview.png, summary.txt/json, transforms.json)")
    print("Research and teaching prototype; model-predicted regions; not for clinical use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
