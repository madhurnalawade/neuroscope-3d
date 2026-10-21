"""Delete expired, rejected, failed, abandoned or corrupt upload case folders under the uploads root.

Only folders created by neuroscope.storage (32-hex name + marker file) are ever touched.
  python scripts/cleanup_uploads.py --uploads-root <UPLOADS_ROOT> --dry-run     # list only
  python scripts/cleanup_uploads.py --uploads-root <UPLOADS_ROOT>               # delete
Pass --dataset-root <DATASET_ROOT> too, so the guard can refuse an uploads root that contains the dataset.
Exit code 0 = done, 1 = some deletions failed (listed), 2 = refused to run.
"""
import logging
import sys

from _cli import base_parser, finish, require


def main(argv=None) -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--uploads-root", help="the dedicated uploads root (never the dataset or the repository)")
    ap.add_argument("--dataset-root", help="development dataset location, used only for the safety check")
    ap.add_argument("--dry-run", action="store_true", help="list what would be removed; delete nothing")
    ap.add_argument("--retention-hours", type=float, default=None, help="retention, at most 24 (default 24)")
    ap.add_argument("--allow-cloud-synced-root", action="store_true",
                    help="override the cloud-sync refusal (only if you verified the folder is not synced)")
    args = finish(ap, argv)
    require(ap, args, "uploads_root")

    from neuroscope.security.hygiene import configure_local_logging
    from neuroscope.storage import CloudSyncedRootError, UnsafeRootError, UploadStore
    configure_local_logging(logging.INFO)
    try:
        store = UploadStore(args.uploads_root, retention_hours=args.retention_hours or 24,
                            dataset_root=args.dataset_root, allow_cloud_synced=args.allow_cloud_synced_root,
                            create=False)
    except (UnsafeRootError, CloudSyncedRootError, ValueError) as exc:
        print(f"REFUSED: {exc}")
        return 2
    if not store.root.is_dir():
        print("Uploads root does not exist yet; nothing to do.")
        return 0
    rep = store.sweep(dry_run=args.dry_run)
    if args.dry_run:
        print(f"DRY RUN - nothing deleted. Would delete {len(rep.would_delete)} case folder(s):")
        for cid, why in rep.would_delete:
            print(f"  {cid}  ({why})")
    else:
        print(f"Deleted {len(rep.deleted)} case folder(s):")
        for cid, why in rep.deleted:
            print(f"  {cid}  ({why})")
    print(f"Kept (not expired): {len(rep.kept)}")
    for name, why in rep.skipped:
        print(f"  left untouched: {name[:40]}  ({why})")
    for cid, err in rep.failures:
        print(f"  FAILED: {cid}  ({err}) - close any program using these files and run again")
    return 1 if rep.failures else 0


if __name__ == "__main__":
    sys.exit(main())
