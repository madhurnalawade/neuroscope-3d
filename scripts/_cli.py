"""Shared helpers for the neuroscope command-line scripts: repo import path, optional JSON config file,
and the opt-in network guard. Every location comes from arguments or the config file, never from defaults.

Config file (JSON, all keys optional), passed with --config <FILE>:
    {"dataset_root": "...", "uploads_root": "...", "output_dir": "...", "retention_hours": 24}
Command-line arguments override the config file.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_KEYS = {"dataset_root", "uploads_root", "output_dir", "retention_hours"}


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="optional JSON config file with dataset_root / uploads_root / "
                                                "output_dir / retention_hours")
    ap.add_argument("--no-network", action="store_true",
                    help="block every network connection and DNS lookup in this process (recommended)")
    return ap


def finish(ap: argparse.ArgumentParser, argv=None) -> argparse.Namespace:
    """Parse, merge the config file (arguments win), and enable the network guard if requested."""
    args = ap.parse_args(argv)
    if args.config:
        try:
            cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            ap.error(f"cannot read --config file ({type(exc).__name__})")
        unknown = set(cfg) - CONFIG_KEYS
        if unknown:
            ap.error(f"unknown config keys: {', '.join(sorted(unknown))}")
        for key, val in cfg.items():
            if hasattr(args, key) and getattr(args, key) in (None, False):
                setattr(args, key, Path(val) if key.endswith(("_root", "_dir")) else val)
    if args.no_network:
        from neuroscope.security import netguard
        netguard.enable()
    return args


def require(ap, args, *names):
    for n in names:
        if getattr(args, n, None) in (None, ""):
            ap.error(f"--{n.replace('_', '-')} is required (or set it in --config)")
