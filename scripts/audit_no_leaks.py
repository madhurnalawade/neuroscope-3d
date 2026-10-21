"""Static leak audit of this repository (no network needed, nothing is imported or executed).

  python scripts/audit_no_leaks.py [--root <REPO_ROOT>]

Prints every hit with file and line as ALLOWED (with reason) or PROBLEM, then PASS or FAIL.
Exit code 0 = PASS, 1 = FAIL.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from neuroscope.security.static_audit import run_audit  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                    help="repository root to scan (default: the repository containing this script)")
    args = ap.parse_args(argv)
    hits = run_audit(args.root)
    problems = [h for h in hits if not h.ok]
    for h in sorted(hits, key=lambda h: (h.ok, h.path, h.line)):
        tag = "ALLOWED" if h.ok else "PROBLEM"
        print(f"{tag:<8} {h.path}:{h.line}  [{h.category}] {h.detail}")
        if h.ok:
            print(f"         reason: {h.allowed_reason}")
    print()
    print(f"{len(hits)} hit(s): {len(hits) - len(problems)} allowed, {len(problems)} problem(s).")
    print("Not covered: native code, other processes, the browser, the OS, and anything not written in this "
          "repository's Python/HTML/JS/CSS files (see neuroscope/README.md).")
    print("PASS" if not problems else "FAIL")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
