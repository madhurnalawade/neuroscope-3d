"""List every installed Python distribution (like 'pip list', without running pip) and, for the packages
this input layer needs, what is known about network access, telemetry and update checks.

  python scripts/list_dependencies.py

Claims below are limited to what we could state with confidence; everything else says "not verified".
"""
import sys
from importlib import metadata

NOTES = {
    "numpy": "No network access, telemetry or update checks in normal use.",
    "nibabel": "Reads/writes local files; no telemetry or update check in the functions this layer uses "
               "(checked by tests/test_pipeline_offline.py under the network guard).",
    "pytest": "No telemetry or update check. Writes a local .pytest_cache folder (test names only). "
              "Third-party plugins may behave differently: check any you add.",
    "pip": "Contacts the package index when installing and runs a self-version check against PyPI; disable the "
           "check with PIP_DISABLE_PIP_VERSION_CHECK=1. Installing never uploads scan data.",
    "huggingface_hub": "Contacts huggingface.co when downloading; set HF_HUB_OFFLINE=1 and "
                       "HF_HUB_DISABLE_TELEMETRY=1 once the weights are cached. Not needed by this input layer.",
    "monai": "monai.bundle.download fetches model weights over the network when called. Not needed by this layer.",
    "opentelemetry-api": "API package only; it sends nothing unless an SDK and exporter are configured. "
                         "Not needed by this layer; arrives as a dependency of other packages.",
}
REQUIRED = ("nibabel", "numpy", "pytest")


def main() -> int:
    dists = sorted({(d.metadata["Name"] or "?", d.version) for d in metadata.distributions()},
                   key=lambda x: x[0].lower())
    print(f"Python {sys.version.split()[0]}; {len(dists)} installed distributions:")
    for name, version in dists:
        print(f"  {name}=={version}")
    print("\nPackages required by the neuroscope input layer:")
    installed = {n.lower(): v for n, v in dists}
    for pkg in REQUIRED:
        print(f"  {pkg:<10} {installed.get(pkg, 'NOT INSTALLED'):<10} {NOTES[pkg]}")
    print("\nNotes on other installed packages:")
    for pkg, note in NOTES.items():
        if pkg not in REQUIRED and pkg.lower() in installed:
            print(f"  {pkg}: {note}")
    print("\nEvery other package: not verified. The network-guard tests show only that the code paths they "
          "exercise opened no connection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
