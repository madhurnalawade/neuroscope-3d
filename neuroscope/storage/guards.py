"""Safety checks for the uploads root: never cloud-synced, never a dangerous location."""
import os
from pathlib import Path
from typing import List, Optional

# Lower-case path components (or component prefixes) used by common sync clients.
CLOUD_MARKERS = (
    "onedrive", "dropbox", "google drive", "googledrive", "my drive", "icloud", "icloud drive",
    "iclouddrive", "mobile documents", "clouddocs", "cloudstorage", "box sync", "box", "pcloud", "mega",
    "megasync", "sync.com", "nextcloud", "owncloud", "seafile", "tresorit", "yandex.disk",
    "creative cloud files", "spideroak", "syncthing",
)
# Environment variables that name a synced folder on Windows.
CLOUD_ENV_VARS = ("OneDrive", "OneDriveConsumer", "OneDriveCommercial")


class UnsafeRootError(RuntimeError):
    """The uploads root is a location where cleanup could delete the wrong thing."""


class CloudSyncedRootError(RuntimeError):
    """The uploads root looks like it is inside a cloud-synced folder."""


def _component_hits(path: Path) -> List[str]:
    hits = []
    for part in path.parts:
        low = part.lower().strip()
        for marker in CLOUD_MARKERS:
            if low == marker or (len(marker) > 4 and low.startswith(marker)):
                hits.append(part)
                break
    return hits


def cloud_sync_reasons(path) -> List[str]:
    """Reasons to believe ``path`` is synced to a cloud service (empty list: none found by these heuristics).
    Checks the path as given and its resolved form, plus Windows OneDrive environment variables."""
    raw = Path(path).expanduser()
    reasons = [f"path component '{p}' looks like a sync folder" for p in _component_hits(raw)]
    try:
        resolved = raw.resolve()
    except OSError:
        resolved = raw
    reasons += [f"resolved path component '{p}' looks like a sync folder" for p in _component_hits(resolved)
                if f"path component '{p}' looks like a sync folder" not in reasons]
    for var in CLOUD_ENV_VARS:
        val = os.environ.get(var)
        if val:
            try:
                base = Path(val).resolve()
                if resolved == base or base in resolved.parents:
                    reasons.append(f"inside the folder named by %{var}%")
            except OSError:
                pass
    return reasons


def _is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def check_uploads_root(root, dataset_root: Optional[os.PathLike] = None,
                       repo_root: Optional[os.PathLike] = None) -> Path:
    """Return the resolved uploads root or raise UnsafeRootError. Does not create anything."""
    if root is None or str(root).strip() in ("", "."):
        raise UnsafeRootError("The uploads root is empty. Pass --uploads-root <UPLOADS_ROOT> explicitly.")
    raw = Path(str(root)).expanduser()
    if not raw.is_absolute():
        raw = Path.cwd() / raw
    if raw.exists() and raw.is_symlink():
        raise UnsafeRootError("The uploads root is a symbolic link; give its real location instead.")
    resolved = raw.resolve()
    if resolved == Path(resolved.anchor) or resolved.parent == resolved:
        raise UnsafeRootError("The uploads root is a filesystem root.")
    home = Path.home().resolve()
    if resolved == home or _is_within(home, resolved):
        raise UnsafeRootError("The uploads root is the home directory or contains it.")
    if dataset_root is not None:
        ds = Path(dataset_root).expanduser().resolve()
        if _is_within(ds, resolved):
            raise UnsafeRootError("The uploads root equals or contains the dataset root.")
        if _is_within(resolved, ds):
            raise UnsafeRootError("The uploads root is inside the dataset root.")
    if repo_root is not None:
        repo = Path(repo_root).expanduser().resolve()
        if _is_within(repo, resolved):
            raise UnsafeRootError("The uploads root equals or contains the repository.")
    return resolved
