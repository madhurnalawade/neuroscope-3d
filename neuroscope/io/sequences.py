"""Sequence names and unambiguous assignment of files to sequences.

Explicit assignment always wins. Otherwise each file name is split into lower-case alphanumeric tokens
and matched against whole-token aliases (adjacent token pairs such as "t1_gd" are joined first), so
"t1gd" never also counts as "t1". A file matching zero or several sequences, two files claiming the same
sequence, or a missing sequence is refused with a request for explicit assignment. Nothing is guessed.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..security.hygiene import safe_name
from . import errors as E

# Canonical names. This is a set of keys, NOT a model channel order.
SEQUENCES = ("FLAIR", "T1", "T1Gd", "T2")

ALIASES = {
    "T1": {"t1", "t1w", "t1weighted"},
    "T1Gd": {"t1gd", "t1ce", "t1c", "t1post", "t1wce", "t1gad", "t1contrast"},
    "T2": {"t2", "t2w", "t2weighted"},
    "FLAIR": {"flair", "t2flair", "t2wflair"},
}
_TOKEN_TO_SEQ = {alias: seq for seq, names in ALIASES.items() for alias in names}
_NIFTI_SUFFIXES = (".nii.gz", ".nii")


def canonical_sequence_name(name: str) -> Optional[str]:
    """Map a whole label such as 'T1w', 't1gd', 'T1-CE' or 'FLAIR' to a canonical name, else None."""
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return _TOKEN_TO_SEQ.get(key)


def _stem(filename: str) -> str:
    base = Path(str(filename).replace("\\", "/")).name
    low = base.lower()
    for suf in _NIFTI_SUFFIXES:
        if low.endswith(suf):
            return low[: -len(suf)]
    return low


def sequences_in_filename(filename: str) -> List[str]:
    """All sequences named in a file name (sorted, de-duplicated). One entry means unambiguous."""
    tokens = [t for t in re.split(r"[^a-z0-9]+", _stem(filename)) if t]
    found, i = set(), 0
    while i < len(tokens):
        pair = tokens[i] + tokens[i + 1] if i + 1 < len(tokens) else None
        if pair in _TOKEN_TO_SEQ:
            found.add(_TOKEN_TO_SEQ[pair])
            i += 2
        else:
            if tokens[i] in _TOKEN_TO_SEQ:
                found.add(_TOKEN_TO_SEQ[tokens[i]])
            i += 1
    return sorted(found)


@dataclass
class Assignment:
    ok: bool
    files: Dict[str, Path] = field(default_factory=dict)   # sequence -> path
    issues: List[E.Issue] = field(default_factory=list)
    inferred: Dict[str, bool] = field(default_factory=dict)  # sequence -> True if taken from the file name


def assign_sequences(explicit: Optional[Dict[str, object]] = None,
                     unlabelled: Iterable[object] = ()) -> Assignment:
    """Build the sequence -> file mapping.

    ``explicit``: {'T1': path, 'T1Gd': path, ...} (keys may use any alias, e.g. 't1ce'); these always win.
    ``unlabelled``: further paths whose sequence is inferred from the file name.
    """
    issues: List[E.Issue] = []
    files: Dict[str, Path] = {}
    inferred: Dict[str, bool] = {}

    for key, path in (explicit or {}).items():
        if path is None:
            continue
        seq = canonical_sequence_name(key)
        if seq is None:
            issues.append(E.issue(E.UNKNOWN_SEQUENCE, f"'{safe_name(key)}' is not a sequence name; use T1, T1Gd, T2 or FLAIR."))
            continue
        if seq in files:
            issues.append(E.issue(E.DUPLICATE_SEQUENCE, f"{seq} was given twice. Assign each sequence exactly once."))
            continue
        files[seq], inferred[seq] = Path(path), False

    for path in unlabelled:
        path = Path(path)
        found = sequences_in_filename(path.name)
        short = safe_name(path.name)
        if len(found) != 1:
            what = "no sequence name" if not found else "several sequence names (" + ", ".join(found) + ")"
            issues.append(E.issue(E.AMBIGUOUS_SEQUENCE,
                                  f"Cannot tell which sequence '{short}' is: its name contains {what}. "
                                  "Assign it explicitly (--t1 / --t1gd / --t2 / --flair).", file=short))
            continue
        seq = found[0]
        if seq in files:
            issues.append(E.issue(E.DUPLICATE_SEQUENCE,
                                  f"'{short}' looks like {seq}, but {seq} is already assigned to "
                                  f"'{safe_name(files[seq].name)}'. Assign the files explicitly.", file=short))
            continue
        files[seq], inferred[seq] = path, True

    resolved = [p.resolve() for p in files.values()]
    if len(set(resolved)) != len(resolved):
        issues.append(E.issue(E.DUPLICATE_SEQUENCE, "The same file was assigned to more than one sequence."))

    missing = [s for s in SEQUENCES if s not in files]
    if missing:
        issues.append(E.issue(E.MISSING_SEQUENCE,
                              f"Four sequences are required (T1, T1Gd, T2, FLAIR); missing: {', '.join(missing)}.",
                              missing=missing))

    return Assignment(ok=not issues, files=files if not issues else {}, issues=issues, inferred=inferred)
