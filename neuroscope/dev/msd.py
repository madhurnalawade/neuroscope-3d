"""Read a Decathlon dataset.json and convert one 4D case into the production CanonicalCase.

Channel names and label names are always read from dataset.json, never assumed. Label values are
mapped by NAME to BraTS 2021 codes (0 background, 1 NCR, 2 ED, 4 ET). Files whose names start with
"._" (macOS sidecars) are ignored everywhere. Volumes are read one case at a time via img.dataobj slicing.
"""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np

from ..io.case import CanonicalCase
from ..io.sequences import SEQUENCES, canonical_sequence_name
from ..io.validate import sha256_file
from ..security.hygiene import safe_name

SIDECAR_PREFIX = "._"

# Normalised label name -> BraTS 2021 code.
LABEL_NAME_TO_BRATS = {
    "background": 0,
    "edema": 2, "oedema": 2, "peritumoraledema": 2, "ed": 2,
    "nonenhancingtumor": 1, "nonenhancingtumorcore": 1, "necrosis": 1, "necrotic": 1, "necroticcore": 1,
    "necrotictumorcore": 1, "ncr": 1, "net": 1, "ncrnet": 1, "necroticandnonenhancingtumorcore": 1,
    "enhancingtumor": 4, "et": 4,
}


class DatasetError(ValueError):
    pass


def is_sidecar(name: str) -> bool:
    return Path(name).name.startswith(SIDECAR_PREFIX)


def normalise_label_name(name: str) -> str:
    s = str(name).lower().replace("tumour", "tumor").replace("oedema", "edema")
    return re.sub(r"[^a-z]", "", s)


def label_mapping(labels: Dict[str, str]) -> Dict[int, int]:
    """dataset.json 'labels' ({"0": "background", ...}) -> {source value: BraTS code}, by name."""
    out, seen = {}, {}
    for key, name in labels.items():
        try:
            value = int(key)
        except ValueError:
            raise DatasetError(f"dataset.json label key {key!r} is not an integer")
        norm = normalise_label_name(name)
        if norm not in LABEL_NAME_TO_BRATS:
            raise DatasetError(f"Unknown label name {name!r} in dataset.json; add it to LABEL_NAME_TO_BRATS "
                               "only after checking its meaning.")
        if norm in seen:
            raise DatasetError(f"Label name {name!r} appears twice in dataset.json (values {seen[norm]} and {value}).")
        seen[norm] = value
        out[value] = LABEL_NAME_TO_BRATS[norm]
    if 0 not in out or out[0] != 0:
        raise DatasetError("dataset.json must define value 0 as background.")
    return out


def channel_mapping(modality: Dict[str, str]) -> Dict[int, str]:
    """dataset.json 'modality' ({"0": "FLAIR", "1": "T1w", ...}) -> {channel index: canonical sequence name}."""
    out: Dict[int, str] = {}
    for key, name in modality.items():
        seq = canonical_sequence_name(name)
        if seq is None:
            raise DatasetError(f"Unknown channel name {name!r} in dataset.json 'modality'.")
        if seq in out.values():
            raise DatasetError(f"Two channels in dataset.json map to {seq}.")
        out[int(key)] = seq
    missing = [s for s in SEQUENCES if s not in out.values()]
    if missing or sorted(out) != list(range(len(out))):
        raise DatasetError(f"dataset.json 'modality' must name channels 0..3 as T1, T1Gd, T2 and FLAIR "
                           f"(missing: {missing or 'none'}).")
    return out


@dataclass
class CaseEntry:
    case_id: str
    image: str                    # path relative to the dataset root (POSIX style)
    label: Optional[str]
    subset: str                   # 'training' or 'test'


@dataclass
class DatasetInfo:
    root: Path
    channels: Dict[int, str]
    labels: Dict[str, str]
    label_map: Dict[int, int]
    cases: List[CaseEntry] = field(default_factory=list)
    skipped_sidecars: int = 0

    def find(self, case_id: str) -> CaseEntry:
        for c in self.cases:
            if c.case_id == case_id:
                return c
        raise DatasetError(f"Case {safe_name(case_id)} is not listed in dataset.json.")


def _relative(root: Path, rel: str) -> str:
    """Normalise a dataset.json path ('./imagesTr/x.nii.gz') and refuse anything leaving the root."""
    p = PurePosixPath(str(rel).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", str(rel)):
        raise DatasetError(f"dataset.json path {rel!r} must be relative to the dataset root.")
    return p.as_posix()          # PurePosixPath already drops a leading "./"


def case_id_from(rel: str) -> str:
    name = PurePosixPath(rel).name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def read_dataset(root) -> DatasetInfo:
    root = Path(root).expanduser().resolve()
    js = root / "dataset.json"
    if not js.is_file():
        raise DatasetError("dataset.json not found in the dataset root.")
    meta = json.loads(js.read_text(encoding="utf-8"))
    info = DatasetInfo(root=root, channels=channel_mapping(meta["modality"]), labels=dict(meta["labels"]),
                       label_map=label_mapping(meta["labels"]))
    for entry in meta.get("training", []):
        img, lab = _relative(root, entry["image"]), entry.get("label")
        if is_sidecar(img):
            info.skipped_sidecars += 1
            continue
        info.cases.append(CaseEntry(case_id_from(img), img, _relative(root, lab) if lab else None, "training"))
    for entry in meta.get("test", []):
        img = _relative(root, entry if isinstance(entry, str) else entry["image"])
        if is_sidecar(img):
            info.skipped_sidecars += 1
            continue
        info.cases.append(CaseEntry(case_id_from(img), img, None, "test"))
    return info


def list_nifti(folder: Path) -> Tuple[List[str], int, int]:
    """(real NIfTI file names, sidecar count, raw entry count) in one folder, sorted."""
    if not folder.is_dir():
        return [], 0, 0
    names = [p.name for p in folder.iterdir()]
    side = sum(is_sidecar(n) for n in names)
    real = sorted(n for n in names if not is_sidecar(n) and n.lower().endswith((".nii", ".nii.gz")))
    return real, side, len(names)


def to_brats(mask: np.ndarray, label_map: Dict[int, int]) -> np.ndarray:
    """Map source label values to BraTS 2021 codes; unknown values raise DatasetError."""
    values = np.unique(mask)
    unknown = [int(v) for v in values if int(v) not in label_map or v != int(v)]
    if unknown:
        raise DatasetError(f"Mask contains values {unknown} not defined in dataset.json 'labels'.")
    lut = np.zeros(max(label_map) + 1, dtype=np.uint8)
    for src, dst in label_map.items():
        lut[src] = dst
    return lut[mask.astype(np.int64)]


def load_msd_case(info: DatasetInfo, case_id: str, with_label: bool = True) -> CanonicalCase:
    """One Decathlon 4D case -> CanonicalCase (arrays keyed by sequence name, mask in BraTS codes)."""
    entry = info.find(case_id)
    path = info.root / entry.image
    img = nib.load(str(path), mmap=False)
    if img.ndim != 4 or img.shape[3] != len(info.channels):
        raise DatasetError(f"{safe_name(path.name)} has shape {img.shape}; expected (X, Y, Z, {len(info.channels)}).")
    arrays = {}
    for idx, seq in info.channels.items():
        a = np.asarray(img.dataobj[..., idx])
        if a.dtype.kind in "fc" and not np.all(np.isfinite(a)):
            raise DatasetError(f"{case_id} channel {idx} ({seq}) has non-finite values.")
        if a.min() == a.max():
            raise DatasetError(f"{case_id} channel {idx} ({seq}) is constant.")
        arrays[seq] = a
    affine = np.asarray(img.affine, dtype=np.float64)
    digest = sha256_file(path)
    seg = seg_name = seg_hash = None
    if with_label:
        if entry.label is None:
            raise DatasetError(f"{case_id} has no label in dataset.json.")
        lpath = info.root / entry.label
        limg = nib.load(str(lpath), mmap=False)
        if tuple(limg.shape) != tuple(img.shape[:3]):
            raise DatasetError(f"{case_id}: label shape {limg.shape} != image spatial shape {img.shape[:3]}.")
        if not np.allclose(limg.affine, affine, rtol=0, atol=1e-3):
            raise DatasetError(f"{case_id}: label affine differs from the image affine.")
        seg = to_brats(np.asarray(limg.dataobj), info.label_map)
        seg_name, seg_hash = safe_name(lpath.name), sha256_file(lpath)
    by_index = {seq: idx for idx, seq in info.channels.items()}
    return CanonicalCase(
        arrays=arrays, affine=affine, shape=tuple(int(s) for s in img.shape[:3]),
        spacing=tuple(float(z) for z in img.header.get_zooms()[:3]),
        orientation=tuple(nib.aff2axcodes(affine)),
        file_names={s: f"{safe_name(path.name)}[ch{by_index[s]}]" for s in SEQUENCES},
        sha256={s: digest for s in SEQUENCES},
        seg=seg, seg_file=seg_name, seg_sha256=seg_hash, source="msd-dev",
        notes=["Channel names from dataset.json: "
               + ", ".join(f"{i}={info.channels[i]}" for i in sorted(info.channels)),
               "SHA-256 values are of the single 4D source file."],
    )
