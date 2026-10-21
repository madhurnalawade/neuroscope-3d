"""Manifest of real Decathlon cases from headers only (no voxel data is read)."""
import datetime
import json
from pathlib import Path

import nibabel as nib

from .msd import case_id_from, list_nifti, read_dataset


def _header(path: Path):
    img = nib.load(str(path), mmap=False)        # parses the header; voxel data is not read
    return [int(s) for s in img.shape], [round(float(z), 6) for z in img.header.get_zooms()[:3]]


def build_manifest(dataset_root, require_labels: bool = True, progress=None) -> dict:
    info = read_dataset(dataset_root)
    root = info.root
    counts = {}
    on_disk = {}
    for sub in ("imagesTr", "labelsTr", "imagesTs"):
        real, side, raw = list_nifti(root / sub)
        on_disk[sub] = {case_id_from(n) for n in real}
        counts[sub] = {"raw_entries": raw, "macos_sidecars_ignored": side, "nifti_files": len(real)}

    listed_train = {c.case_id for c in info.cases if c.subset == "training"}
    cases, missing = [], []
    selected = [c for c in info.cases if c.label or not require_labels]
    for i, c in enumerate(selected, 1):
        img_path = root / c.image
        lab_path = root / c.label if c.label else None
        if not img_path.is_file() or (lab_path is not None and not lab_path.is_file()):
            missing.append(c.case_id)
            continue
        shape, spacing = _header(img_path)
        row = {"case_id": c.case_id, "subset": c.subset, "image": c.image, "label": c.label,
               "image_bytes": img_path.stat().st_size,
               "label_bytes": lab_path.stat().st_size if lab_path else None,
               "shape": shape, "spacing_mm": spacing}
        if lab_path is not None:
            row["label_shape"], _ = _header(lab_path)
        cases.append(row)
        if progress:
            progress(i, len(selected), c.case_id)

    counts["dataset_json"] = {
        "training_listed": len(listed_train),
        "test_listed": sum(c.subset == "test" for c in info.cases),
        "sidecar_entries_ignored": info.skipped_sidecars,
    }
    counts["unmatched"] = {
        "images_without_label_on_disk": sorted(on_disk["imagesTr"] - on_disk["labelsTr"]),
        "labels_without_image_on_disk": sorted(on_disk["labelsTr"] - on_disk["imagesTr"]),
        "listed_but_missing_on_disk": sorted(missing),
        "on_disk_but_not_listed": sorted(on_disk["imagesTr"] - listed_train),
    }
    counts["manifest_cases"] = len(cases)
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_root_name": root.name,            # name only; absolute paths are never written
        "paths_relative_to": "dataset root",
        "require_labels": require_labels,
        "channels": {str(k): v for k, v in sorted(info.channels.items())},
        "labels_dataset_json": info.labels,
        "label_map_to_brats2021": {str(k): v for k, v in sorted(info.label_map.items())},
        "counts": counts,
        "cases": cases,
    }


def write_json(obj: dict, out_dir, name: str) -> Path:
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    return path
