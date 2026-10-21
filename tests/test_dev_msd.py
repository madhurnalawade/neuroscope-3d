"""Decathlon adapter (development only) on tiny synthetic 4D cases with a synthetic dataset.json."""
import json

import numpy as np
import pytest

from conftest import make_msd
from neuroscope.dev.audit import audit_dataset
from neuroscope.dev.export import export_case, verify_export
from neuroscope.dev.manifest import build_manifest
from neuroscope.dev.msd import DatasetError, label_mapping, load_msd_case, read_dataset, to_brats
from neuroscope.dev.split import make_split
from neuroscope.io import load_four_files


def test_channel_names_come_from_dataset_json(tmp_path):
    # deliberately unusual order: channel 0 = T2, 1 = T1Gd, 2 = FLAIR, 3 = T1
    root = make_msd(tmp_path / "ds", modality={"0": "T2w", "1": "T1-Gd", "2": "FLAIR", "3": "T1w"})
    case = load_msd_case(read_dataset(root), "BRATS_001")
    # make_msd offsets channel c by 100*(c+1): the mean tells which source channel each array came from
    means = {seq: float(a.mean()) for seq, a in case.arrays.items()}
    assert 550 < means["T2"] < 650 and 650 < means["T1Gd"] < 750
    assert 750 < means["FLAIR"] < 850 and 850 < means["T1"] < 950
    assert case.source == "msd-dev" and case.seg is not None


def test_standard_decathlon_order(tmp_path):
    case = load_msd_case(read_dataset(make_msd(tmp_path / "ds")), "BRATS_002")
    assert 550 < case.arrays["FLAIR"].mean() < 650 and 850 < case.arrays["T2"].mean() < 950


@pytest.mark.parametrize("modality", [
    {"0": "FLAIR", "1": "T1w", "2": "T1w", "3": "T2w"},          # duplicate
    {"0": "FLAIR", "1": "T1w", "2": "DWI", "3": "T2w"},          # unknown
    {"0": "FLAIR", "1": "T1w", "2": "t1gd"},                      # missing
])
def test_bad_channel_names_rejected(tmp_path, modality):
    root = make_msd(tmp_path / "ds", modality=modality)
    with pytest.raises(DatasetError):
        read_dataset(root)


def test_label_names_mapped_to_brats_codes():
    m = label_mapping({"0": "background", "1": "edema", "2": "non-enhancing tumor", "3": "enhancing tumour"})
    assert m == {0: 0, 1: 2, 2: 1, 3: 4}
    # same names, different values: mapping follows the names, not the numbers
    m2 = label_mapping({"0": "background", "1": "Enhancing Tumor", "2": "Peritumoral Edema", "3": "necrotic"})
    assert m2 == {0: 0, 1: 4, 2: 2, 3: 1}
    mask = np.array([[[0, 1, 2, 3]]], dtype=np.uint8)
    assert to_brats(mask, m).tolist() == [[[0, 2, 1, 4]]]


@pytest.mark.parametrize("labels", [
    {"0": "background", "1": "tumour"},                            # unknown name
    {"0": "background", "1": "edema", "2": "Edema"},               # duplicate name
    {"0": "edema", "1": "background"},                             # 0 is not background
])
def test_bad_label_names_rejected(labels):
    with pytest.raises(DatasetError):
        label_mapping(labels)


def test_mask_value_not_in_labels_rejected():
    with pytest.raises(DatasetError):
        to_brats(np.array([[[0, 5]]], dtype=np.uint8), {0: 0, 1: 2})


def test_mask_converted_by_name(tmp_path):
    case = load_msd_case(read_dataset(make_msd(tmp_path / "ds")), "BRATS_001")
    assert set(np.unique(case.seg)) == {0, 1, 2, 4}
    assert case.seg[4, 4, 4] == 4 and case.seg[3, 3, 3] == 1 and case.seg[2, 2, 2] == 2


def test_sidecar_files_ignored(tmp_path):
    root = make_msd(tmp_path / "ds", n_cases=3)
    js = json.loads((root / "dataset.json").read_text())
    js["training"].append({"image": "./imagesTr/._BRATS_001.nii.gz", "label": "./labelsTr/._BRATS_001.nii.gz"})
    (root / "dataset.json").write_text(json.dumps(js))
    info = read_dataset(root)
    assert [c.case_id for c in info.cases if c.subset == "training"] == ["BRATS_001", "BRATS_002", "BRATS_003"]
    assert info.skipped_sidecars == 1
    man = build_manifest(root)
    c = man["counts"]
    assert c["imagesTr"] == {"raw_entries": 6, "macos_sidecars_ignored": 3, "nifti_files": 3}
    assert c["manifest_cases"] == 3 and not any(c["unmatched"].values())
    assert all(not r["image"].split("/")[-1].startswith("._") for r in man["cases"])


def test_manifest_relative_paths_and_headers(tmp_path):
    root = make_msd(tmp_path / "ds")
    man = build_manifest(root, require_labels=False)
    assert str(tmp_path) not in json.dumps(man)                      # no absolute paths
    rows = {r["case_id"]: r for r in man["cases"]}
    assert rows["BRATS_001"]["image"] == "imagesTr/BRATS_001.nii.gz"
    assert rows["BRATS_001"]["shape"] == [12, 10, 8, 4] and rows["BRATS_001"]["label_shape"] == [12, 10, 8]
    assert rows["BRATS_900"]["label"] is None and rows["BRATS_900"]["subset"] == "test"
    assert man["channels"] == {"0": "FLAIR", "1": "T1", "2": "T1Gd", "3": "T2"}


def test_dataset_json_paths_cannot_escape_root(tmp_path):
    root = make_msd(tmp_path / "ds")
    js = json.loads((root / "dataset.json").read_text())
    js["training"][0]["image"] = "../../secret.nii.gz"
    (root / "dataset.json").write_text(json.dumps(js))
    with pytest.raises(DatasetError):
        read_dataset(root)


def test_export_roundtrip_matches_exactly(tmp_path):
    root = make_msd(tmp_path / "ds", modality={"0": "T2w", "1": "T1-Gd", "2": "FLAIR", "3": "T1w"})
    info = read_dataset(root)
    case = load_msd_case(info, "BRATS_003")
    paths = export_case(info, case, "BRATS_003", tmp_path / "out")
    assert sorted(p.name for p in paths.values()) == ["BRATS_003_flair.nii.gz", "BRATS_003_seg.nii.gz",
                                                      "BRATS_003_t1.nii.gz", "BRATS_003_t1gd.nii.gz",
                                                      "BRATS_003_t2.nii.gz"]
    assert verify_export(case, paths) == []
    # the exported names alone are enough for the production loader to assign sequences correctly
    res = load_four_files(unlabelled=[paths[s] for s in ("FLAIR", "T1", "T1Gd", "T2")], seg=paths["seg"])
    assert res.ok and np.array_equal(res.case.arrays["T1Gd"], case.arrays["T1Gd"])


def test_export_refuses_dataset_dir_and_overwrite(tmp_path):
    root = make_msd(tmp_path / "ds")
    info = read_dataset(root)
    case = load_msd_case(info, "BRATS_001")
    with pytest.raises(DatasetError):
        export_case(info, case, "BRATS_001", root / "exports")
    export_case(info, case, "BRATS_001", tmp_path / "out")
    with pytest.raises(DatasetError):
        export_case(info, case, "BRATS_001", tmp_path / "out")


def test_split_reproducible_and_patient_caveat(tmp_path):
    root = make_msd(tmp_path / "ds", n_cases=14, sidecars=False)
    a = make_split(root, n_heldout=10, n_dev=2, seed=7)
    b = make_split(root, n_heldout=10, n_dev=2, seed=7)
    assert a["heldout"] == b["heldout"] and len(a["heldout"]) == 10
    assert not set(a["heldout"]) & set(a["train"]) and not set(a["dev"]) & set(a["train"])
    assert "cannot be confirmed" in a["patient_level_caveat"] and "BraTS 2018" in a["pretraining_overlap_reminder"]
    with pytest.raises(ValueError):
        make_split(root, n_heldout=9)


def test_audit_collects_failures_and_continues(tmp_path):
    root = make_msd(tmp_path / "ds", n_cases=3)
    (root / "labelsTr" / "BRATS_002.nii.gz").write_bytes(b"broken")
    rep = audit_dataset(root, progress=None)
    assert rep["checked"] == 3 and rep["failed"] == 1 and rep["failed_case_ids"] == ["BRATS_002"]
