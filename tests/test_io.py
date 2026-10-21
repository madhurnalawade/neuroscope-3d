"""Core four-file path: assignment, per-file validation, cross-file checks, canonical case."""
import gzip
from unittest import mock

import numpy as np
import pytest

from conftest import AFFINE, SHAPE, save, volume
from neuroscope.io import errors as E
from neuroscope.io import load_four_files, validate_file
from neuroscope.io.sequences import assign_sequences, sequences_in_filename


def codes(res):
    return [i.code for i in res.issues]


# ------------------------------------------------------------------ assignment

@pytest.mark.parametrize("name,expected", [
    ("BRATS_002_t1ce.nii.gz", ["T1Gd"]), ("sub01_T1GD.nii", ["T1Gd"]), ("x_t1_gd.nii.gz", ["T1Gd"]),
    ("BraTS2021_00001_t1.nii.gz", ["T1"]), ("scan-T1w.nii.gz", ["T1"]), ("a_t2.nii.gz", ["T2"]),
    ("a_T2_FLAIR.nii.gz", ["FLAIR"]), ("flair.nii.gz", ["FLAIR"]), ("t1gd_and_t2.nii.gz", ["T1Gd", "T2"]),
    ("brain.nii.gz", []), ("t10.nii.gz", []),
])
def test_filename_inference_is_whole_token(name, expected):
    assert sequences_in_filename(name) == expected


def test_valid_set(four_files):
    res = load_four_files(unlabelled=four_files.values())
    assert res.ok, res.issues
    c = res.case
    assert set(c.arrays) == {"FLAIR", "T1", "T1Gd", "T2"}
    assert c.shape == SHAPE and c.spacing == (1.0, 1.0, 1.0) and "".join(c.orientation) == "LAS"
    assert np.array_equal(c.affine, AFFINE)
    assert all(len(h) == 64 for h in c.sha256.values())
    assert np.array_equal(c.arrays["T1Gd"], volume(2))          # keyed by name, not by position


def test_explicit_assignment_overrides_filenames(tmp_path):
    # file names say the opposite of the truth; explicit flags must win
    t1 = save(tmp_path / "looks_like_t2.nii.gz", volume(1))
    t2 = save(tmp_path / "looks_like_t1.nii.gz", volume(2))
    gd = save(tmp_path / "x_flair.nii.gz", volume(3))
    fl = save(tmp_path / "y_t1gd.nii.gz", volume(4))
    res = load_four_files(explicit={"t1": t1, "t2": t2, "t1gd": gd, "flair": fl})
    assert res.ok, res.issues
    assert np.array_equal(res.case.arrays["T1"], volume(1))
    assert np.array_equal(res.case.arrays["FLAIR"], volume(4))


def test_ambiguous_filename_refused(four_files, tmp_path):
    odd = save(tmp_path / "scan_t1_t2.nii.gz", volume(9))
    res = load_four_files(unlabelled=[four_files["FLAIR"], four_files["T1Gd"], four_files["T2"], odd])
    assert not res.ok and E.AMBIGUOUS_SEQUENCE in codes(res) and res.case is None


def test_unrecognised_filename_refused(four_files, tmp_path):
    odd = save(tmp_path / "brain_scan.nii.gz", volume(9))
    res = load_four_files(unlabelled=[four_files["FLAIR"], four_files["T1Gd"], four_files["T2"], odd])
    assert E.AMBIGUOUS_SEQUENCE in codes(res)


def test_duplicate_sequence_refused(four_files, tmp_path):
    dup = save(tmp_path / "other_t2.nii.gz", volume(9))
    res = load_four_files(unlabelled=[four_files["FLAIR"], four_files["T1Gd"], four_files["T2"], dup])
    assert E.DUPLICATE_SEQUENCE in codes(res)


def test_explicit_and_inferred_conflict_refused(four_files):
    res = load_four_files(explicit={"t2": four_files["T1"]}, unlabelled=four_files.values())
    assert E.DUPLICATE_SEQUENCE in codes(res)


def test_three_files_only(four_files):
    res = load_four_files(unlabelled=[four_files["FLAIR"], four_files["T1"], four_files["T2"]])
    assert not res.ok and codes(res) == [E.MISSING_SEQUENCE] and "T1Gd" in res.issues[0].message


# ------------------------------------------------------------------ cross-file

def test_mismatched_shapes(four_files):
    save(four_files["T2"], volume(3, shape=(12, 10, 7)))
    res = load_four_files(unlabelled=four_files.values())
    assert codes(res) == [E.SHAPE_MISMATCH]
    assert "T2" in res.issues[0].message and "(12, 10, 7)" in res.issues[0].message


def test_same_shape_different_affine(four_files):
    shifted = AFFINE.copy()
    shifted[0, 3] += 2.5
    save(four_files["T1"], volume(1), affine=shifted)
    res = load_four_files(unlabelled=four_files.values())
    assert codes(res) == [E.AFFINE_MISMATCH]
    assert "T1" in res.issues[0].message and "origin differs" in res.issues[0].message


def test_flipped_affine_reports_orientation(four_files):
    flipped = AFFINE.copy()
    flipped[0, 0] = 1.0
    save(four_files["FLAIR"], volume(0), affine=flipped)
    res = load_four_files(unlabelled=four_files.values())
    assert all(c == E.AFFINE_MISMATCH for c in codes(res)) and "orientation" in res.issues[0].message


def test_different_spacing(four_files):
    aff = AFFINE.copy()
    aff[2, 2] = 2.0
    save(four_files["T2"], volume(3), affine=aff)
    res = load_four_files(unlabelled=four_files.values())
    assert E.SPACING_MISMATCH in codes(res) and E.AFFINE_MISMATCH in codes(res)


# ------------------------------------------------------------------ per file

def test_2d_image(four_files):
    save(four_files["T1"], volume(1, shape=(12, 10)))
    assert codes(load_four_files(unlabelled=four_files.values())) == [E.NOT_3D]


def test_single_slice_counts_as_2d(tmp_path):
    assert validate_file(save(tmp_path / "s_t1.nii.gz", volume(1, shape=(12, 10, 1)))).code == E.NOT_3D


def test_4d_image_as_single_sequence(four_files):
    save(four_files["T1Gd"], np.stack([volume(1), volume(2)], -1))
    res = load_four_files(unlabelled=four_files.values())
    assert codes(res) == [E.NOT_3D] and "own 3D file" in res.issues[0].message


def test_all_zero_volume(four_files):
    save(four_files["FLAIR"], np.zeros(SHAPE, np.float32))
    assert codes(load_four_files(unlabelled=four_files.values())) == [E.CONSTANT]


def test_nan_values(four_files):
    v = volume(1)
    v[1, 2, 3] = np.nan
    v[2, 2, 2] = np.inf
    save(four_files["T1"], v)
    res = load_four_files(unlabelled=four_files.values())
    assert codes(res) == [E.NON_FINITE] and "2 NaN" in res.issues[0].message


def test_non_nifti_named_nii_gz(tmp_path):
    p = tmp_path / "fake_t1.nii.gz"
    p.write_text("this is not an image")
    assert validate_file(p).code == E.NOT_GZIP


def test_gzip_of_non_nifti(tmp_path):
    p = tmp_path / "fake_t1.nii.gz"
    p.write_bytes(gzip.compress(b"hello, not a NIfTI header" * 20))
    assert validate_file(p).code == E.NOT_NIFTI


def test_corrupt_gzip(tmp_path):
    good = save(tmp_path / "ok_t1.nii.gz", volume(1)).read_bytes()
    p = tmp_path / "bad_t1.nii.gz"
    p.write_bytes(good[: len(good) // 2])                         # truncated
    assert validate_file(p).code == E.CORRUPT_GZIP
    damaged = bytearray(good)
    damaged[len(good) // 2] ^= 0xFF                               # flipped byte in the middle
    p.write_bytes(bytes(damaged))
    assert validate_file(p).code == E.CORRUPT_GZIP


def test_truncated_uncompressed_nii(tmp_path):
    good = save(tmp_path / "ok_t1.nii", volume(1)).read_bytes()
    p = tmp_path / "cut_t1.nii"
    p.write_bytes(good[:600])
    assert validate_file(p).code in (E.UNREADABLE, E.NOT_NIFTI)


def test_wrong_extension_and_missing_and_empty(tmp_path):
    p = tmp_path / "t1.txt"
    p.write_text("x")
    assert validate_file(p).code == E.BAD_EXTENSION
    assert validate_file(tmp_path / "nope_t1.nii.gz").code == E.FILE_NOT_FOUND
    (tmp_path / "empty_t1.nii.gz").write_bytes(b"")
    assert validate_file(tmp_path / "empty_t1.nii.gz").code == E.EMPTY_FILE


def test_oversize_rejected_before_loading(four_files):
    with mock.patch("nibabel.load") as load:
        r = validate_file(four_files["T1"], max_bytes=100)
    assert r.code == E.FILE_TOO_LARGE and not load.called
    res = load_four_files(unlabelled=four_files.values(), max_bytes=100)
    assert set(codes(res)) == {E.FILE_TOO_LARGE} and len(res.issues) == 4


def test_gzip_bomb_capped(tmp_path, monkeypatch):
    monkeypatch.setattr("neuroscope.io.validate.GZIP_SLACK_BYTES", 0)
    p = tmp_path / "bomb_t1.nii.gz"
    p.write_bytes(gzip.compress(b"\0" * 5_000_000))               # 5 KB on disk, 5 MB decompressed
    assert validate_file(p, max_data_bytes=1000).code == E.DATA_TOO_LARGE


def test_all_problems_reported_not_just_first(four_files):
    save(four_files["FLAIR"], np.zeros(SHAPE, np.float32))
    four_files["T2"].write_text("garbage")
    assert sorted(codes(load_four_files(unlabelled=four_files.values()))) == sorted([E.CONSTANT, E.NOT_GZIP])


# ------------------------------------------------------------------ expert mask

def test_expert_mask_valid_and_zero_mask_allowed(four_files, tmp_path):
    m = np.zeros(SHAPE, np.uint8)
    m[2:4, 2:4, 2:4] = 2
    m[3, 3, 3] = 4
    res = load_four_files(unlabelled=four_files.values(), seg=save(tmp_path / "seg.nii.gz", m))
    assert res.ok and res.case.summary()["expert_mask"]["voxels_per_code"] == {0: 952, 2: 7, 4: 1}
    empty = load_four_files(unlabelled=four_files.values(), seg=save(tmp_path / "e.nii.gz", np.zeros(SHAPE, np.uint8)))
    assert empty.ok


def test_expert_mask_in_decathlon_codes_rejected(four_files, tmp_path):
    m = np.zeros(SHAPE, np.uint8)
    m[1, 1, 1] = 3
    res = load_four_files(unlabelled=four_files.values(), seg=save(tmp_path / "seg.nii.gz", m))
    assert codes(res) == [E.BAD_LABELS] and "Decathlon" in res.issues[0].message


def test_expert_mask_on_other_grid_rejected(four_files, tmp_path):
    aff = AFFINE.copy()
    aff[1, 3] += 1
    res = load_four_files(unlabelled=four_files.values(),
                          seg=save(tmp_path / "seg.nii.gz", np.zeros(SHAPE, np.uint8), affine=aff))
    assert codes(res) == [E.AFFINE_MISMATCH] and "expert mask" in res.issues[0].message


def test_assignment_returns_no_files_on_failure(four_files):
    a = assign_sequences(unlabelled=list(four_files.values())[:2])
    assert not a.ok and a.files == {}
