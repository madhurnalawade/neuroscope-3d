"""Overlap metrics on small masks whose answers can be worked out by hand."""
import numpy as np
import pytest

from brainseg.metrics import case_metrics, dice, hd95, iou, precision, sensitivity


def box(shape, lo, hi, value=1):
    a = np.zeros(shape, np.uint8)
    a[tuple(slice(l, h) for l, h in zip(lo, hi))] = value
    return a


def test_half_overlapping_cubes():
    # two 10x10x10 cubes shifted by 5 along x: overlap 5x10x10 = 500 voxels
    p, r = box((30, 12, 12), (0, 1, 1), (10, 11, 11)) > 0, box((30, 12, 12), (5, 1, 1), (15, 11, 11)) > 0
    assert dice(p, r) == pytest.approx(0.5)              # 2*500 / (1000 + 1000)
    assert sensitivity(p, r) == pytest.approx(0.5)       # 500 / 1000
    assert precision(p, r) == pytest.approx(0.5)         # 500 / 1000
    assert iou(p, r) == pytest.approx(1 / 3)             # 500 / 1500


def test_prediction_inside_reference():
    # prediction is a 4x4x4 cube inside an 8x8x8 reference: every predicted voxel is correct, 64 of 512 are found
    p, r = box((10, 10, 10), (2, 2, 2), (6, 6, 6)) > 0, box((10, 10, 10), (0, 0, 0), (8, 8, 8)) > 0
    assert precision(p, r) == pytest.approx(1.0)         # 64 / 64
    assert sensitivity(p, r) == pytest.approx(64 / 512)
    assert iou(p, r) == pytest.approx(64 / 512)
    assert dice(p, r) == pytest.approx(2 * 64 / (64 + 512))


def test_empty_conventions():
    e, full = np.zeros((5, 5, 5), bool), box((5, 5, 5), (1, 1, 1), (3, 3, 3)) > 0
    # both empty: agreement is perfect, rates are undefined
    assert dice(e, e) == 1.0 and iou(e, e) == 1.0 and hd95(e, e, (1, 1, 1)) == 0.0
    assert sensitivity(e, e) is None and precision(e, e) is None
    # expert empty, model not: overlap 0, sensitivity undefined, precision 0, distance undefined
    assert dice(full, e) == 0.0 and iou(full, e) == 0.0
    assert sensitivity(full, e) is None and precision(full, e) == 0.0 and hd95(full, e, (1, 1, 1)) is None
    # model empty, expert not: overlap 0, sensitivity 0, precision undefined
    assert dice(e, full) == 0.0 and sensitivity(e, full) == 0.0 and precision(e, full) is None


def test_hd95_uses_voxel_spacing():
    # single voxels 5 apart along x; at 2 mm spacing the surfaces are 10 mm apart in both directions
    p, r = box((8, 3, 3), (0, 1, 1), (1, 2, 2)) > 0, box((8, 3, 3), (5, 1, 1), (6, 2, 2)) > 0
    assert hd95(p, r, (2.0, 1.0, 1.0)) == pytest.approx(10.0)
    assert hd95(p, p, (2.0, 1.0, 1.0)) == 0.0


def test_case_metrics_uses_brats_regions():
    # model: ED 2 shell with ET 4 and NCR 1 inside; expert: same with the enhancing part missing
    shape = (20, 20, 20)
    pred = box(shape, (2, 2, 2), (18, 18, 18), 2)
    pred[6:14, 6:14, 6:14] = 4
    pred[8:12, 8:12, 8:12] = 1
    ref = box(shape, (2, 2, 2), (18, 18, 18), 2)
    ref[8:12, 8:12, 8:12] = 1
    m = case_metrics(pred, ref, (1, 1, 1), with_hd95=False)
    assert m["WT"]["dice"] == 1.0 and m["WT"]["pred_voxels"] == 16 ** 3      # WT = 1+2+4, identical
    assert m["TC"]["pred_voxels"] == 8 ** 3 and m["TC"]["ref_voxels"] == 4 ** 3  # TC = 1+4
    assert m["TC"]["sensitivity"] == 1.0 and m["TC"]["precision"] == pytest.approx(64 / 512)
    assert m["ET"]["ref_voxels"] == 0 and m["ET"]["sensitivity"] is None    # ET = 4, expert has none
    assert m["ET"]["dice"] == 0.0 and m["ET"]["precision"] == 0.0 and m["ET"]["iou"] == 0.0
    assert set(m["WT"]) >= {"dice", "sensitivity", "precision", "iou", "hd95_mm"}
