"""Print-ready clear brain shell export (synthetic phantom; geometry plumbing only)."""

import re
import zipfile
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("manifold3d")
pytest.importorskip("pymeshlab")
trimesh = pytest.importorskip("trimesh")

from app.services import print_export  # noqa: E402

CFG = replace(print_export.CONFIG, SCALE=1.0, TARGET_FACES=(20_000, 6_000), SMOOTH_ITERATIONS=10)


def _phantom(shape=(96, 96, 64)):
    """Ellipsoid brain with a nested lesion off to the right: oedema shell, enhancing rim, necrotic core."""
    g = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    brain = ((g[0] / 0.75) ** 2 + (g[1] / 0.85) ** 2 + (g[2] / 0.85) ** 2 <= 1).astype(np.uint8)
    r = np.sqrt((g[0] - 0.3) ** 2 + (g[1] - 0.1) ** 2 + g[2] ** 2)
    labels = np.zeros(shape, np.uint8)
    labels[(r < 0.35) & (brain > 0)] = 2
    labels[r < 0.2] = 4
    labels[r < 0.12] = 1
    return brain, labels, np.eye(4)


@pytest.fixture(scope="module")
def solid():
    groups, report = print_export.build_parts(*_phantom(), CFG)
    return groups, report, print_export.check_parts(groups, report, CFG)


def test_solid_clear_parts_are_watertight_nested_and_do_not_overlap(solid):
    groups, report, rows = solid
    assert list(groups) == ["assembly"]
    assert list(groups["assembly"]) == ["brain_clear", "edema_green", "enhancing_yellow", "necrotic_red"]
    assert all(r["watertight"] for r in rows)
    assert all(v <= print_export.OVERLAP_TOLERANCE_MM3 for v in report["overlap_mm3"].values())
    assert all(abs(s["change_percent"]) < print_export.SMOOTH_SHIFT_WARN_PERCENT for s in report["smoothing"])
    by = {r["part"]: r for r in rows}
    assert by["brain_clear"]["internal_cavities"] >= 1                         # the tumour cavity
    assert by["edema_green"]["internal_cavities"] >= 1 and by["enhancing_yellow"]["internal_cavities"] >= 1
    # each part sits in its parent's cavity: parts nest by bounding box too
    ext = {k: np.ptp(tm.vertices, axis=0) for k, tm in groups["assembly"].items()}
    assert np.all(ext["necrotic_red"] < ext["enhancing_yellow"]) and np.all(ext["enhancing_yellow"] < ext["edema_green"])
    assert any("multi-material" in n for n in report["notes"])


def test_l_marker_is_a_flat_face_on_the_left(solid):
    brain = solid[0]["assembly"]["brain_clear"]
    xmin = brain.vertices[:, 0].min()
    face = brain.vertices[np.abs(brain.vertices[:, 0] - xmin) < 1e-3]
    k = print_export.L_MARKER
    assert np.allclose(np.ptp(face[:, 1:], axis=0), [k["width"], k["height"]], atol=1e-3)


def test_3mf_has_named_coloured_parts(solid, tmp_path):
    groups = solid[0]
    path = tmp_path / "case_print.3mf"
    print_export.write_3mf(path, groups, "phantom")
    xml = zipfile.ZipFile(path).read("3D/3dmodel.model").decode()
    colours = dict(re.findall(r'<base name="([^"]+)" displaycolor="([^"]+)"', xml))
    assert colours == print_export.PART_COLOURS
    assert "<components>" in xml and 'name="brain_model"' in xml
    pytest.importorskip("lxml")
    scene = trimesh.load(path)
    assert set(scene.geometry) == set(groups["assembly"])
    assert all(g.is_watertight for g in scene.geometry.values())


def test_split_halves_lie_flat_with_magnet_pockets():
    cfg = replace(CFG, MODE="split_halves")
    groups, report = print_export.build_parts(*_phantom(), cfg)
    rows = print_export.check_parts(groups, report, cfg)
    assert set(groups) == {"left", "right"}
    assert {"brain_clear_left", "necrotic_red_left", "brain_clear_right", "necrotic_red_right"} <= {r["part"] for r in rows}
    assert all(r["watertight"] for r in rows)
    assert len(report["transforms"]["magnet_pockets_yz_mm"]) == 2
    pocket = cfg.MAGNET[0] + cfg.CLEARANCE
    for side in ("left", "right"):
        brain = groups[side][f"brain_clear_{side}"]
        assert abs(brain.vertices[:, 2].min()) < 1e-6                     # cut face on the bed
        loops = brain.section(plane_origin=[0, 0, 1.0], plane_normal=[0, 0, 1]).discrete
        round_holes = [lp for lp in loops if np.allclose(np.ptp(lp, axis=0)[:2], pocket, atol=0.3)]
        assert len(round_holes) == 2


def test_fit_check_suggests_a_smaller_scale(solid):
    groups, report, _ = solid
    report = {**report, "notes": []}
    print_export.check_parts(groups, report, replace(CFG, BUILD_VOLUME=(40.0, 40.0, 40.0)))
    assert not report["fit"]["assembly"]["fits"]
    assert any("Use SCALE <=" in n for n in report["notes"])


def test_label_map_without_tumour_is_a_clear_error():
    brain, labels, aff = _phantom()
    with pytest.raises(print_export.PrintExportError, match="no tumour voxels"):
        print_export.build_parts(brain, np.zeros_like(labels), aff, CFG)
    with pytest.raises(print_export.PrintExportError, match="MODE"):
        print_export.build_parts(brain, labels, aff, replace(CFG, MODE="nope"))
