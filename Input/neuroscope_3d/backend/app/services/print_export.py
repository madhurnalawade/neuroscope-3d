"""
Print-ready clear brain shell: the brain as a clear solid with a cavity, and the tumour regions as separate
nested solids (oedema shell, enhancing shell, necrotic core) that a slicer can give their own materials.

Surfaces come from the same masks and marching cubes as the viewer (`viewer_data._surface`), but are built
from the nested regions brain ⊃ WT ⊃ TC ⊃ NCR rather than from the per-label meshes: the oedema label alone
is a hollow shell, so subtracting it would leave brain material where the tumour core is.

Parts (all after scaling; cavity(X) = X offset outward by CLEARANCE):
    brain_clear      = brain - cavity(WT)                          (+ raised "L" on the patient's left)
    edema_green      = (WT ∩ brain) - cavity(TC)
    enhancing_yellow = (TC ∩ WT ∩ brain) - cavity(NCR)
    necrotic_red     = NCR ∩ TC ∩ WT ∩ brain
Each part is clipped to its parent, so separately smoothed layers cannot overlap.

Research and teaching prototype; model-predicted regions; not for clinical use.
"""

import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.services import viewer_data

from brainseg.labels import region_masks

try:
    import manifold3d
    import pymeshlab
    import trimesh
except ImportError as _e:  # pragma: no cover - reported as a clear message by the CLI
    raise ImportError(f"The print export needs trimesh, manifold3d and pymeshlab ({_e}). "
                      "Install them with: pip install -r requirements.txt") from _e


# ---------------------------------------------------------------- config: every print parameter lives here

@dataclass(frozen=True)
class PrintConfig:
    SCALE: float = 1.5                                  # uniform, every part (1.0 = life size)
    CLEARANCE: float = 0.2                              # mm gap around each nested part, after scaling
    SMOOTH_ITERATIONS: int = 15                         # Taubin passes (lambda 0.5, mu -0.53: no shrinkage)
    TARGET_FACES: Tuple[int, int] = (250_000, 60_000)   # brain, each tumour layer
    MODE: str = "solid_clear"                           # "solid_clear" or "split_halves"
    BUILD_VOLUME: Tuple[float, float, float] = (220.0, 220.0, 250.0)   # mm, x y z
    MIN_WALL: float = 1.2                               # mm; thinner walls are reported
    MAGNET: Tuple[float, float] = (6.0, 2.0)            # magnet diameter, height (mm); pockets add CLEARANCE
    SOURCE: str = "pred"                                # "pred" (model) or "expert" label map
    MIN_COMPONENT_MM3: float = 2.0                      # loose bits and bubbles smaller than this are removed
    MARKER: bool = True                                 # raised "L" on the left side


CONFIG = PrintConfig()
MODES = ("solid_clear", "split_halves")
SIGMA = {"brain": 1.0, "WT": 0.6, "TC": 0.6, "NCR": 0.6}   # gaussian before marching cubes, as in the viewer
TAUBIN = (0.5, 0.53)                                        # lambda, -mu
PART_COLOURS = {                                            # 3MF displaycolor, sRGB + alpha
    "brain_clear": "#D8D0C040",
    "edema_green": "#3FA34DFF",
    "enhancing_yellow": "#F2C230FF",
    "necrotic_red": "#C8302CFF",
}
SMOOTH_SHIFT_WARN_PERCENT = 2.0
WALL_SAMPLES = 4000
OVERLAP_TOLERANCE_MM3 = 1e-3
SNAP_MM = 0.01                                              # shortest edge kept in the exported parts
L_MARKER = {"stroke": 2.4, "height": 14.0, "width": 9.0, "proud": 1.0, "embed": 2.5}   # mm, not scaled


class PrintExportError(Exception):
    """A clear, user-facing reason the print export cannot go on."""


# ---------------------------------------------------------------- mesh helpers

def _tm(v, f) -> "trimesh.Trimesh":
    return trimesh.Trimesh(np.asarray(v, np.float64), np.asarray(f, np.int64), process=True)


def _drop_small(tm, min_mm3: float):
    """Remove loose shells (solid bits or bubbles) under min_mm3; returns (mesh, number removed)."""
    bodies = tm.split(only_watertight=False)
    if len(bodies) <= 1:
        return tm, 0
    keep = [b for b in bodies if abs(b.volume) >= min_mm3]
    if not keep:
        return tm, 0
    return (trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]), len(bodies) - len(keep)


def _repair(tm, min_mm3: float):
    """pymeshlab clean-up (duplicates, degenerate faces, non-manifold edges/vertices, holes), then consistent
    outward winding and removal of tiny loose shells. Returns (mesh, number of shells removed)."""
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(tm.vertices, np.float64),
                               face_matrix=np.asarray(tm.faces, np.int32)))
    for flt in ("meshing_remove_duplicate_vertices", "meshing_remove_duplicate_faces", "meshing_remove_null_faces",
                "meshing_repair_non_manifold_edges", "meshing_repair_non_manifold_vertices",
                "meshing_close_holes", "meshing_remove_unreferenced_vertices"):
        ms.apply_filter(flt)
    m = ms.current_mesh()
    out = _tm(m.vertex_matrix(), m.face_matrix())
    out.update_faces(out.nondegenerate_faces())
    out.update_faces(out.unique_faces())
    out.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(out, multibody=False)   # multibody=True would turn cavities into solids
    return _drop_small(out, min_mm3)


def _decimate(tm, target: int):
    if len(tm.faces) <= target:
        return tm
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(tm.vertices, np.float64),
                               face_matrix=np.asarray(tm.faces, np.int32)))
    ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetfacenum=int(target), preservenormal=True,
                    preservetopology=True, optimalplacement=True, qualitythr=0.3)
    m = ms.current_mesh()
    return _tm(m.vertex_matrix(), m.face_matrix())


def _to_manifold(name: str, tm, min_mm3: float):
    """manifold3d solid; on a non-manifold input, repair with pymeshlab once and retry."""
    for attempt in range(2):
        m = manifold3d.Manifold(manifold3d.Mesh(vert_properties=np.ascontiguousarray(tm.vertices, np.float32),
                                                tri_verts=np.ascontiguousarray(tm.faces, np.uint32)))
        if m.status() == manifold3d.Error.NoError and not m.is_empty():
            return m
        status = m.status()
        if attempt == 0:
            tm, _ = _repair(tm, min_mm3)
    raise PrintExportError(f"{name}: the surface is not a closed manifold even after pymeshlab repair "
                           f"(manifold3d status {status}). Try more SMOOTH_ITERATIONS or a different case.")


def _from_manifold(m, snap: float = 0.0):
    """trimesh of a manifold; snap > 0 first collapses edges shorter than snap mm (booleans leave
    near-duplicate vertices that break watertightness once a slicer or float32 file welds them)."""
    mesh = (m.simplify(snap) if snap else m).to_mesh()
    if len(mesh.tri_verts) == 0:
        return None
    return _tm(np.asarray(mesh.vert_properties)[:, :3], np.asarray(mesh.tri_verts))


def _boolean(name: str, result, inputs: Dict[str, Any], op, min_mm3: float):
    """Check a boolean result; if manifold3d flags it, repair the inputs with pymeshlab and redo it once."""
    if result.status() == manifold3d.Error.NoError:
        return result
    fixed = {k: _to_manifold(k, _repair(_from_manifold(v), min_mm3)[0], min_mm3) for k, v in inputs.items()}
    result = op(**fixed)
    if result.status() != manifold3d.Error.NoError:
        raise PrintExportError(f"{name}: boolean failed after repair (manifold3d status {result.status()}).")
    return result


def _offset(tm, d: float):
    out = tm.copy()
    out.vertices = out.vertices + out.vertex_normals * d   # outward normals after fix_normals
    return out


def _volume(m) -> float:
    tm = _from_manifold(m)
    return 0.0 if tm is None else float(tm.volume)


# ---------------------------------------------------------------- layers

def _layer_surfaces(brain: np.ndarray, labels: np.ndarray, canon_affine: np.ndarray, cfg: PrintConfig):
    """Smoothed, decimated, scaled closed surfaces of brain/WT/TC/NCR, plus a volume-shift row per layer."""
    reg = region_masks(labels)
    masks = {"brain": brain > 0, "WT": reg["WT"], "TC": reg["TC"], "NCR": labels == viewer_data.LABEL_CODES["NCR"]}
    voxel_mm3 = abs(float(np.linalg.det(canon_affine[:3, :3])))
    min_raw = cfg.MIN_COMPONENT_MM3 / cfg.SCALE ** 3          # threshold is in printed mm³
    layers, rows = {}, []
    for name, mask in masks.items():
        surf = viewer_data._surface(mask, canon_affine, SIGMA[name], 1)
        if surf is None:
            continue
        tm, dropped = _repair(_tm(*surf), min_raw)
        before = float(tm.volume)
        trimesh.smoothing.filter_taubin(tm, lamb=TAUBIN[0], nu=TAUBIN[1], iterations=cfg.SMOOTH_ITERATIONS)
        tm = _decimate(tm, cfg.TARGET_FACES[0] if name == "brain" else cfg.TARGET_FACES[1])
        tm, dropped2 = _repair(tm, min_raw)
        after = float(tm.volume)
        shift = 100.0 * (after - before) / before if before else 0.0
        rows.append({"layer": name, "voxel_mm3": int(mask.sum()) * voxel_mm3, "before_smoothing_mm3": before,
                     "after_smoothing_mm3": after, "change_percent": shift, "loose_bits_removed": dropped + dropped2,
                     "warning": (f"volume changed {shift:+.2f}% by smoothing (limit {SMOOTH_SHIFT_WARN_PERCENT}%)"
                                 if abs(shift) > SMOOTH_SHIFT_WARN_PERCENT else "")})
        tm.apply_scale(cfg.SCALE)
        layers[name] = tm
    if "brain" not in layers:
        raise PrintExportError("No brain surface: the sequences are empty.")
    if "WT" not in layers:
        raise PrintExportError(f"The {cfg.SOURCE} label map has no tumour voxels, so there is nothing to put "
                               "inside the brain. Run segmentation first or choose another case.")
    centre_mask = masks["TC"] if masks["TC"].any() else masks["WT"]
    centroid = (canon_affine @ np.r_[np.argwhere(centre_mask).mean(axis=0), 1.0])[:3] * cfg.SCALE
    return layers, rows, centroid


# ---------------------------------------------------------------- marker, magnets

def _l_marker(brain_tm, cavity, notes: List[str]):
    """Raised "L" at the brain's most-left point (-x in RAS), readable when viewed from the left."""
    k = L_MARKER
    x0, yc, zc = brain_tm.vertices[np.argmin(brain_tm.vertices[:, 0])]
    depth = k["proud"] + k["embed"]
    x = x0 - k["proud"]
    # seen from the left (looking toward +x, up +z) the viewer's right is -y: stem at +y, foot runs toward -y
    stem = manifold3d.Manifold.cube([depth, k["stroke"], k["height"]]).translate(
        [x, yc + k["width"] / 2 - k["stroke"], zc - k["height"] / 2])
    foot = manifold3d.Manifold.cube([depth, k["width"], k["stroke"]]).translate(
        [x, yc - k["width"] / 2, zc - k["height"] / 2])
    marker = stem + foot
    if _volume(marker ^ cavity) > 0:
        notes.append("L marker skipped: the tumour cavity reaches the brain's left surface where it would go.")
        return None
    return marker


def _section(tm, cx: float):
    """Cross-section at x = cx as a shapely shape in (y, z), even-odd so cavities are holes."""
    from shapely.geometry import Polygon
    sec = tm.section(plane_origin=[cx, 0, 0], plane_normal=[1, 0, 0])
    if sec is None:
        return None
    shape = None
    for loop in sec.discrete:
        if len(loop) < 3:
            continue
        p = Polygon(loop[:, 1:]).buffer(0)
        shape = p if shape is None else shape.symmetric_difference(p)
    return shape


def _magnet_sites(brain_tm, cavity_tm, cx: float, cfg: PrintConfig, notes: List[str]):
    """Two pocket centres (y, z) on the cut face: >= 3 mm inside the brain outline, >= 5 mm from the tumour
    cavity, with enough material behind the pocket on both halves; as far apart as possible."""
    import shapely
    from scipy.spatial.distance import cdist
    r = (cfg.MAGNET[0] + cfg.CLEARANCE) / 2
    depth = cfg.MAGNET[1] + cfg.CLEARANCE
    solid = _section(brain_tm, cx)
    if solid is None or solid.is_empty:
        notes.append("Magnet pockets skipped: the cut plane misses the brain.")
        return []
    allowed = solid.buffer(-(r + 3.0))
    tumour = _section(cavity_tm, cx)
    if tumour is not None and not tumour.is_empty:
        allowed = allowed.difference(tumour.buffer(r + 5.0))
    if allowed.is_empty:
        notes.append("Magnet pockets skipped: no room on the cut face away from the tumour.")
        return []
    y0, z0, y1, z1 = allowed.bounds
    yy, zz = np.meshgrid(np.arange(y0, y1, 1.5), np.arange(z0, z1, 1.5))
    pts = np.c_[yy.ravel(), zz.ravel()]
    pts = pts[shapely.contains_xy(allowed, pts[:, 0], pts[:, 1])]
    if len(pts):
        need = depth + cfg.MIN_WALL
        good = np.ones(len(pts), bool)
        for sign in (1.0, -1.0):   # material behind the pocket on each half
            origins = np.c_[np.full(len(pts), cx), pts]
            dirs = np.tile([sign, 0.0, 0.0], (len(pts), 1))
            loc, idx, _ = brain_tm.ray.intersects_location(origins, dirs, multiple_hits=False)
            dist = np.zeros(len(pts))
            dist[idx] = np.abs(loc[:, 0] - cx)
            good &= dist >= need
        pts = pts[good]
    if len(pts) < 2:
        notes.append("Magnet pockets skipped: no place on the cut face is thick enough on both halves.")
        return []
    if len(pts) > 800:
        pts = pts[np.random.default_rng(0).choice(len(pts), 800, replace=False)]
    d = cdist(pts, pts)
    i, j = np.unravel_index(np.argmax(d), d.shape)
    if d[i, j] < 2 * r + 4:
        notes.append("Only one magnet pocket fits on the cut face; none added.")
        return []
    return [tuple(pts[i]), tuple(pts[j])]


def _pocket(cx: float, y: float, z: float, cfg: PrintConfig):
    r = (cfg.MAGNET[0] + cfg.CLEARANCE) / 2
    depth = cfg.MAGNET[1] + cfg.CLEARANCE
    # cylinder along z, turned onto x, spanning the cut plane: each half gets a pocket `depth` deep
    return manifold3d.Manifold.cylinder(2 * depth, r, r, 48, True).rotate([0, 90, 0]).translate([cx, y, z])


# ---------------------------------------------------------------- checks

def _thin_walls(tm, min_wall: float, rng) -> Dict[str, Any]:
    """Local wall thickness from inward rays at sampled faces."""
    idx = rng.choice(len(tm.faces), min(WALL_SAMPLES, len(tm.faces)), replace=False)
    n = tm.face_normals[idx]
    origins = tm.triangles_center[idx] - n * 1e-3
    loc, ray, _ = tm.ray.intersects_location(origins, -n, multiple_hits=False)
    d = np.linalg.norm(loc - origins[ray], axis=1) if len(ray) else np.zeros(0)
    if not len(d):
        return {"min_mm": None, "thin_percent": 0.0}
    return {"min_mm": float(d.min()), "thin_percent": 100.0 * float((d < min_wall).mean())}


def _fit(tms: List[Any], build: Tuple[float, float, float]) -> Dict[str, Any]:
    """Largest uniform factor (<1 means too big) that fits the build volume, rotating about z on the bed."""
    v = np.concatenate([t.vertices for t in tms])
    height = float(np.ptp(v[:, 2]))
    from scipy.spatial import ConvexHull
    xy = v[:, :2][ConvexHull(v[:, :2]).vertices]
    best, best_deg = 0.0, 0
    for deg in range(0, 180, 3):
        a = math.radians(deg)
        p = xy @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        w, h = np.ptp(p[:, 0]), np.ptp(p[:, 1])
        f = min(build[0] / w, build[1] / h, build[2] / height)
        if f > best:
            best, best_deg = f, deg
    e = np.ptp(v, axis=0)
    return {"extent_mm": [float(x) for x in e], "fit_factor": float(best), "bed_rotation_deg": best_deg,
            "fits": bool(best >= 1.0), "fits_unrotated": bool(np.all(e <= np.asarray(build)))}


# ---------------------------------------------------------------- build

def build_parts(brain: np.ndarray, labels: np.ndarray, canon_affine: np.ndarray, cfg: PrintConfig = CONFIG):
    """Canonical-order brain mask + BraTS label map -> {group: {part: trimesh}} in printed mm, and a report."""
    if cfg.MODE not in MODES:
        raise PrintExportError(f"MODE must be one of {', '.join(MODES)}, got '{cfg.MODE}'.")
    if cfg.SCALE <= 0 or cfg.CLEARANCE < 0:
        raise PrintExportError("SCALE must be > 0 and CLEARANCE >= 0.")
    notes: List[str] = []
    min_mm3 = cfg.MIN_COMPONENT_MM3
    layers, smoothing, centroid = _layer_surfaces(brain, labels, canon_affine, cfg)

    M = {k: _to_manifold(k, tm, min_mm3) for k, tm in layers.items()}
    # union with the layer itself: a normal offset can fold inward at sharp creases and miss a sliver
    cav = {k: _to_manifold(f"{k} cavity", _offset(layers[k], cfg.CLEARANCE), min_mm3) + M[k]
           for k in ("WT", "TC", "NCR") if k in layers}

    def cut(name, op, **inputs):
        return _boolean(name, op(**inputs), inputs, op, min_mm3)

    wt = cut("edema_green", lambda a, b: a ^ b, a=M["WT"], b=M["brain"])
    parts = {"brain_clear": cut("brain_clear", lambda a, b: a - b, a=M["brain"], b=cav["WT"])}
    parts["edema_green"] = cut("edema_green", lambda a, b: a - b, a=wt, b=cav["TC"]) if "TC" in cav else wt
    if "TC" in M:
        tc = cut("enhancing_yellow", lambda a, b: a ^ b, a=M["TC"], b=wt)
        parts["enhancing_yellow"] = (cut("enhancing_yellow", lambda a, b: a - b, a=tc, b=cav["NCR"])
                                     if "NCR" in cav else tc)
        if "NCR" in M:
            parts["necrotic_red"] = cut("necrotic_red", lambda a, b: a ^ b, a=M["NCR"], b=tc)

    if cfg.MARKER:
        marker = _l_marker(layers["brain"], cav["WT"], notes)
        if marker is not None:
            parts["brain_clear"] = cut("brain_clear", lambda a, b: a + b, a=parts["brain_clear"], b=marker)

    overlaps = {}
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}/{b}"] = _volume(parts[a] ^ parts[b])

    transforms: Dict[str, Any] = {}
    cut_x = None
    if cfg.MODE == "solid_clear":
        groups = {"assembly": {k: _from_manifold(m, SNAP_MM) for k, m in parts.items()}}
        notes.append("solid_clear is for multi-material printing in place (e.g. AMS/MMU): the tumour parts are "
                     "sealed inside the brain. Resin and single-material printers need MODE split_halves.")
    else:
        cut_x = float(centroid[0])
        brain_tm = _from_manifold(parts["brain_clear"])
        sites = _magnet_sites(brain_tm, layers["WT"], cut_x, cfg, notes)
        for y, z in sites:
            parts["brain_clear"] = cut("brain_clear", lambda a, b: a - b, a=parts["brain_clear"],
                                       b=_pocket(cut_x, y, z, cfg))
        transforms["magnet_pockets_yz_mm"] = [[float(y), float(z)] for y, z in sites]
        groups = {"left": {}, "right": {}}
        for name, m in parts.items():
            for half in m.split_by_plane([1.0, 0.0, 0.0], cut_x):
                tm = _from_manifold(half, SNAP_MM)
                if tm is None or abs(tm.volume) < min_mm3:
                    continue
                side = "right" if tm.bounds.mean(axis=0)[0] > cut_x else "left"   # RAS: +x is the patient's right
                groups[side][f"{name}_{side}"] = tm
        groups = {g: p for g, p in groups.items() if p}

    out_groups: Dict[str, Dict[str, Any]] = {}
    for g, group in groups.items():
        cleaned = {}
        for name, tm in group.items():
            tm, dropped = _drop_small(tm, min_mm3)
            if dropped:
                notes.append(f"{name}: removed {dropped} loose bit(s) under {min_mm3:g} mm³ left by the booleans.")
            cleaned[name] = tm
        out_groups[g] = cleaned

    # placement: solid keeps its orientation (shifted to positive coordinates); halves lie cut face down
    x_offset = 0.0
    for g, group in out_groups.items():
        T = np.eye(4)
        if g in ("left", "right"):
            # left half's cut face looks +x, right half's -x; rotate about y so it looks -z (onto the bed)
            T = trimesh.transformations.rotation_matrix(math.pi / 2 if g == "left" else -math.pi / 2, [0, 1, 0])
        ref_name = "brain_clear" if g == "assembly" else f"brain_clear_{g}"
        ref = group[ref_name] if ref_name in group else next(iter(group.values()))
        lo = trimesh.transform_points(ref.vertices, T).min(axis=0)
        T = trimesh.transformations.translation_matrix([x_offset - lo[0], -lo[1], -lo[2]]) @ T
        for tm in group.values():
            tm.apply_transform(T)
        x_offset += float(np.ptp(np.concatenate([t.vertices for t in group.values()])[:, 0])) + 15.0
        transforms[g] = {"world_mm_times_scale_to_print": T.tolist(), "scale": cfg.SCALE}

    report = {"config": asdict(cfg), "smoothing": smoothing, "notes": notes, "overlap_mm3": overlaps,
              "cut_plane_x_mm": cut_x, "tumour_core_centroid_mm": [float(c) for c in centroid],
              "transforms": transforms}
    return out_groups, report


def check_parts(groups, report, cfg: PrintConfig = CONFIG) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    rows = []
    for g, group in groups.items():
        for name, tm in group.items():
            warn = []
            stored = _tm(np.asarray(tm.vertices, np.float32), tm.faces)   # as the float32 STL stores it
            watertight = all(bool(t.is_watertight and t.is_winding_consistent and t.volume > 0) for t in (tm, stored))
            if not watertight:
                warn.append("not watertight")
            wall = _thin_walls(tm, cfg.MIN_WALL, rng)
            if wall["min_mm"] is not None and wall["min_mm"] < cfg.MIN_WALL:
                warn.append(f"walls < {cfg.MIN_WALL:g} mm on {wall['thin_percent']:.1f}% of sampled surface "
                            f"(min {wall['min_mm']:.2f} mm)")
            shells = [float(b.volume) for b in tm.split(only_watertight=False)]
            pieces = sorted(v for v in shells if v > 0)
            cavities = sum(v < 0 for v in shells)
            if len(pieces) > 1:
                warn.append(f"{len(pieces)} separate pieces (smallest {pieces[0]:.0f} mm³)")
            rows.append({"part": name, "group": g, "volume_mm3": float(tm.volume),
                         "bbox_mm": [float(x) for x in tm.extents], "watertight": watertight,
                         "faces": int(len(tm.faces)), "pieces": len(pieces), "internal_cavities": int(cavities),
                         "wall": wall, "warnings": warn})
    bad = {k: v for k, v in report["overlap_mm3"].items() if v > OVERLAP_TOLERANCE_MM3}
    if bad:
        report["notes"].append("Parts overlap: " + ", ".join(f"{k} {v:.3f} mm³" for k, v in bad.items()))
    fits = {}
    for g, group in groups.items():
        fit = _fit(list(group.values()), cfg.BUILD_VOLUME)
        fits[g] = fit
        if not fit["fits"]:
            suggest = math.floor(cfg.SCALE * fit["fit_factor"] * 0.98 * 20) / 20
            hint = "" if cfg.MODE == "split_halves" else " or MODE split_halves"
            report["notes"].append(
                f"{g}: {' x '.join(f'{e:.0f}' for e in fit['extent_mm'])} mm does not fit the build volume "
                f"{' x '.join(f'{b:g}' for b in cfg.BUILD_VOLUME)} mm even rotated on the bed. "
                f"Use SCALE <= {suggest:.2f}{hint}.")
        elif not fit["fits_unrotated"]:
            report["notes"].append(f"{g}: fits only when rotated {fit['bed_rotation_deg']} deg about z on the bed.")
    report["fit"] = fits
    return rows


# ---------------------------------------------------------------- files

def _stl(tm) -> bytes:
    """Binary STL with the same disclaimer header as the viewer's STL export."""
    rec = np.zeros(len(tm.faces), viewer_data.STL_RECORD)
    rec["normal"], rec["v"] = tm.face_normals, tm.triangles
    return viewer_data.STL_HEADER + np.uint32(len(tm.faces)).tobytes() + rec.tobytes()


def _base(name: str) -> str:
    return name.rsplit("_", 1)[0] if name.endswith(("_left", "_right")) else name


def write_3mf(path: Path, groups, title: str):
    """Core-spec 3MF: one named mesh object per part with a basematerials colour, grouped into one
    component object per printable group so slicers load it as one object with named parts in place."""
    bases = list(dict.fromkeys(_base(n) for g in groups.values() for n in g))
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<model unit="millimeter" xml:lang="en-US" '
           'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
           f' <metadata name="Title">{title}</metadata>',
           ' <metadata name="Description">Research prototype: model-predicted tumour regions, not for clinical use'
           '</metadata>',
           ' <resources>', '  <basematerials id="1">']
    out += [f'   <base name="{b}" displaycolor="{PART_COLOURS[b]}"/>' for b in bases]
    out.append('  </basematerials>')
    oid, items = 2, []
    for g, group in groups.items():
        ids = []
        for name, tm in group.items():
            out.append(f'  <object id="{oid}" type="model" name="{name}" pid="1" pindex="{bases.index(_base(name))}">')
            out.append('   <mesh><vertices>')
            out += [f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in tm.vertices.tolist()]
            out.append('   </vertices><triangles>')
            out += [f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in tm.faces.tolist()]
            out.append('   </triangles></mesh>', )
            out.append('  </object>')
            ids.append(oid)
            oid += 1
        out.append(f'  <object id="{oid}" type="model" name="{"brain_model" if g == "assembly" else g + "_half"}">')
        out.append('   <components>' + "".join(f'<component objectid="{i}"/>' for i in ids) + '</components>')
        out.append('  </object>')
        items.append(oid)
        oid += 1
    out += [' </resources>', ' <build>'] + [f'  <item objectid="{i}"/>' for i in items] + [' </build>', '</model>']
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n'
             '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
             '</Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", "\n".join(out))


def write_preview(path: Path, groups):
    """Two views; the brain semi-transparent so the tumour parts show through."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(12, 6), dpi=120)
    allv = np.concatenate([t.vertices for g in groups.values() for t in g.values()])
    for k, (elev, azim) in enumerate(((15, 180), (55, -60))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for g in groups.values():
            for name, tm in sorted(g.items(), key=lambda kv: _base(kv[0]) != "brain_clear"):
                base = _base(name)
                small = _decimate(tm, 12_000 if base == "brain_clear" else 4_000)
                rgba = PART_COLOURS[base]
                colour = tuple(int(rgba[i:i + 2], 16) / 255 for i in (1, 3, 5))
                ax.add_collection3d(Poly3DCollection(small.triangles, facecolor=colour, linewidths=0,
                                                     alpha=0.12 if base == "brain_clear" else 0.95))
        lo, hi = allv.min(0), allv.max(0)
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(hi - lo)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x mm"); ax.set_ylabel("y mm"); ax.set_zlabel("z mm")
    fig.suptitle("Print preview - research prototype, model-predicted regions, not for clinical use")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def format_summary(rows, report) -> str:
    lines = ["Smoothing (volume per layer, mm³, before scaling):",
             f"  {'layer':<6} {'voxels':>12} {'before':>12} {'after':>12} {'change':>8}  warnings"]
    for s in report["smoothing"]:
        lines.append(f"  {s['layer']:<6} {s['voxel_mm3']:>12.0f} {s['before_smoothing_mm3']:>12.0f} "
                     f"{s['after_smoothing_mm3']:>12.0f} {s['change_percent']:>+7.2f}%  {s['warning']}")
    lines += ["", f"Parts (printed mm, SCALE {report['config']['SCALE']:g}, MODE {report['config']['MODE']}):",
              f"  {'part':<24} {'volume mm3':>12} {'bbox mm':>22} {'watertight':>10} {'faces':>8}  warnings"]
    for r in rows:
        bbox = " x ".join(f"{b:.1f}" for b in r["bbox_mm"])
        lines.append(f"  {r['part']:<24} {r['volume_mm3']:>12.0f} {bbox:>22} {'yes' if r['watertight'] else 'NO':>10} "
                     f"{r['faces']:>8}  {'; '.join(r['warnings'])}")
    if report["notes"]:
        lines += ["", "Notes:"] + [f"  - {n}" for n in report["notes"]]
    return "\n".join(lines)


def export_study(study, out_dir: Path, cfg: PrintConfig = CONFIG, case_name: str = "case") -> Dict[str, Any]:
    """Build every part for a study and write the 3MF, per-part STLs, preview and summary into out_dir."""
    if cfg.SOURCE not in ("pred", "expert"):
        raise PrintExportError(f"SOURCE must be 'pred' or 'expert', got '{cfg.SOURCE}'.")
    _, ornt, canon_affine = viewer_data._canonical(study)
    labels = viewer_data._labels(study, cfg.SOURCE)
    if labels is None:
        what = "prediction (run segmentation first)" if cfg.SOURCE == "pred" else "expert mask"
        raise PrintExportError(f"This study has no {what}.")
    brain = viewer_data.brain_mask(study, ornt)
    groups, report = build_parts(brain, viewer_data._reorient(labels, ornt), canon_affine, cfg)
    rows = check_parts(groups, report, cfg)

    out_dir = Path(out_dir)
    (out_dir / "stl").mkdir(parents=True, exist_ok=True)
    for group in groups.values():
        for name, tm in group.items():
            (out_dir / "stl" / f"{name}.stl").write_bytes(_stl(tm))
    write_3mf(out_dir / f"{case_name}_print.3mf", groups, f"{case_name} print model")
    write_preview(out_dir / "preview.png", groups)
    (out_dir / "transforms.json").write_text(json.dumps(report["transforms"], indent=2), encoding="utf-8")
    text = format_summary(rows, report)
    (out_dir / "summary.txt").write_text(text + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"parts": rows, **report}, indent=2), encoding="utf-8")
    return {"rows": rows, "report": report, "text": text, "out_dir": str(out_dir)}
