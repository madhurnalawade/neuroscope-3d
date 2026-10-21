"""Render the landing-page "plate": an axial sweep of a public case with the model's predicted contours.

    python scripts/make_hero_plate.py backup_case

Reads <dir>/*_flair.nii.gz and <dir>/prediction.nii.gz, writes a vertical JPEG sprite and frame metadata to
Input/neuroscope_3d/backend/app/static/hero/. Source: Medical Segmentation Decathlon Task01 (CC BY-SA 4.0).
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, zoom

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Input" / "neuroscope_3d" / "backend" / "app" / "static" / "hero"
FRAMES, WIDTH = 44, 300
# label code -> RGB (NCR vermilion, ED sage, ET ochre) — same palette as the app
COLOURS = {1: (214, 69, 44), 2: (120, 168, 112), 4: (232, 176, 48)}


def main(case_dir: Path):
    flair_img = nib.as_closest_canonical(nib.load(str(next(case_dir.glob("*_flair.nii.gz")))))
    pred_img = nib.as_closest_canonical(nib.load(str(case_dir / "prediction.nii.gz")))
    flair = flair_img.get_fdata(dtype=np.float32)
    pred = np.asarray(pred_img.dataobj).astype(np.uint8)
    sp = flair_img.header.get_zooms()[:3]

    brain = flair > 0
    xs, ys, zs = np.nonzero(brain)
    x0, x1, y0, y1 = xs.min() - 6, xs.max() + 7, ys.min() - 6, ys.max() + 7
    z_lo, z_hi = int(np.percentile(zs, 4)), int(np.percentile(zs, 96))
    z_list = np.linspace(z_lo, z_hi, FRAMES).round().astype(int)
    vmax = float(np.percentile(flair[brain], 99.6))

    frames, meta = [], []
    for z in z_list:
        # radiological axial view: patient right on screen left, anterior at top
        img = flair[x0:x1, y0:y1, z][::-1, ::-1].T
        lab = pred[x0:x1, y0:y1, z][::-1, ::-1].T
        k = WIDTH / img.shape[1]
        g = np.clip(zoom(img, k, order=1) / vmax, 0, 1)
        lab = zoom(lab, k, order=0)
        rgb = np.stack([g * 238, g * 232, g * 220], -1)  # faintly warm film tone
        for code, col in COLOURS.items():
            m = lab == code
            edge = m & ~binary_erosion(m, iterations=2)
            rgb[m] = rgb[m] * 0.82 + np.array(col) * 0.18
            rgb[edge] = col
        frames.append(rgb.astype(np.uint8))
        world_z = float((flair_img.affine @ [0, 0, z, 1])[2])
        meta.append({"z_index": int(z), "z_mm": round(world_z, 1),
                     "area_mm2": {str(c): float((pred[:, :, z] == c).sum() * sp[0] * sp[1]) for c in COLOURS}})

    h = min(f.shape[0] for f in frames)
    sprite = np.concatenate([f[:h] for f in frames], axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sprite).save(OUT / "plate.jpg", quality=84, optimize=True, progressive=True)
    pmeta = json.loads((case_dir / "prediction_meta.json").read_text(encoding="utf-8"))
    (OUT / "plate.json").write_text(json.dumps({
        "case": "BRATS_001 · Medical Segmentation Decathlon Task01 (CC BY-SA 4.0)",
        "frame_width": WIDTH, "frame_height": h, "frames": meta,
        "volumes_mL": {k: round(v["mL"], 1) for k, v in pmeta["estimated_volumes"].items()},
    }, indent=1), encoding="utf-8")
    print(f"{len(frames)} frames {WIDTH}x{h} -> {OUT / 'plate.jpg'} ({(OUT / 'plate.jpg').stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
