"""Run the pretrained MONAI SegResNet on one case and return a BraTS label map on the input grid."""
import copy
import datetime
import hashlib
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import Compose, NormalizeIntensity, Orientation, Spacing

from .constants import (  # noqa: F401
    CHECKPOINT, DISCLAIMER, MODEL_INFO, MODEL_INPUT_ORDER, MSD_CHANNEL_ORDER, PREPROCESSING, ROOT,
)
from .labels import channels_to_brats, region_volumes

ROI_SIZES = [(240, 240, 160), (128, 128, 128)]  # largest first; smaller fallback for low-memory GPUs
UNCERTAINTY_FILENAME = "uncertainty_tta.nii.gz"  # per voxel: how many of the 4 TTA passes call it WT


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    try:
        git = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        commit = git("rev-parse", "HEAD")
        return commit + ("-dirty (uncommitted changes)" if git("status", "--porcelain", "--untracked-files=no") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def _load_3d(path) -> nib.Nifti1Image:
    img = nib.load(str(path), mmap=False)
    if img.ndim == 4 and img.shape[3] == 1:
        img = img.slicer[..., 0]
    if img.ndim != 3:
        raise ValueError(f"{Path(path).name}: expected a 3D volume, got shape {img.shape}")
    return img


def load_sequences(paths: dict):
    """Load four co-registered 3D sequences. Returns (stack[4,X,Y,Z] in model order, reference image)."""
    missing = [s for s in MODEL_INPUT_ORDER if s not in paths]
    if missing:
        raise ValueError(f"Missing sequences: {missing}")
    imgs = {s: _load_3d(paths[s]) for s in MODEL_INPUT_ORDER}
    ref = imgs["t1gd"]
    for s, img in imgs.items():
        if img.shape != ref.shape:
            raise ValueError(f"Shape mismatch: {s} {img.shape} vs t1gd {ref.shape}")
        if not np.allclose(img.affine, ref.affine, atol=1e-3):
            raise ValueError(f"Affine mismatch between {s} and t1gd")
    stack = np.stack([np.asarray(imgs[s].dataobj, dtype=np.float32) for s in MODEL_INPUT_ORDER])
    if not np.isfinite(stack).all():
        raise ValueError("Input contains non-finite values")
    return stack, ref


def load_msd_4d(path):
    """Load an MSD Task01 4D image (channels FLAIR, T1w, T1Gd, T2w) as (stack in model order, reference)."""
    img = nib.load(str(path), mmap=False)
    if img.ndim != 4 or img.shape[3] != 4:
        raise ValueError(f"Expected (X, Y, Z, 4), got {img.shape}")
    data = np.asarray(img.dataobj, dtype=np.float32)
    stack = np.stack([data[..., MSD_CHANNEL_ORDER.index(s)] for s in MODEL_INPUT_ORDER])
    ref = nib.Nifti1Image(np.zeros(img.shape[:3], np.uint8), img.affine, img.header.copy())
    ref.header.set_data_shape(img.shape[:3])
    return stack, ref


_MODEL_CACHE = {}


def load_model(device) -> torch.nn.Module:
    key = str(device)
    if key not in _MODEL_CACHE:
        net = SegResNet(
            blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1], init_filters=16,
            in_channels=4, out_channels=3, dropout_prob=0.2,
        )
        state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        net.load_state_dict(state.get("model", state))
        _MODEL_CACHE[key] = net.to(device).eval()
    return _MODEL_CACHE[key]


def _infer(x: torch.Tensor, device) -> torch.Tensor:
    """Sliding-window inference with automatic fallback to a smaller window, then CPU, on out-of-memory."""
    attempts = [(device, roi) for roi in ROI_SIZES]
    if device.type == "cuda":
        attempts.append((torch.device("cpu"), ROI_SIZES[0]))
    last_err = None
    for dev, roi in attempts:
        try:
            net = load_model(dev)
            with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logits = sliding_window_inference(x[None].to(dev), roi, 1, net, overlap=0.5)
            return torch.sigmoid(logits[0].float()).cpu(), f"{dev.type} roi={roi}"
        except torch.OutOfMemoryError as e:
            last_err = e
            torch.cuda.empty_cache()
    raise RuntimeError(f"Inference failed on all devices: {last_err}")


TTA_PASSES = (("none", ()), ("flip L-R", (1,)), ("flip A-P", (2,)), ("flip S-I", (3,)))


def predict(stack: np.ndarray, affine: np.ndarray, device=None, tta: bool = False):
    """Return (BraTS uint8 label map on the input grid, info dict).

    The label map always comes from the plain (un-augmented) pass. With tta=True the network is also run on
    the volume mirrored along each axis (mirrored back afterwards); info["tta"] then holds the per-pass label
    maps' agreement count (0-4, on the input grid) and per-pass volumes, as an uncertainty estimate."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spatial = Compose([
        Orientation(axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ])
    # torch.tensor copies: no-op spatial transforms keep shared storage and normalisation is in place.
    img = spatial(MetaTensor(torch.tensor(stack), affine=torch.as_tensor(affine, dtype=torch.float64)))
    x = NormalizeIntensity(nonzero=True, channel_wise=True)(img.as_tensor())

    def to_labels(probs_canonical):
        pred = MetaTensor(probs_canonical, meta=img.meta)
        pred.applied_operations = copy.deepcopy(img.applied_operations)  # inverse() pops from this list
        probs = spatial.inverse(pred).as_tensor().numpy()
        if probs.shape[1:] != stack.shape[1:]:
            raise RuntimeError(f"Restored grid {probs.shape[1:]} != input grid {stack.shape[1:]}")
        tc, wt, et = (probs[i] > 0.5 for i in range(3))
        return channels_to_brats(tc, wt, et)

    passes = TTA_PASSES if tta else TTA_PASSES[:1]
    label_maps, where = [], None
    for _, dims in passes:
        xi = torch.flip(x, dims) if dims else x
        probs, where = _infer(xi, device)
        label_maps.append(to_labels(torch.flip(probs, dims) if dims else probs))

    info = {"device": where}
    if tta:
        info["tta"] = {"passes": [name for name, _ in passes], "label_maps": label_maps}
    return label_maps[0], info


def tta_summary(label_maps, spacing) -> tuple:
    """Agreement map (number of passes calling each voxel WT) and per-region volume spread / overlap."""
    from .labels import region_masks
    from .metrics import dice

    masks = [region_masks(m) for m in label_maps]
    agree = np.sum([m["WT"] for m in masks], axis=0).astype(np.uint8)
    n = len(label_maps)
    summary = {"passes": n, "regions": {}}
    for r in ("WT", "TC", "ET"):
        vols = [region_volumes(m, spacing)[r]["mL"] for m in label_maps]
        summary["regions"][r] = {
            "volume_mL_per_pass": [round(v, 3) for v in vols],
            "volume_mL_min": round(min(vols), 3), "volume_mL_max": round(max(vols), 3),
            "dice_vs_main_per_pass": [round(dice(masks[k][r], masks[0][r]), 4) for k in range(1, n)],
        }
    any_wt, all_wt = agree > 0, agree == n
    summary["wt_uncertain_fraction"] = round(float((any_wt & ~all_wt).sum() / max(any_wt.sum(), 1)), 4)
    summary["wt_unanimous_mL"] = round(float(all_wt.sum() * np.prod(spacing[:3]) / 1000), 3)
    summary["wt_any_mL"] = round(float(any_wt.sum() * np.prod(spacing[:3]) / 1000), 3)
    return agree, summary


def modality_ablation(stack: np.ndarray, affine: np.ndarray, spacing, device=None) -> dict:
    """Re-run the model with each input sequence zeroed and compare with the full-input prediction.
    Shows which sequences this model relies on for this case. NOT clinical evidence."""
    from .metrics import dice
    from .labels import region_masks

    base, _ = predict(stack, affine, device)
    base_m, base_v = region_masks(base), region_volumes(base, spacing)
    out = {}
    for i, seq in enumerate(MODEL_INPUT_ORDER):
        s = stack.copy()
        s[i] = 0
        lab, _ = predict(s, affine, device)
        m, v = region_masks(lab), region_volumes(lab, spacing)
        out[seq] = {r: {"dice_vs_full": round(dice(m[r], base_m[r]), 4), "volume_mL": round(v[r]["mL"], 3),
                        "volume_change_mL": round(v[r]["mL"] - base_v[r]["mL"], 3)} for r in ("WT", "TC", "ET")}
    return {"label": "Modality ablation — not clinical evidence", "removed_sequence": out,
            "method": "Each sequence in turn is replaced by zeros before inference; Dice compares the result with "
                      "the full-input prediction. It shows what this model relies on, not what is in the patient."}


def _save_like(arr, ref, path):
    header = ref.header.copy()
    header.set_data_dtype(np.uint8)
    header.set_slope_inter(1, 0)
    out = nib.Nifti1Image(arr.astype(np.uint8), ref.affine, header)
    out.set_qform(ref.get_qform(), code=int(ref.header["qform_code"]) or 1)
    out.set_sform(ref.get_sform(), code=int(ref.header["sform_code"]) or 1)
    nib.save(out, str(path))


def run_case(stack, ref: nib.Nifti1Image, out_path, input_files: dict, device=None, tta: bool = True) -> dict:
    """Predict, save the NIfTI label map (and TTA agreement map) on the input grid/affine, return the record."""
    t0 = time.perf_counter()
    labels, info = predict(stack, ref.affine, device, tta=tta)
    elapsed = time.perf_counter() - t0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_like(labels, ref, out_path)

    spacing = [float(z) for z in ref.header.get_zooms()[:3]]
    uncertainty = None
    if "tta" in info:
        agree, uncertainty = tta_summary(info["tta"]["label_maps"], spacing)
        uncertainty["passes_names"] = info["tta"]["passes"]
        uncertainty["agreement_file"] = UNCERTAINTY_FILENAME
        _save_like(agree, ref, out_path.with_name(UNCERTAINTY_FILENAME))
    return {
        "uncertainty": uncertainty,
        "prediction_file": out_path.name,
        "label_codes": {"0": "background", "1": "NCR", "2": "ED", "4": "ET"},
        "shape": list(labels.shape),
        "spacing_mm": spacing,
        "affine": ref.affine.tolist(),
        "orientation": "".join(nib.aff2axcodes(ref.affine)),
        "estimated_volumes": region_volumes(labels, spacing),
        "provenance": {
            "model": MODEL_INFO,
            "preprocessing": PREPROCESSING,
            "inputs": {k: {"file": Path(v).name, "sha256": sha256(v)} for k, v in input_files.items()},
            "code_commit": code_commit(),
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "device": info["device"],
            "inference_seconds": round(elapsed, 2),
            "test_time_augmentation": "4 passes (none, flip L-R, A-P, S-I); label map from the un-augmented pass"
            if tta else "off",
        },
        "disclaimer": DISCLAIMER,
    }
