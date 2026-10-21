"""Evaluate the pipeline on held-out labelled Medical Segmentation Decathlon Task01 cases.

    python scripts/evaluate.py --data-dir data/Task01_BrainTumour --n 20

Runs exactly the same inference code as the app (brainseg.inference.predict) on each case, converts the
Decathlon expert labels to BraTS codes, and reports Dice, sensitivity and HD95 per region (WT/TC/ET).
Writes evaluation/metrics.json, evaluation/per_case.csv, evaluation/split.json and evaluation/RESULTS.md.
"""
import argparse
import csv
import datetime
import json
import statistics
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from brainseg.constants import MODEL_INFO, PREPROCESSING  # noqa: E402
from brainseg.inference import code_commit, load_msd_4d, predict  # noqa: E402
from brainseg.labels import msd_to_brats  # noqa: E402
from brainseg.metrics import case_metrics  # noqa: E402

OVERLAP_DISCLOSURE = (
    "We did not train or fine-tune the model, so none of these cases were used by us for training or tuning. "
    "However, the pretrained MONAI model was trained on BraTS 2018, and Decathlon Task01 is built from BraTS 2016 "
    "and 2017 cases, which BraTS 2018 largely re-uses. Decathlon renamed the cases, so we cannot tell which "
    "evaluation cases the model saw during its training; most probably overlap. These metrics are therefore "
    "likely optimistic and are not an independent held-out estimate."
)
SPLIT_NOTE = (
    "Patient-wise split: Decathlon Task01 provides one multi-sequence study per subject (BRATS_xxx), so each "
    "case ID is one patient. Evaluation cases were sampled at random (fixed seed) from the labelled training "
    "list. Our training/fine-tuning set is empty (pretrained model used as-is). Judge-provided cases are never "
    "used for training or tuning and uploads are deleted within 24 h."
)


def summarise(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None, None
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True, help="Folder with imagesTr/, labelsTr/, dataset.json")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--cases", nargs="*", help="Explicit case IDs instead of random sampling")
    ap.add_argument("--out", type=Path, default=ROOT / "evaluation")
    args = ap.parse_args()

    meta = json.loads((args.data_dir / "dataset.json").read_text(encoding="utf-8"))
    available = sorted(
        Path(t["image"]).name.replace(".nii.gz", "")
        for t in meta["training"]
        if (args.data_dir / "imagesTr" / Path(t["image"]).name).exists()
        and (args.data_dir / "labelsTr" / Path(t["label"]).name).exists()
    )
    if args.cases:
        cases = args.cases
    else:
        rng = np.random.default_rng(args.seed)
        cases = sorted(rng.choice(available, size=min(args.n, len(available)), replace=False).tolist())
    print(f"{len(available)} labelled cases available; evaluating {len(cases)}: {', '.join(cases)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, per_case = [], {}
    for n, cid in enumerate(cases, 1):
        t0 = time.perf_counter()
        stack, ref = load_msd_4d(args.data_dir / "imagesTr" / f"{cid}.nii.gz")
        labels, info = predict(stack, ref.affine, device)
        gt_img = nib.load(str(args.data_dir / "labelsTr" / f"{cid}.nii.gz"), mmap=False)
        gt = msd_to_brats(np.asarray(gt_img.dataobj))
        assert gt.shape == labels.shape
        spacing = [float(z) for z in ref.header.get_zooms()[:3]]
        m = case_metrics(labels, gt, spacing)
        secs = time.perf_counter() - t0
        per_case[cid] = {"metrics": m, "seconds": round(secs, 2), "device": info["device"]}
        for r, v in m.items():
            rows.append({"case": cid, "region": r, **{k: v[k] for k in ("dice", "sensitivity", "hd95_mm", "pred_voxels", "ref_voxels")}})
        print(f"[{n:>2}/{len(cases)}] {cid}: " + "  ".join(
            f"{r} Dice {v['dice']:.3f} Sens {('%.3f' % v['sensitivity']) if v['sensitivity'] is not None else '  — '}"
            for r, v in m.items()) + f"  ({secs:.1f} s)")

    summary = {}
    for r in ("WT", "TC", "ET"):
        d = [per_case[c]["metrics"][r]["dice"] for c in cases]
        s = [per_case[c]["metrics"][r]["sensitivity"] for c in cases]
        h = [per_case[c]["metrics"][r]["hd95_mm"] for c in cases]
        dm, dsd, dmed = summarise(d)
        sm, ssd, _ = summarise(s)
        _, _, hmed = summarise(h)
        summary[r] = {
            "dice_mean": dm, "dice_sd": dsd, "dice_median": dmed,
            "sensitivity_mean": sm, "sensitivity_sd": ssd,
            "sensitivity_undefined_cases": sum(v is None for v in s),
            "hd95_median_mm": hmed, "hd95_undefined_cases": sum(v is None for v in h),
        }

    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset": "Medical Segmentation Decathlon Task01_BrainTumour (labelled training list), labels converted to BraTS codes",
        "n_cases": len(cases),
        "case_ids": cases,
        "sampling": {"seed": args.seed, "n_available": len(available)} if not args.cases else {"explicit": True},
        "summary": summary,
        "per_case": per_case,
        "metric_definitions": {
            "dice": "2|P∩R|/(|P|+|R|); 1.0 when both empty",
            "sensitivity": "|P∩R|/|R|; undefined (excluded from mean) when the reference region is empty",
            "hd95_mm": "max of the two directed 95th-percentile surface distances, in mm; undefined if exactly one is empty",
        },
        "model": MODEL_INFO,
        "preprocessing": PREPROCESSING,
        "code_commit": code_commit(),
        "overlap_disclosure": OVERLAP_DISCLOSURE,
        "split_note": SPLIT_NOTE,
    }
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "split.json").write_text(json.dumps({
        "evaluation_cases": cases, "training_or_tuning_cases_used_by_us": [],
        "model_pretraining_data": MODEL_INFO["training_data"], "note": SPLIT_NOTE,
        "overlap_disclosure": OVERLAP_DISCLOSURE,
    }, indent=2), encoding="utf-8")
    with open(args.out / "per_case.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    md = [f"# Evaluation — {len(cases)} held-out labelled cases", "",
          f"Dataset: {result['dataset']}. Generated {result['generated_utc']}, code {result['code_commit']}.", "",
          "| Region | Dice mean ± SD | Dice median | Sensitivity mean ± SD | HD95 median (mm) |",
          "|---|---|---|---|---|"]
    for r, s in summary.items():
        hd = s["hd95_median_mm"]
        md.append(f"| {r} | {f3(s['dice_mean'])} ± {f3(s['dice_sd'])} | {f3(s['dice_median'])} | "
                  f"{f3(s['sensitivity_mean'])} ± {f3(s['sensitivity_sd'])} | {'—' if hd is None else f'{hd:.1f}'} |")
    md += ["", "## Cases (patient-wise split)", "", ", ".join(cases), "", SPLIT_NOTE, "",
           "## Overlap with training data", "", OVERLAP_DISCLOSURE, "",
           "## Per case", "", "| Case | WT Dice | TC Dice | ET Dice | WT Sens | TC Sens | ET Sens |", "|---|---|---|---|---|---|---|"]
    for c in cases:
        m = per_case[c]["metrics"]
        md.append(f"| {c} | " + " | ".join(f3(m[r]["dice"]) for r in ("WT", "TC", "ET")) + " | "
                  + " | ".join(f3(m[r]["sensitivity"]) for r in ("WT", "TC", "ET")) + " |")
    (args.out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {args.out / 'metrics.json'} and RESULTS.md")
    for r, s in summary.items():
        print(f"  {r}: Dice {f3(s['dice_mean'])} ± {f3(s['dice_sd'])}, sensitivity {f3(s['sensitivity_mean'])}, "
              f"HD95 median {s['hd95_median_mm']}")


if __name__ == "__main__":
    main()
