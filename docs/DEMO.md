# 5-minute demo script + viva prep

**Before the slot:** `python run.py` running; browser at http://localhost:8000/; a file explorer open on
`demo_cases/BRATS_002/` (four files + `_seg` mask) and on one Decathlon 4D file; a terminal in the repo root.
Backup: http://localhost:8000/backup (every screen is bannered "PRECOMPUTED BACKUP CASE … NOT a live run").

| Time | Show | Say (key points) |
|---|---|---|
| 0:00–0:30 | Landing page: safety band, hero sweep | "Ishita's trainee scrolls for ten minutes to build the tumour in his head. This shows the *suspected tumour region predicted by the model*, in 3D, with an *estimated volume* — and how far to trust it. Never a diagnosis." |
| 0:30–1:15 | Drop the four BraTS files + `_seg` into the drop zone | Roles are read from filenames; grid and spacing are read from the headers in the browser before upload. Then drop a single 4-channel file to show it is split on the server. *Run analysis*; the run log times each step. |
| 1:15–1:45 | Result card | Five volumes with N × voxel volume; the trust line (volume range over 4 mirrored passes, % disputed voxels); agreement with the expert mask. |
| 1:45–3:15 | *Open 3D & slice viewer* | Rotate / zoom; toggle and fade each region; click a slice → crosshair moves in all three views and in 3D; voxel indices + world mm; L/R, A/P, S/I. Overlay → *Agreement — WT*, then *Uncertainty (TTA)*. *Run modality ablation*: without T1Gd the enhancing region disappears — "what the model relies on, not clinical evidence". |
| 3:15–4:15 | *Report* → PDF | Volumes and formula, trust section, evaluation on 20 held-out cases with the overlap disclosure, provenance (model, version, licence, preprocessing, SHA-256, commit, timestamp), limitations, disclaimer on every page. |
| 4:15–5:00 | Terminal | `python -m brainseg.predict --t1 … --out pred.nii.gz` (the scorer's command); `python tests/judge_scenarios.py …` 8/8. Close on the limits section. |

## Likely viva questions

- **Label codes?** BraTS 1 NCR, 2 ED, 4 ET (category IDs, not severity). Network outputs overlapping TC/WT/ET;
  ET→4, TC∖ET→1, WT∖TC→2. Decathlon labels (1 oedema, 2 non-enhancing, 3 enhancing) → 2, 1, 4.
- **Original grid?** Reorient to RAS + resample to 1 mm with recorded MONAI transforms, infer, invert the same
  transforms on the probabilities, threshold; save with the uploaded affine. Tested on flipped, 1×1×2 mm and shifted grids.
- **Orientation tests?** Same anatomy stored flipped → same side, WT +0.00 %. Mirrored anatomy → side changes.
- **How do you measure trust?** Held-out Dice/sensitivity/HD95 (optimistic — overlap disclosed); per-case flip TTA
  (volume range, disputed voxels, overlay); agreement with an expert mask if given; modality ablation.
- **Limits of TTA?** It only measures sensitivity to mirroring; a confidently wrong region is unanimous.
- **Is the evaluation independent?** No — Decathlon Task01 derives from BraTS 2016/17, overlapping the model's
  BraTS 2018 training data; case-level overlap cannot be determined. We did no training.
- **Bad inputs?** Rejected with a reason before the model: unreadable, not 3D, NaN/Inf, constant, > 1 GB, missing or
  duplicate sequence, shape/affine mismatch, non-skull-stripped. Non-brain-like inputs get size/shape/texture warnings.
- **Data protection?** Public de-identified data only; nothing leaves the machine; uploads deleted < 24 h.
