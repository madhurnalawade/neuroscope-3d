# Model card — Voxels tumour-region segmentation

> Research and teaching prototype. Outputs are **suspected tumour regions predicted by the model** and
> **estimated volumes**. Not a diagnosis; not clinically validated; not for clinical use.

| | |
|---|---|
| **Model** | MONAI Model Zoo `brats_mri_segmentation` — SegResNet (Myronenko 2018), 3D, 4 input channels, 3 sigmoid outputs (TC, WT, ET) |
| **Version** | 0.5.4 (bundle), weights `models/brats_mri_segmentation/models/model.pt`, public before the event |
| **Licence** | Apache-2.0, © MONAI Consortium (`models/brats_mri_segmentation/LICENSE`); data licence notes in `docs/data_license.txt` of the bundle |
| **Source** | https://huggingface.co/MONAI/brats_mri_segmentation |
| **Training data** | BraTS 2018 training set (285 pre-operative adult glioma cases), split by the MONAI authors into 200 train / 42 val / 43 test. **We did not train or fine-tune** — weights are used unchanged. |
| **Intended use** | Teaching and research demos on public, de-identified, skull-stripped, co-registered brain MRI (T1, T1Gd, T2, FLAIR). |
| **Out of scope** | Diagnosis, grading, genetics, prognosis, treatment advice, surgical or radiotherapy planning, any clinical decision. |

## Input → output

- Input: four co-registered 3D NIfTI sequences on one grid. Channel order fed to the network: **T1Gd, T1, T2, FLAIR**.
- Preprocessing: reorient to RAS → resample to 1 mm isotropic (bilinear) → per-channel z-score over non-zero voxels →
  sliding-window inference (ROI 240×240×160, overlap 0.5, AMP) → sigmoid → invert resampling/orientation to the input
  grid (bilinear on probabilities) → threshold 0.5.
- Output: `uint8` NIfTI on the **input grid and affine**, BraTS codes **1 = NCR, 2 = ED, 4 = ET** (ET → 4; TC without
  ET → 1; WT without TC → 2). Regions: WT = {1,2,4}, TC = {1,4}, ET = {4}.
- Volumes: voxel count × voxel volume (product of header spacing), mm³ → mL ÷ 1000.
- Uncertainty: 3 extra test-time-augmentation passes (input mirrored L-R, A-P, S-I, output mirrored back). The label
  map always comes from the plain pass; TTA gives a per-voxel WT agreement map (0–4) and per-region volume ranges.
- Input checks before inference: structural (readable, 3D, finite, non-constant, ≤ 1 GB, matching shape/affine);
  refusal if not skull-stripped (zero background < 5 % or tissue > 2.6 L); warnings for implausible size, brain-unlike
  shape (bounding-box fill outside 0.36–0.52) or unusually smooth texture — thresholds calibrated on 41 Decathlon brains.

## Evaluation

20 held-out labelled Medical Segmentation Decathlon Task01 cases (random, seed 2026; list in `evaluation/RESULTS.md`):

| Region | Dice mean ± SD | Sensitivity mean | HD95 median |
|---|---|---|---|
| WT | 0.885 ± 0.108 | 0.932 | 4.6 mm |
| TC | 0.851 ± 0.212 | 0.912 | 2.2 mm |
| ET | 0.764 ± 0.274 | 0.857 | 1.7 mm |

Bundle authors' own validation Dice (BraTS 2018): TC 0.856, WT 0.903, ET 0.791.

**Overlap disclosure.** Decathlon Task01 is derived from BraTS 2016/2017, which BraTS 2018 largely re-uses; case IDs
were renamed, so overlap with the model's training data cannot be ruled out per case and is likely for most cases.
Our metrics are therefore optimistic, not an independent estimate.

## Limitations and expected failure modes

- Trained on pre-operative adult glioma only: may fail on metastases, meningioma, paediatric or post-operative scans.
- Requires skull-stripped, co-registered sequences; heuristic checks refuse clearly non-stripped inputs and warn on
  borderline ones, but cannot guarantee suitability.
- Small or low-contrast regions (ET, NCR) are least reliable; ET may be missed or hallucinated. Observed in our
  evaluation: on non-enhancing tumours the model predicted enhancing tumour / tumour core where the expert marked
  none (BRATS_175: WT Dice 0.47, TC 0.06, ET 0.00; BRATS_169: ET Dice 0.01).
- Non-1 mm or thick-slice data is resampled; accuracy and volume precision drop.
- Volumes inherit all segmentation errors; they are estimates of predicted regions, not tumour measurements.
- Sigmoid outputs are not calibrated probabilities. Flip-TTA agreement measures sensitivity to mirroring only; it
  under-states true error (a confidently wrong region is unanimous across passes).
