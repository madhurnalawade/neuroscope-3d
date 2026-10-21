# Evaluation — 20 held-out labelled cases

Dataset: Medical Segmentation Decathlon Task01_BrainTumour (labelled training list), labels converted to BraTS codes. Generated 2026-10-17T00:18:40+00:00, code db28f88841d639b391be0fc5f42e00a5444b1bfb.

| Region | Dice mean ± SD | Dice median | Sensitivity mean ± SD | HD95 median (mm) |
|---|---|---|---|---|
| WT | 0.885 ± 0.108 | 0.927 | 0.932 ± 0.062 | 4.6 |
| TC | 0.851 ± 0.212 | 0.936 | 0.912 ± 0.092 | 2.2 |
| ET | 0.764 ± 0.274 | 0.858 | 0.857 ± 0.189 | 1.7 |

## Cases (patient-wise split)

BRATS_013, BRATS_038, BRATS_048, BRATS_084, BRATS_086, BRATS_145, BRATS_169, BRATS_172, BRATS_175, BRATS_220, BRATS_300, BRATS_305, BRATS_315, BRATS_337, BRATS_346, BRATS_377, BRATS_395, BRATS_397, BRATS_413, BRATS_433

Patient-wise split: Decathlon Task01 provides one multi-sequence study per subject (BRATS_xxx), so each case ID is one patient. Evaluation cases were sampled at random (fixed seed) from the labelled training list. Our training/fine-tuning set is empty (pretrained model used as-is). Judge-provided cases are never used for training or tuning and uploads are deleted within 24 h.

## Overlap with training data

We did not train or fine-tune the model, so none of these cases were used by us for training or tuning. However, the pretrained MONAI model was trained on BraTS 2018, and Decathlon Task01 is built from BraTS 2016 and 2017 cases, which BraTS 2018 largely re-uses. Decathlon renamed the cases, so we cannot tell which evaluation cases the model saw during its training; most probably overlap. These metrics are therefore likely optimistic and are not an independent held-out estimate.

## Per case

| Case | WT Dice | TC Dice | ET Dice | WT Sens | TC Sens | ET Sens |
|---|---|---|---|---|---|---|
| BRATS_013 | 0.926 | 0.952 | 0.929 | 0.939 | 0.978 | 0.966 |
| BRATS_038 | 0.942 | 0.941 | 0.927 | 0.986 | 0.936 | 0.939 |
| BRATS_048 | 0.940 | 0.963 | 0.936 | 0.905 | 0.947 | 0.909 |
| BRATS_084 | 0.842 | 0.878 | 0.772 | 0.755 | 0.810 | 0.796 |
| BRATS_086 | 0.913 | 0.769 | 0.679 | 0.940 | 0.909 | 0.893 |
| BRATS_145 | 0.944 | 0.970 | 0.932 | 0.961 | 0.982 | 0.952 |
| BRATS_169 | 0.831 | 0.590 | 0.009 | 0.980 | 0.961 | 0.115 |
| BRATS_172 | 0.969 | 0.950 | 0.943 | 0.952 | 0.943 | 0.944 |
| BRATS_175 | 0.467 | 0.061 | 0.000 | 0.983 | 0.917 | — |
| BRATS_220 | 0.877 | 0.673 | 0.855 | 0.819 | 0.573 | 0.888 |
| BRATS_300 | 0.933 | 0.916 | 0.816 | 0.905 | 0.886 | 0.903 |
| BRATS_305 | 0.908 | 0.908 | 0.777 | 0.949 | 0.931 | 0.977 |
| BRATS_315 | 0.815 | 0.882 | 0.765 | 0.884 | 0.851 | 0.948 |
| BRATS_337 | 0.839 | 0.876 | 0.620 | 0.853 | 0.874 | 0.733 |
| BRATS_346 | 0.890 | 0.942 | 0.846 | 0.988 | 0.932 | 0.856 |
| BRATS_377 | 0.928 | 0.941 | 0.861 | 0.973 | 0.971 | 0.923 |
| BRATS_395 | 0.935 | 0.932 | 0.894 | 0.962 | 0.975 | 0.897 |
| BRATS_397 | 0.932 | 0.951 | 0.881 | 0.968 | 0.973 | 0.907 |
| BRATS_413 | 0.933 | 0.946 | 0.923 | 0.969 | 0.913 | 0.880 |
| BRATS_433 | 0.939 | 0.972 | 0.909 | 0.968 | 0.969 | 0.860 |
