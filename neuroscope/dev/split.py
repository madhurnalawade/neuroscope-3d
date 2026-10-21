"""Reproducible patient-level split of the labelled Decathlon cases (fixed seed)."""
import datetime
import random

from .msd import read_dataset

MIN_HELDOUT = 10

PATIENT_CAVEAT = (
    "The Decathlon release provides no patient identifiers, so this split assumes one case ID = one patient. "
    "That cannot be confirmed from the dataset: if one patient contributed two studies under different IDs, "
    "they could end up on both sides of the split.")
OVERLAP_REMINDER = (
    "Disclose: the pretrained MONAI brats_mri_segmentation bundle was trained on BraTS 2018, which re-uses "
    "BraTS 2016/2017 cases that the Decathlon set is built from. Held-out cases here may have been seen by "
    "the pretrained model, so metrics on them are likely optimistic.")


def make_split(dataset_root, n_heldout: int = 20, n_dev: int = 20, seed: int = 20261017) -> dict:
    if n_heldout < MIN_HELDOUT:
        raise ValueError(f"At least {MIN_HELDOUT} held-out cases are required (got {n_heldout}).")
    info = read_dataset(dataset_root)
    labelled = sorted(c.case_id for c in info.cases if c.label and (info.root / c.label).is_file()
                      and (info.root / c.image).is_file())
    if len(labelled) < n_heldout + n_dev + 1:
        raise ValueError(f"Only {len(labelled)} labelled cases; cannot hold out {n_heldout} + {n_dev}.")
    order = list(labelled)
    random.Random(seed).shuffle(order)
    heldout, dev, train = sorted(order[:n_heldout]), sorted(order[n_heldout:n_heldout + n_dev]), \
        sorted(order[n_heldout + n_dev:])
    return {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "dataset_root_name": info.root.name,
        "seed": seed,
        "method": "sorted labelled case IDs shuffled with random.Random(seed); first n_heldout = held-out, "
                  "next n_dev = dev, rest = train",
        "counts": {"labelled": len(labelled), "train": len(train), "dev": len(dev), "heldout": len(heldout)},
        "rules": "Never train or tune on held-out cases. Use dev for any threshold or setting choices.",
        "patient_level_caveat": PATIENT_CAVEAT,
        "pretraining_overlap_reminder": OVERLAP_REMINDER,
        "heldout": heldout,
        "dev": dev,
        "train": train,
    }
