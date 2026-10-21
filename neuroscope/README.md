# neuroscope — data-input layer (validation, upload storage, leak checks, dev dataset tools)

Research and teaching prototype. Nothing here makes a diagnosis or detects anything; it checks and loads
images.

| Package | Role | Used by the live pipeline? |
|---|---|---|
| `neuroscope/io/` | four-file validation, sequence assignment, cross-file checks, canonical case object | yes |
| `neuroscope/storage/` | local upload folders, expiry, automatic deletion (≤ 24 h), safety guards | yes |
| `neuroscope/security/` | network guard, log/message hygiene, static leak audit | yes |
| `neuroscope/dev/` | Medical Segmentation Decathlon adapter, export, manifest, audit, split | **never** (development and evaluation only) |

`scripts/audit_no_leaks.py` fails if production code (`neuroscope/io|storage|security`, `brainseg/`,
the FastAPI `app/`, `run.py`) imports `neuroscope.dev`.

## Requirements

Python 3.9 or newer on Windows, Linux or macOS. Packages: `nibabel`, `numpy`, `pytest` (nothing else).

```bash
pip install "nibabel>=5.2" "numpy>=1.24" "pytest>=7"
```

Pin exact versions once they work (`pip freeze > requirements-input-layer.txt`) and, for offline machines,
build a local wheel cache with `pip download -d <WHEEL_DIR> nibabel numpy pytest` and then install with
`pip install --no-index --find-links <WHEEL_DIR> nibabel numpy pytest`. `pip install` itself contacts the
package index (and runs a self-version check: set `PIP_DISABLE_PIP_VERSION_CHECK=1`); it sends no scan data.

## Commands (run from the repository root)

Placeholders: `<DATASET_ROOT>` (the Task01_BrainTumour folder), `<UPLOADS_ROOT>`, `<OUTPUT_DIR>`.
Use `python` on Windows and `python3` on Linux/macOS if that is how your Python is named; paths with spaces
need quotes. Add `--no-network` to any script to block all connections in that process.

```bash
# tests (synthetic data in temporary folders; network blocked in every test)
python -m pytest tests -q

# static leak audit (PASS/FAIL)
python scripts/audit_no_leaks.py

# installed packages and what is known about their network behaviour
python scripts/list_dependencies.py

# one case, four files (explicit sequences always win; otherwise inferred from file names)
python scripts/load_case.py --no-network --t1 <T1> --t1gd <T1GD> --t2 <T2> --flair <FLAIR> [--seg <SEG>]

# upload cleanup: list, then delete
python scripts/cleanup_uploads.py --uploads-root <UPLOADS_ROOT> --dataset-root <DATASET_ROOT> --dry-run
python scripts/cleanup_uploads.py --uploads-root <UPLOADS_ROOT> --dataset-root <DATASET_ROOT>

# development only: Decathlon dataset
python scripts/build_manifest.py   --no-network --dataset-root <DATASET_ROOT> --output-dir <OUTPUT_DIR>
python scripts/audit_dataset.py    --no-network --dataset-root <DATASET_ROOT> --output-dir <OUTPUT_DIR>
python scripts/make_split.py       --no-network --dataset-root <DATASET_ROOT> --output-dir <OUTPUT_DIR>
python scripts/export_msd_case.py  --no-network --dataset-root <DATASET_ROOT> --case BRATS_001 --output-dir <OUTPUT_DIR>
python scripts/load_case.py        --no-network --dataset-root <DATASET_ROOT> --msd-case BRATS_001
```

Instead of repeating locations you can pass `--config <FILE>` with a small JSON file
(`{"dataset_root": "...", "uploads_root": "...", "output_dir": "...", "retention_hours": 24}`; see
`neuroscope/config.example.json`). Command-line arguments override it. No location is ever assumed.

## Design decisions

- **Sequence assignment.** Explicit `--t1/--t1gd/--t2/--flair` always win. Otherwise each file name is split
  into whole alphanumeric tokens (`t1gd` never also counts as `t1`; `t1_gd` is joined first). A name naming zero
  or several sequences, two files claiming one sequence, or a missing sequence is refused. Nothing is guessed.
- **Validation order.** Size on disk (before reading) → extension → gzip magic and a full, constant-memory
  gzip read with a cap on decompressed size (catches corrupt gzip and gzip bombs) → NIfTI header → 3D
  (a single-slice volume counts as 2D) → declared data size → spacing → affine → voxel data → finite → constant.
  Every file is checked and every problem reported; an expert mask must hold BraTS 2021 codes 0/1/2/4
  (Decathlon coding is refused with a hint) and may be empty.
- **Alignment.** Shapes, voxel spacing and full affines (1e-3 mm tolerance) are compared with the FLAIR file;
  the message names the file and says whether rotation/scaling, origin or orientation differ. Equal shapes are
  not accepted as proof of alignment. Nothing is reordered, resampled, flipped or realigned.
- **Canonical case.** Arrays keyed `FLAIR`, `T1`, `T1Gd`, `T2`; shared affine, shape, spacing, orientation code,
  sanitised file names and SHA-256 per input. No model channel order is defined here: build it from the
  model's own configuration.
- **Decathlon.** Channel names (`modality`) and label names (`labels`) are read from `dataset.json`; labels
  are mapped by name (edema → 2, non-enhancing tumor → 1, enhancing tumour → 4). Unknown or duplicate names
  are errors. `._` sidecar files are ignored everywhere. Manifests read headers only; cases are loaded one at a
  time with `img.dataobj` slicing.
- **Export.** The 3D header is copied from the 4D source (qform/sform fields and codes verbatim), so the
  exported affine is bit-identical; data are written as loaded without rescaling; the files are then reloaded
  through the production loader and compared voxel by voxel.

## What is stored, where, and for how long

| What | Where | How long |
|---|---|---|
| Uploaded scans, expert masks | `<UPLOADS_ROOT>/<random 32-hex id>/` under fixed names (`T1.nii.gz` …) | until `expires_at`, at most 24 h after upload (default 24 h, configurable lower, never higher) |
| Derived files (masks, meshes, thumbnails, reports, temp files) | the same case folder (`write_derived`, `case_tempdir`) | deleted with the case |
| Library temp files | `<UPLOADS_ROOT>/_scratch/` (after `redirect_library_temp`) | deleted when older than the retention period |
| Rejected uploads | case folder | deleted as soon as the rejection is returned (`reject_case`) |
| Failed jobs | case folder | deleted at the next sweep |
| Abandoned uploads / crashed jobs (`receiving`/`processing` > 30 min) | case folder | deleted at the next sweep |
| Deletion log | `<UPLOADS_ROOT>/_audit/deletions.log` | kept; holds only time, random case id and reason |
| Aggregated metrics with no image data | wherever the application writes them | may be kept; nothing from which a scan could be rebuilt |

Expiry is read from the marker file on disk (`created_at`, `expires_at`, UTC ISO 8601), never from in-memory
timers. A sweep runs at startup (`BackgroundSweeper.start()`) and every 10 minutes; after the machine was off,
the first sweep deletes everything that expired. Cleanup refuses an uploads root that is empty, a filesystem
root, the home folder, inside or containing the dataset, or containing the repository; it deletes only direct,
non-link children with the marker file, never follows links or junctions, retries locked files and reports
failures. It refuses (unless explicitly overridden) a root that looks cloud-synced.

**Integration status:** these modules are standalone. The running FastAPI app still uses its own store
(`Input/neuroscope_3d/backend/app/services/study_store.py`, uploads in the system temp folder, 23 h TTL).
Wiring this module into the app is a separate, approved change.

**Remote hosting:** if the platform is ever run on a remote server or cloud VM, that server's disk is a third
party holding the scans. Decide that deliberately; the checks here cannot tell.

**Permissions:** folders are created 0700 and files 0600 on Linux/macOS. On Windows these modes only control
the read-only flag; access is governed by the folder's NTFS permissions, so put the uploads root in a folder
only your account can read (not a shared or public folder).

## Environment checklist (you must check these yourself; code cannot verify them)

- [ ] The uploads root is outside every cloud-synced folder (OneDrive, Dropbox, Google Drive, iCloud, Box,
      Nextcloud…), including Windows "Known Folder" redirection of Desktop/Documents to OneDrive.
- [ ] No sync client is syncing the repository folder or the dataset folder.
- [ ] No cloud backup (OneDrive backup, iCloud, Time Machine to a network drive, Backblaze…) covers the uploads root.
- [ ] No screen-sharing, remote-assistance or recording software is running while real scans are open.
- [ ] Antivirus "automatic sample submission" / cloud-delivered protection settings reviewed (they can upload
      unknown files, which a `.nii.gz` may look like).
- [ ] No browser extension or IDE assistant that uploads open files is active in the browser, editor or
      terminal used with scans (this includes AI coding assistants: share code and console text, never scans).
- [ ] The web server listens on 127.0.0.1 only unless judges must reach it over a trusted network.

## Dependency notes

`python scripts/list_dependencies.py` lists every installed distribution. For the three required packages:
numpy — no network, telemetry or update checks; nibabel — reads and writes local files, no telemetry or update
check in the functions used here (and the offline pipeline test proves the exercised paths open no
connection); pytest — no telemetry, writes a local `.pytest_cache` (test names only). Anything else installed
in the same environment (PyTorch, MONAI, huggingface_hub, uvicorn …) belongs to other parts of the platform and
is **not verified** by these checks; `huggingface_hub` and `monai.bundle` download weights over the network when
asked to (set `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` once the weights are cached).

## What these checks do and do not prove

The tests prove that, for the synthetic cases they use, validation, loading, hashing, storage, cleanup and the
dev adapter run without opening a network connection or DNS lookup through Python's socket module, and that
their logs contain no voxel values, array dumps, absolute paths or raw file names. The static audit proves that
the repository's Python, HTML, JS and CSS files contain no unexplained imports or calls of network, cloud,
telemetry, LLM or process-spawning libraries, and that production code does not import the dev package. They
do **not** prove anything about native code that opens sockets without Python's socket module, other
processes, the browser, the operating system, sync or backup clients, antivirus, other users of the machine,
code paths the tests do not exercise, or packages installed later. A sanitised short file name can still
contain an identifier if the original file name did, so use de-identified file names. "No leak found by these
checks" is the strongest statement they support.
