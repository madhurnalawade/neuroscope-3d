"""Start NeuroScope 3D with one command:  python run.py  [--host 0.0.0.0] [--port 8000]

Downloads the pretrained MONAI weights on first run if they are missing, then serves the upload page,
viewer, report and API at http://<host>:<port>/ (API docs at /docs).
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "Input" / "neuroscope_3d" / "backend"
sys.path[:0] = [str(BACKEND), str(ROOT)]


def ensure_weights():
    from brainseg.constants import BUNDLE_DIR, CHECKPOINT
    if CHECKPOINT.exists():
        return
    print(f"Model weights not found at {CHECKPOINT}; downloading MONAI 'brats_mri_segmentation' bundle...")
    from monai.bundle import download
    download(name="brats_mri_segmentation", version="0.5.4", bundle_dir=str(BUNDLE_DIR.parent))
    if not CHECKPOINT.exists():
        sys.exit(f"Download finished but {CHECKPOINT} is missing.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    ensure_weights()
    os.environ["PYTHONPATH"] = os.pathsep.join([str(BACKEND), str(ROOT), os.environ.get("PYTHONPATH", "")])
    os.chdir(BACKEND)
    import torch
    print(f"PyTorch {torch.__version__}; CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else " — running on CPU (slower)"))
    print(f"NeuroScope 3D: open http://localhost:{args.port}/   (API docs: http://localhost:{args.port}/docs)")
    import uvicorn
    uvicorn.run("app.main:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
