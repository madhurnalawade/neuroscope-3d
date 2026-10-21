#!/usr/bin/env bash
# One-time setup on Linux/macOS. Usage:  bash scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "from monai.bundle import download; download(name='brats_mri_segmentation', version='0.5.4', bundle_dir='models')"
echo "Setup complete. Start the platform with:  .venv/bin/python run.py"
