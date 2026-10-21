#!/bin/bash
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/backend:${PYTHONPATH}"  # real nibabel; lib/ shim would shadow it

cd "${PROJECT_ROOT}/backend"
echo "Starting NeuroScope 3D FastAPI server on http://0.0.0.0:8000 ..."
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
