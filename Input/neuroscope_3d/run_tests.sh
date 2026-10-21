#!/bin/bash
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/backend:${PYTHONPATH}"  # real nibabel; lib/ shim would shadow it

echo "======================================================================"
echo " Running NeuroScope 3D Test Suite"
echo "======================================================================"

cd "${PROJECT_ROOT}/backend"
python3 -m pytest tests/test_judges_failure_pack.py
python3 -m pytest tests/test_additional_validation.py
python3 -m pytest tests/test_api_integration.py

echo "======================================================================"
echo " ALL 33 TESTS COMPLETED AND VERIFIED!"
echo "======================================================================"
