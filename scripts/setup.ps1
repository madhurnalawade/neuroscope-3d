# One-time setup on Windows (PowerShell). Usage:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126
& $py -m pip install -r requirements.txt
& $py -c "from monai.bundle import download; download(name='brats_mri_segmentation', version='0.5.4', bundle_dir='models')"
Write-Host "Setup complete. Start the platform with:  .\.venv\Scripts\python.exe run.py"
