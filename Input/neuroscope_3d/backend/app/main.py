"""
FastAPI application entrypoint for NeuroScope 3D.
Provides NIfTI upload & validation API, segmentation routing, health checks,
and serves the React web interface.
"""

import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import app.core.paths  # noqa: F401
from app.api import backup, privacy, segmentation, upload, viewer
from app.core.exceptions import NIfTIValidationError
from app.services.study_store import study_store

from brainseg.constants import DISCLAIMER

mimetypes.add_type("image/webp", ".webp")  # not in the Windows registry everywhere; used by the loader images

app = FastAPI(
    title="Voxels — research prototype",
    description=f"**{DISCLAIMER}**\n\nUpload four co-registered MRI sequences, run the pretrained MONAI SegResNet, "
                "view suspected tumour regions predicted by the model in 3D and linked slices, and download the "
                "NIfTI label map (BraTS codes 1 NCR, 2 ED, 4 ET) and a report.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=4096)


@app.on_event("startup")
def start_upload_sweeper():
    """Reload persisted studies and delete uploads older than the TTL (< 24 h), now and every 10 min."""
    study_store.start_sweeper()


@app.exception_handler(NIfTIValidationError)
async def nifti_validation_exception_handler(request: Request, exc: NIfTIValidationError):
    """Centralized handler for custom NIfTI validation errors."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


# Include API routers
app.include_router(upload.router)
app.include_router(segmentation.router)
app.include_router(viewer.router)
app.include_router(backup.router)
app.include_router(privacy.router)
app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

# Paths for frontend web app
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST / "index.html"


@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint verifying backend service status."""
    return {
        "status": "healthy",
        "service": "Voxels",
        "version": "1.0.0",
        "validation_engine": "active",
        "ready_for_uploads": True,
    }


if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")


STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"
STATIC_ANALYSE = Path(__file__).resolve().parent / "static" / "analyse.html"


@app.get("/api/about", tags=["System"])
def about():
    """Model, device, evaluation summary and limitations, for the landing page."""
    import torch

    from app.services.report import load_evaluation
    from brainseg.constants import LIMITATIONS, MODEL_INFO

    ev = load_evaluation()
    return {
        "model": MODEL_INFO,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "cuda": torch.cuda.is_available(),
        "evaluation": {k: ev[k] for k in ("n_cases", "case_ids", "summary", "overlap_disclosure", "dataset")} if ev else None,
        "limitations": LIMITATIONS,
        "disclaimer": DISCLAIMER,
        "upload_ttl_hours": round(study_store.ttl_seconds / 3600, 1),
    }


@app.get("/analyse", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_analyse():
    """Serve the analysis workflow: drop a study, run the model, open the viewer and report."""
    return FileResponse(str(STATIC_ANALYSE), headers={"Cache-Control": "no-cache"})


@app.get("/legacy", response_class=HTMLResponse, include_in_schema=False)
async def serve_legacy():
    """The earlier four-slot React upload page, kept for reference."""
    return FileResponse(str(FRONTEND_INDEX), headers={"Cache-Control": "no-cache"})


@app.get("/", response_class=HTMLResponse, tags=["Web Interface"])
async def serve_root():
    """Serve the landing page + analysis workflow."""
    if STATIC_INDEX.exists():
        return FileResponse(str(STATIC_INDEX), headers={"Cache-Control": "no-cache"})
    if FRONTEND_INDEX.exists():
        return FileResponse(str(FRONTEND_INDEX), headers={"Cache-Control": "no-cache"})

    fallback_index = PROJECT_ROOT / "frontend" / "index.html"
    if fallback_index.exists():
        return FileResponse(str(fallback_index))

    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html>
        <head><title>Voxels</title></head>
        <body style="font-family:sans-serif;background:#0b0f19;color:#f9fafb;padding:2rem;">
            <h1>Voxels API is Running</h1>
            <p>Interactive API documentation is available at <a href="/docs" style="color:#06b6d4;">/docs</a>.</p>
        </body>
        </html>
        """
    )
