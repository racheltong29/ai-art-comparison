"""Web API for artwork AI-likeness analysis."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.detector import AnalysisResult, ArtLikenessDetector

STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}

app = FastAPI(
    title="Art Originality Checker",
    description="Estimate how much uploaded artwork resembles AI-generated imagery.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_detector() -> ArtLikenessDetector:
    return ArtLikenessDetector()


def result_to_dict(result: AnalysisResult) -> dict:
    return {
        "ai_probability": result.ai_probability,
        "human_probability": result.human_probability,
        "predicted_label": result.predicted_label,
        "originality_score": result.originality_score,
        "ai_likeness_percent": round(result.ai_probability * 100, 1),
        "model_id": result.model_id,
        "device": result.device,
    }


@app.get("/api/health")
def health() -> dict:
    detector_ready = False
    device = "unknown"
    try:
        detector = get_detector()
        detector_ready = True
        device = str(detector.device)
    except Exception:  # noqa: BLE001
        pass
    return {"status": "ok", "model_loaded": detector_ready, "device": device}


@app.post("/api/analyze")
async def analyze_artwork(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type or 'unknown'}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 15 MB limit.")

    try:
        result = get_detector().analyze(data)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Could not analyze image: {exc}") from exc

    return result_to_dict(result)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
