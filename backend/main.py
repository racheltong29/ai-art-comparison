"""Web API for artwork AI-likeness analysis."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.composition import CompositionAnalyzer, CompositionResult
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
    description="Estimate stylistic similarity to common AI vs original art aesthetics.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MAX_BATCH_FILES = 50


@lru_cache(maxsize=1)
def get_detector() -> ArtLikenessDetector:
    return ArtLikenessDetector()


@lru_cache(maxsize=1)
def get_composition_analyzer() -> CompositionAnalyzer:
    return CompositionAnalyzer()


def result_to_dict(result: AnalysisResult) -> dict:
    return {
        "score_method": result.score_method,
        "ai_aesthetic_similarity": result.ai_aesthetic_similarity,
        "original_aesthetic_similarity": result.original_aesthetic_similarity,
        "originality_score": result.originality_score,
        "ai_likeness_percent": result.ai_likeness_percent,
        "dominant_aesthetic": result.dominant_aesthetic,
        "model_id": result.model_id,
        "device": result.device,
        # Backward-compatible fields
        "ai_probability": result.ai_probability,
        "human_probability": result.human_probability,
        "predicted_label": result.predicted_label,
    }


@app.get("/api/health")
def health() -> dict:
    detector_ready = False
    device = "unknown"
    score_method = "stylistic_text_similarity"
    try:
        detector = get_detector()
        detector_ready = True
        device = str(detector.device)
    except Exception:  # noqa: BLE001
        pass
    return {
        "status": "ok",
        "model_loaded": detector_ready,
        "device": device,
        "score_method": score_method,
    }


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


def composition_to_dict(result: CompositionResult) -> dict:
    return result.to_dict()


async def _read_upload(file: UploadFile) -> bytes:
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
    return data


@app.post("/api/analyze-batch")
async def analyze_batch(
    files: list[UploadFile] = File(...),
    with_segmentation: bool = Query(False),
) -> list[dict]:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MAX_BATCH_FILES} files per batch."
        )

    detector = get_detector()
    composition_analyzer = get_composition_analyzer()
    results = []
    for file in files:
        data = await _read_upload(file)
        try:
            ai_result = detector.analyze(data)
            composition_result = composition_analyzer.analyze(
                data, with_segmentation=with_segmentation
            )
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422, detail=f"Could not analyze {file.filename}: {exc}"
            ) from exc

        entry = {"filename": file.filename}
        entry.update(result_to_dict(ai_result))
        entry.update(composition_to_dict(composition_result))
        results.append(entry)

    return results


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
