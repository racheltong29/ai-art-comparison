# Art Originality Checker (webapp)

Upload an image and get an AI-likeness score. Backend runs locally on **CPU** — no GPU required.

## Quick start

```powershell
cd c:\Users\rache\Downloads\ai-art-comparison
.\run.ps1
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Manual setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

First analysis downloads the model (~400 MB from Hugging Face).

## How scoring works

This is **stylistic similarity**, not forensic classification.

The model ([`google/siglip-base-patch16-224`](https://huggingface.co/google/siglip-base-patch16-224)) compares your image to text descriptions of:

- **AI-generic aesthetics** — smooth rendering, stock composition, diffusion look
- **Original aesthetics** — brushwork, imperfections, personal style

The percentages show **relative visual alignment** to those aesthetic poles. They always sum to ~100% but do **not** mean “this file was/wasn't made by AI.”

Optional env vars:

- `FORCE_CPU=0` — use NVIDIA GPU when available
- `SCORE_TEMPERATURE=1.5` — higher = softer, less extreme percentages (default 1.5)

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Server + model status |
| `POST /api/analyze` | Multipart field `file` (image) |

Example response:

```json
{
  "score_method": "stylistic_text_similarity",
  "originality_score": 62.3,
  "ai_likeness_percent": 37.7,
  "dominant_aesthetic": "original-aesthetic"
}
```

## Krita plugin

See the **`krita-plugin`** branch for a dock panel with live incremental feedback. Keep this server running while using Krita.
