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

First analysis downloads the model (~350 MB from Hugging Face).

## API

| Endpoint | Description |
|----------|-------------|
| `GET /api/health` | Server + model status |
| `POST /api/analyze` | Multipart field `file` (image) |

## Model

Free classifier: [`Ateeqq/ai-vs-human-image-detector`](https://huggingface.co/Ateeqq/ai-vs-human-image-detector) (~93M params).

Set `FORCE_CPU=0` to use an NVIDIA GPU when available.

## Krita plugin

See the **`krita-plugin`** branch for a dock panel with live incremental feedback. Keep this server running while using Krita.
