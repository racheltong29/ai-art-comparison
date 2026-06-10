$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
