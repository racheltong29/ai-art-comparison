# Krita plugin — incremental originality feedback

Dock panel that periodically checks your canvas against the local originality API.

## Prerequisites

The plugin calls `http://127.0.0.1:8000/api/analyze`. Start the server from the **`webapp`** branch:

```powershell
git checkout webapp
.\run.ps1
```

## Install

Copy `krita-plugin/ai_originality` to your Krita pykrita folder:

| OS | Path |
|----|------|
| Windows | `%APPDATA%\krita\pykrita\ai_originality` |
| Linux | `~/.local/share/krita/pykrita/ai_originality` |

Restart Krita → **Settings → Dockers → Originality Check**.

## Usage

- **Check now** — analyze the active layer once
- **Live feedback** — auto re-check every 15–300 s (default 45 s)
- **Trend** — shows if your last edit moved the score up or down

Details: [krita-plugin/README.md](krita-plugin/README.md)
