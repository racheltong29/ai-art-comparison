# Krita plugin — incremental originality feedback

Dock panel that periodically checks your canvas against the local originality API.

## Prerequisites

Start the **webapp** server (separate branch):

```powershell
git checkout webapp
.\run.ps1
```

Then switch back to this branch to install the plugin, or keep both checked out in different folders.

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

See [krita-plugin/README.md](krita-plugin/README.md) for details.
