# Krita plugin — incremental originality feedback

Dock panel that periodically checks your canvas against the local originality API.

**Requires the webapp server** from the `webapp` branch to be running (`.\run.ps1`).

## Install

1. Copy the `ai_originality` folder into your Krita pykrita directory:

   **Windows:** `%APPDATA%\krita\pykrita\ai_originality`

   **Linux:** `~/.local/share/krita/pykrita/ai_originality`

2. Restart Krita.

3. Enable the dock: **Settings → Dockers → Originality Check** (or find it in the right docker panel).

4. Start the backend from the `webapp` branch:

   ```powershell
   .\run.ps1
   ```

## Usage

- **Check now** — one-shot analysis of the flattened active layer.
- **Live feedback** — re-check every 15–300 seconds (default 45s).
- **Trend** — compares the last two scores so you can see if changes helped.

Canvas is exported as a downscaled PNG (max 512px) to keep CPU inference fast.

## Notes

- Uses only Python stdlib for HTTP — no extra pip packages inside Krita.
- Scores are estimates from a general AI-vs-human classifier, not proof of how you made the art.
- For unfinished sketches, expect noisy results.
