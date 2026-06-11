# Krita plugin — incremental originality feedback

Dock panel that periodically checks your canvas against the local originality API.

**Requires the webapp server** from the `webapp` branch to be running (`.\run.ps1`).

## Install (Windows)

1. Copy **both** of these into `%APPDATA%\krita\pykrita\`:
   - folder `ai_originality\` (the Python code)
   - file `ai_originality.desktop` (tells Krita the plugin exists)

   ```powershell
   Copy-Item -Recurse -Force "krita-plugin\ai_originality" "$env:APPDATA\krita\pykrita\ai_originality"
   Copy-Item -Force "krita-plugin\ai_originality.desktop" "$env:APPDATA\krita\pykrita\ai_originality.desktop"
   ```

2. **Restart Krita.**

3. **Settings → Configure Krita → Python Plugin Manager** → enable **Originality Check** → restart Krita again.

4. **Settings → Dockers → Originality Check** → tick it on.

5. Start the backend (on the `webapp` branch):

   ```powershell
   .\run.ps1
   ```

## Usage

- **Check now** — one-shot analysis of the active layer
- **Live feedback** — re-check every 15–300 seconds (default 45s)
- **Trend** — compares the last two scores

## Troubleshooting

- **Not in Docker list?** Missing `ai_originality.desktop` or plugin not enabled in Python Plugin Manager.
- **Hover over a grayed-out plugin** in Plugin Manager to see load errors.
- **"Could not reach server"** — run `.\run.ps1` on the webapp branch first.
