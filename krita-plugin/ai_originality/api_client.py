"""POST image bytes to the local originality API (stdlib only — no pip deps in Krita)."""

from __future__ import annotations

import json
import uuid
import urllib.error
import urllib.request

DEFAULT_API_URL = "http://127.0.0.1:8000/api/analyze"


def analyze_png(png_bytes: bytes, api_url: str = DEFAULT_API_URL) -> dict:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="canvas.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + png_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    request = urllib.request.Request(
        api_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API error {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not reach the originality server. "
            "Start it from the webapp branch: .\\run.ps1"
        ) from exc
