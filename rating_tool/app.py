"""Standalone local tool for blind-rating artworks 0-100 on perceived ai-likeness.

Images are pre-shuffled/anonymized (item_000.jpg, item_001.jpg, ...) with the
true source held separately in a hidden answer key, so raters never see it.
Multiple raters can use this from the same machine, one after another - each
enters a name once, and ratings are appended to ratings.csv with rater/item/
score/reason/timestamp so per-rater agreement can be analyzed afterward.

Run:
    uvicorn rating_tool.app:app --host 127.0.0.1 --port 8010
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DATA_DIR = Path("/data/rachelto/rating_set")
IMAGES_DIR = DATA_DIR / "images"
RATINGS_CSV = DATA_DIR / "ratings.csv"

app = FastAPI()
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


class Rating(BaseModel):
    rater: str
    item_id: str
    score: float
    reason: str


@app.get("/api/items")
def list_items() -> list[str]:
    return sorted(p.name for p in IMAGES_DIR.glob("*.jpg"))


@app.post("/api/rate")
def submit_rating(rating: Rating) -> dict:
    if not (0 <= rating.score <= 100):
        raise HTTPException(status_code=400, detail="score must be 0-100")
    if not rating.rater.strip():
        raise HTTPException(status_code=400, detail="rater name required")

    is_new = not RATINGS_CSV.exists()
    with open(RATINGS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["timestamp", "rater", "item_id", "score", "reason"])
        writer.writerow(
            [datetime.datetime.now().isoformat(timespec="seconds"), rating.rater, rating.item_id, rating.score, rating.reason]
        )
    return {"status": "ok"}


@app.get("/api/progress")
def progress(rater: str) -> JSONResponse:
    rated = set()
    if RATINGS_CSV.exists():
        with open(RATINGS_CSV) as f:
            for row in csv.DictReader(f):
                if row["rater"] == rater:
                    rated.add(row["item_id"])
    return JSONResponse({"rated_item_ids": sorted(rated)})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _PAGE


_PAGE = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AI-likeness rating</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
  #rater-setup { margin-bottom: 1.5rem; }
  #rater-setup input { font-size: 1rem; padding: 0.4rem; }
  img { max-width: 100%; max-height: 60vh; display: block; margin: 1rem auto; border: 1px solid #ccc; }
  .row { margin: 0.8rem 0; }
  input[type=range] { width: 100%; }
  textarea { width: 100%; min-height: 4rem; font-size: 1rem; box-sizing: border-box; }
  button { font-size: 1rem; padding: 0.5rem 1.2rem; cursor: pointer; }
  #progress { color: #666; font-size: 0.9rem; }
  #score-display { font-weight: bold; font-size: 1.2rem; }
  #done { display: none; text-align: center; padding: 3rem 0; }
</style>
</head>
<body>

<div id="rater-setup">
  <label>Your name: <input id="rater-name" placeholder="e.g. alex"></label>
  <button onclick="startRating()">Start / Resume</button>
</div>

<div id="rating-ui" style="display:none">
  <div id="progress"></div>
  <img id="artwork" src="">
  <div class="row">
    How AI-generated does this look? <span id="score-display">50</span>/100
    <input type="range" id="score" min="0" max="100" value="50" oninput="document.getElementById('score-display').innerText = this.value">
  </div>
  <div class="row">
    <textarea id="reason" placeholder="Why? (e.g. hands look wrong, too smooth/airbrushed, lighting is odd, looks like typical human brushwork, etc.)"></textarea>
  </div>
  <button onclick="submitRating()">Submit &amp; Next</button>
</div>

<div id="done">
  <h2>All done - thank you!</h2>
</div>

<script>
let items = [];
let ratedIds = new Set();
let currentIndex = 0;
let rater = "";

async function startRating() {
  rater = document.getElementById("rater-name").value.trim();
  if (!rater) { alert("Enter your name first."); return; }
  localStorage.setItem("rater_name", rater);

  const [itemsResp, progressResp] = await Promise.all([
    fetch("/api/items").then(r => r.json()),
    fetch("/api/progress?rater=" + encodeURIComponent(rater)).then(r => r.json()),
  ]);
  items = itemsResp;
  ratedIds = new Set(progressResp.rated_item_ids);

  document.getElementById("rater-setup").style.display = "none";
  document.getElementById("rating-ui").style.display = "block";
  showNextUnrated();
}

function showNextUnrated() {
  while (currentIndex < items.length && ratedIds.has(items[currentIndex])) {
    currentIndex++;
  }
  if (currentIndex >= items.length) {
    document.getElementById("rating-ui").style.display = "none";
    document.getElementById("done").style.display = "block";
    return;
  }
  const itemId = items[currentIndex];
  document.getElementById("artwork").src = "/images/" + itemId;
  document.getElementById("progress").innerText = `Image ${currentIndex + 1} of ${items.length} (${ratedIds.size} rated so far)`;
  document.getElementById("score").value = 50;
  document.getElementById("score-display").innerText = "50";
  document.getElementById("reason").value = "";
}

async function submitRating() {
  const itemId = items[currentIndex];
  const score = parseFloat(document.getElementById("score").value);
  const reason = document.getElementById("reason").value;
  await fetch("/api/rate", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({rater, item_id: itemId, score, reason}),
  });
  ratedIds.add(itemId);
  currentIndex++;
  showNextUnrated();
}

window.onload = () => {
  const saved = localStorage.getItem("rater_name");
  if (saved) document.getElementById("rater-name").value = saved;
};
</script>

</body>
</html>
"""
