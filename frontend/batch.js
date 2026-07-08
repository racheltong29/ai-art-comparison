const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const results = document.getElementById("results");
const status = document.getElementById("status");
const errorPanel = document.getElementById("error");
const errorText = document.getElementById("error-text");
const metricSelect = document.getElementById("metric-select");
const correlationBadge = document.getElementById("correlation-badge");
const canvas = document.getElementById("scatter-canvas");
const ctx = canvas.getContext("2d");

const TARGET_FIELD = "ai_likeness_percent";
let currentRows = [];

function hideAll() {
  [dropZone, results, status, errorPanel].forEach((el) => el.classList.add("hidden"));
}

function reset() {
  hideAll();
  dropZone.classList.remove("hidden");
  fileInput.value = "";
  currentRows = [];
}

function showError(message) {
  hideAll();
  errorText.textContent = message;
  errorPanel.classList.remove("hidden");
}

function compositionMetrics(rows) {
  if (!rows.length) return [];
  return Object.keys(rows[0]).filter(
    (key) => (key.startsWith("cv_") || key.startsWith("seg_")) && typeof rows[0][key] === "number"
  );
}

function pearson(xs, ys) {
  const n = xs.length;
  if (n < 2) return NaN;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let cov = 0, varX = 0, varY = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - meanX;
    const dy = ys[i] - meanY;
    cov += dx * dy;
    varX += dx * dx;
    varY += dy * dy;
  }
  if (varX === 0 || varY === 0) return NaN;
  return cov / Math.sqrt(varX * varY);
}

function drawScatter(metric) {
  const points = currentRows
    .map((row) => ({ x: row[metric], y: row[TARGET_FIELD] }))
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const padding = 50;
  const plotW = canvas.width - padding * 2;
  const plotH = canvas.height - padding * 2;

  if (!points.length) return;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(0, Math.min(...ys)), yMax = Math.max(100, Math.max(...ys));
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  ctx.strokeStyle = "#2e3544";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, canvas.height - padding);
  ctx.lineTo(canvas.width - padding, canvas.height - padding);
  ctx.stroke();

  ctx.fillStyle = "#8b95a8";
  ctx.font = "12px sans-serif";
  ctx.fillText(metric, canvas.width / 2 - 30, canvas.height - 15);
  ctx.save();
  ctx.translate(15, canvas.height / 2 + 30);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(TARGET_FIELD, 0, 0);
  ctx.restore();

  ctx.fillStyle = "#6b9fff";
  for (const p of points) {
    const px = padding + ((p.x - xMin) / xRange) * plotW;
    const py = canvas.height - padding - ((p.y - yMin) / yRange) * plotH;
    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  const r = pearson(xs, ys);
  correlationBadge.textContent = Number.isFinite(r) ? `r = ${r.toFixed(3)}` : "r = n/a";
}

function populateMetricSelect(metrics) {
  metricSelect.innerHTML = "";
  for (const metric of metrics) {
    const option = document.createElement("option");
    option.value = metric;
    option.textContent = metric;
    metricSelect.appendChild(option);
  }
}

async function analyzeBatch(files) {
  hideAll();
  status.classList.remove("hidden");

  const formData = new FormData();
  for (const file of files) formData.append("files", file);

  try {
    const response = await fetch("/api/analyze-batch", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Batch analysis failed.");

    currentRows = payload;
    const metrics = compositionMetrics(currentRows);
    if (!metrics.length) throw new Error("No composition metrics returned.");

    populateMetricSelect(metrics);
    hideAll();
    results.classList.remove("hidden");
    drawScatter(metrics[0]);
  } catch (err) {
    showError(err.message || "Something went wrong.");
  }
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) analyzeBatch(fileInput.files);
});

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "var(--accent)";
});

dropZone.addEventListener("dragleave", () => {
  dropZone.style.borderColor = "";
});

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.style.borderColor = "";
  if (e.dataTransfer.files.length) analyzeBatch(e.dataTransfer.files);
});

metricSelect.addEventListener("change", () => drawScatter(metricSelect.value));
document.getElementById("reset-btn").addEventListener("click", reset);
document.getElementById("error-reset").addEventListener("click", reset);
