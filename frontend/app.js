const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const results = document.getElementById("results");
const status = document.getElementById("status");
const errorPanel = document.getElementById("error");
const preview = document.getElementById("preview");
const originalityScore = document.getElementById("originality-score");
const aiPercent = document.getElementById("ai-percent");
const feedback = document.getElementById("feedback");
const statusText = document.getElementById("status-text");
const errorText = document.getElementById("error-text");

function hideAll() {
  [dropZone, results, status, errorPanel].forEach((el) => el.classList.add("hidden"));
}

function reset() {
  hideAll();
  dropZone.classList.remove("hidden");
  fileInput.value = "";
}

function showError(message) {
  hideAll();
  errorText.textContent = message;
  errorPanel.classList.remove("hidden");
}

function tip(originality) {
  if (originality >= 70) return "Reads as fairly human-like to the model.";
  if (originality >= 45) return "Mixed — some AI-like visual patterns detected.";
  return "High AI-likeness — consider pushing texture, composition, or personal style.";
}

async function analyze(file) {
  if (!file?.type.startsWith("image/")) {
    showError("Please choose an image file.");
    return;
  }

  hideAll();
  status.classList.remove("hidden");
  statusText.textContent = "Analyzing…";

  const url = URL.createObjectURL(file);
  preview.src = url;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "Analysis failed.");

    originalityScore.textContent = `${payload.originality_score}%`;
    aiPercent.textContent = `${payload.ai_likeness_percent}%`;
    feedback.textContent = tip(payload.originality_score);

    hideAll();
    results.classList.remove("hidden");
  } catch (err) {
    showError(err.message || "Something went wrong.");
  } finally {
    URL.revokeObjectURL(url);
  }
}

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) analyze(file);
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
  const [file] = e.dataTransfer.files;
  if (file) analyze(file);
});

document.getElementById("reset-btn").addEventListener("click", reset);
document.getElementById("error-reset").addEventListener("click", reset);
