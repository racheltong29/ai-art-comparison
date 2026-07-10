# AI art originality checker

Estimates how closely an artwork's stylistic aesthetic aligns with generic AI-generated
art vs. original hand-made work, and lets you study how that score relates to an
artwork's composition (subject placement, symmetry, region count, edge density, etc.).

## Running the web app (backend + frontend)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt # CPU-only torch; use requirements.txt if you have a GPU
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

(On Windows you can instead just run `.\run.ps1`, which does the same thing.)

Then open `http://127.0.0.1:8000/`:

- **Single image** (`/`) — drop one artwork, get its ai-likeness/originality score.
- **Batch view** (`/static/batch.html`) — drop up to 50 artworks at once and see a scatter
  plot of ai-likeness vs. a selectable composition metric, with a live Pearson correlation.
  Calls `POST /api/analyze-batch` (add `?with_segmentation=true` for the slower Mask R-CNN
  based metrics in addition to the fast saliency/contour ones).

The first request downloads the SigLIP model (~350 MB); segmentation additionally downloads
a Mask R-CNN checkpoint on first use.

## Running the offline dataset analysis

`analysis/` batch-scores a dataset and reports how ai-likeness correlates with composition,
using the [AI-ArtBench](https://ieee-dataport.org/documents/ai-artbench) dataset (human vs.
AI-generated artwork across matched styles) as the reference dataset:

```bash
# Point at a local copy of AI-ArtBench (Kaggle/IEEE DataPort download, unzipped)
python -m analysis.run_analysis --local-dir /path/to/AI-ArtBench --limit 200

# Or a HuggingFace Hub mirror
python -m analysis.run_analysis --hf-dataset some/mirror --limit 200 --with-segmentation

# Then correlate + plot
python -m analysis.correlate --input analysis/output/results.csv
```

`run_analysis.py` writes `analysis/output/results.csv` (one row per image: ai-likeness score,
composition metrics, style, and human/AI ground truth). `correlate.py` reads that CSV and
writes `correlations.csv` (now including Benjamini–Hochberg FDR-corrected p-values), per-metric
scatter plots, and a correlation heatmap into `analysis/output/`.

### Advanced statistics

Raw per-metric correlations are easy to over-read, so `analysis/stats.py` adds a more
defensible layer on top of the same `results.csv`:

```bash
python -m analysis.stats --input analysis/output/results.csv
```

It writes to `analysis/output/`:

| Output | Answers |
|--------|---------|
| `score_validation.json`, `score_roc.png` | Does `ai_likeness_percent` actually separate the human/AI ground truth? (AUROC, best-threshold accuracy, Mann–Whitney gap, plus each composition metric's own AUROC as a baseline.) A low AUROC means composition findings built on the score are weak. |
| `correlations_by_style.csv` | Per-style (metric × style) correlations — shows which effects survive within a single art style rather than being driven by style mix. |
| `partial_correlations.csv` | Correlation of each metric with ai-likeness **after removing the art-style confound** (partial correlation controlling for style), FDR-corrected. Compare `partial_r` to `raw_r`: large shrinkage means the raw correlation was mostly style. |
| `regression_summary.txt`, `feature_vif.csv` | Multiple regression of ai-likeness on all composition metrics at once (standardized coefficients + per-feature p-values + R²), and per-feature VIF flagging multicollinearity. |
| `feature_importances.csv`, `feature_importances.png` | Random-forest impurity and permutation importances — which metrics matter jointly, capturing non-linear effects the regression misses. |

Each section skips cleanly (with a printed note) when the data can't support it — e.g. only
one class present, or too few rows per style.

## Krita plugin — incremental originality feedback

Dock panel that periodically checks your canvas against the local originality API
(`http://127.0.0.1:8000/api/analyze`, so start the server above first).

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
