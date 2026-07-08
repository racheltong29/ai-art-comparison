"""Correlates ai_likeness_percent against composition metrics from results.csv.

Usage:
    python -m analysis.correlate --input analysis/output/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
TARGET_COLUMN = "ai_likeness_percent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(OUTPUT_DIR / "results.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def composition_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("cv_") or c.startswith("seg_")]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    metric_columns = composition_columns(df)
    if not metric_columns:
        raise SystemExit(f"No composition metric columns found in {args.input}")

    correlation_rows = []
    for column in metric_columns:
        valid = df[[TARGET_COLUMN, column]].dropna()
        if len(valid) < 3:
            continue
        pearson_r, pearson_p = stats.pearsonr(valid[TARGET_COLUMN], valid[column])
        spearman_r, spearman_p = stats.spearmanr(valid[TARGET_COLUMN], valid[column])
        correlation_rows.append(
            {
                "metric": column,
                "pearson_r": round(pearson_r, 4),
                "pearson_p": round(pearson_p, 4),
                "spearman_r": round(spearman_r, 4),
                "spearman_p": round(spearman_p, 4),
                "n": len(valid),
            }
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        if "is_ai_generated" in df.columns:
            for is_ai, group in df.groupby("is_ai_generated"):
                ax.scatter(
                    group[column],
                    group[TARGET_COLUMN],
                    label="AI-generated" if is_ai else "Human",
                    alpha=0.6,
                    s=20,
                )
            ax.legend()
        else:
            ax.scatter(df[column], df[TARGET_COLUMN], alpha=0.6, s=20)
        ax.set_xlabel(column)
        ax.set_ylabel(TARGET_COLUMN)
        ax.set_title(f"{TARGET_COLUMN} vs. {column} (r={pearson_r:.2f})")
        fig.tight_layout()
        fig.savefig(output_dir / f"scatter_{column}.png", dpi=150)
        plt.close(fig)

    correlation_df = pd.DataFrame(correlation_rows).sort_values(
        "pearson_r", key=lambda s: s.abs(), ascending=False
    )
    correlation_df.to_csv(output_dir / "correlations.csv", index=False)
    print(correlation_df.to_string(index=False))

    heatmap_columns = [TARGET_COLUMN, *metric_columns]
    corr_matrix = df[heatmap_columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(heatmap_columns), 1 + 0.6 * len(heatmap_columns)))
    im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(heatmap_columns)))
    ax.set_xticklabels(heatmap_columns, rotation=90)
    ax.set_yticks(range(len(heatmap_columns)))
    ax.set_yticklabels(heatmap_columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote correlations.csv, scatter plots, and correlation_heatmap.png to {output_dir}")


if __name__ == "__main__":
    main()
