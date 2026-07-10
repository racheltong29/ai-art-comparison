"""Advanced statistics on results.csv: score validation, style control, joint modeling.

Reads the CSV produced by ``analysis.run_analysis`` and goes beyond the univariate
correlations in ``analysis.correlate``:

- ``validate_score``   - how well ``ai_likeness_percent`` separates the AI/human ground truth.
- ``style_controlled_correlations`` - per-style and style-partial correlations (removes the
  art-style confound).
- ``joint_model``      - multiple regression + random-forest importances over all metrics.

Usage:
    python -m analysis.stats --input analysis/output/results.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor

from analysis.correlate import TARGET_COLUMN, add_fdr, composition_columns

GROUND_TRUTH_COLUMN = "is_ai_generated"
STYLE_COLUMN = "style"
MIN_STYLE_ROWS = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default=str(Path(__file__).resolve().parent / "output" / "results.csv")
    )
    parser.add_argument(
        "--output-dir", default=str(Path(__file__).resolve().parent / "output")
    )
    return parser.parse_args()


def _ground_truth_as_int(df: pd.DataFrame) -> np.ndarray | None:
    """Coerces the is_ai_generated column to 0/1, or None if it isn't usable."""
    if GROUND_TRUTH_COLUMN not in df.columns:
        return None
    series = df[GROUND_TRUTH_COLUMN]
    if series.dtype == bool:
        labels = series.astype(int)
    else:
        mapping = {"true": 1, "1": 1, "ai": 1, "false": 0, "0": 0, "human": 0}
        labels = series.astype(str).str.strip().str.lower().map(mapping)
    labels = labels.dropna()
    if labels.nunique() < 2:
        return None
    return labels.astype(int).to_numpy()


# --------------------------------------------------------------------------- #
# Idea 1: validate the ai_likeness score against ground truth
# --------------------------------------------------------------------------- #
def validate_score(df: pd.DataFrame, output_dir: Path) -> dict | None:
    labels = _ground_truth_as_int(df)
    if labels is None:
        print("[validate_score] skipped: need both AI and human rows in is_ai_generated.")
        return None

    mask = df[GROUND_TRUTH_COLUMN].notna()
    scores = df.loc[mask, TARGET_COLUMN].to_numpy(dtype=float)

    auroc = float(roc_auc_score(labels, scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    youden = tpr - fpr
    best_idx = int(np.argmax(youden))
    best_threshold = float(thresholds[best_idx])
    predictions = (scores >= best_threshold).astype(int)
    best_accuracy = float((predictions == labels).mean())

    ai_scores = scores[labels == 1]
    human_scores = scores[labels == 0]
    mwu_stat, mwu_p = stats.mannwhitneyu(ai_scores, human_scores, alternative="two-sided")

    # Baseline: how well does each raw composition metric separate AI/human on its own?
    metric_auroc = {}
    for column in composition_columns(df):
        values = df.loc[mask, column].to_numpy(dtype=float)
        if np.all(np.isnan(values)) or np.nanstd(values) == 0:
            continue
        try:
            # AUROC is direction-agnostic here: report max(auc, 1-auc) so a metric that
            # separates in either direction is credited.
            auc = roc_auc_score(labels, np.nan_to_num(values, nan=np.nanmean(values)))
        except ValueError:
            continue
        metric_auroc[column] = round(max(auc, 1 - auc), 4)

    if auroc >= 0.75:
        verdict = "score separates AI/human well above chance"
    elif auroc >= 0.6:
        verdict = "score separates AI/human modestly above chance"
    else:
        verdict = "score barely beats chance - composition findings on it are weak"

    summary = {
        "n": int(mask.sum()),
        "n_ai": int((labels == 1).sum()),
        "n_human": int((labels == 0).sum()),
        "score_auroc": round(auroc, 4),
        "best_threshold": round(best_threshold, 4),
        "best_accuracy": round(best_accuracy, 4),
        "ai_mean_score": round(float(ai_scores.mean()), 4),
        "human_mean_score": round(float(human_scores.mean()), 4),
        "mannwhitney_p": round(float(mwu_p), 6),
        "composition_metric_auroc": dict(
            sorted(metric_auroc.items(), key=lambda kv: kv[1], reverse=True)
        ),
        "verdict": verdict,
    }

    with open(output_dir / "score_validation.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ai_likeness (AUROC={auroc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#8b95a8", label="chance")
    ax.scatter([fpr[best_idx]], [tpr[best_idx]], color="#f07178", zorder=5, label="best threshold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("AI-likeness score vs. ground truth")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "score_roc.png", dpi=150)
    plt.close(fig)

    print(f"[validate_score] AUROC {auroc:.3f}, acc {best_accuracy:.3f} - {verdict}")
    return summary


# --------------------------------------------------------------------------- #
# Idea 2: control for the art-style confound
# --------------------------------------------------------------------------- #
def style_controlled_correlations(df: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = composition_columns(df)
    if not metric_columns:
        print("[style_controlled] skipped: no composition metric columns.")
        return
    if STYLE_COLUMN not in df.columns or df[STYLE_COLUMN].nunique() < 2:
        print("[style_controlled] skipped: need >=2 distinct styles.")
        return

    # (a) Per-style stratified Pearson r (metric x style matrix).
    per_style_rows = []
    for style, group in df.groupby(STYLE_COLUMN):
        if len(group) < MIN_STYLE_ROWS:
            continue
        row = {"style": style, "n": len(group)}
        for column in metric_columns:
            valid = group[[TARGET_COLUMN, column]].dropna()
            if len(valid) < 3 or valid[column].nunique() < 2:
                row[column] = np.nan
            else:
                row[column] = round(
                    float(stats.pearsonr(valid[TARGET_COLUMN], valid[column])[0]), 4
                )
        per_style_rows.append(row)

    if per_style_rows:
        pd.DataFrame(per_style_rows).to_csv(
            output_dir / "correlations_by_style.csv", index=False
        )
        print(f"[style_controlled] wrote correlations_by_style.csv ({len(per_style_rows)} styles)")
    else:
        print(f"[style_controlled] no style had >={MIN_STYLE_ROWS} rows; skipped stratified table.")

    # (b) Style-partial correlation: residualize target and metric on style dummies.
    dummies = pd.get_dummies(df[STYLE_COLUMN], drop_first=True).astype(float)
    partial_rows = []
    for column in metric_columns:
        valid = df[[TARGET_COLUMN, column]].join(dummies).dropna()
        if len(valid) < len(dummies.columns) + 3 or valid[column].nunique() < 2:
            continue
        x_dummies = valid[dummies.columns].to_numpy()
        target = valid[TARGET_COLUMN].to_numpy()
        metric = valid[column].to_numpy()

        raw_r = float(stats.pearsonr(target, metric)[0])
        target_resid = target - LinearRegression().fit(x_dummies, target).predict(x_dummies)
        metric_resid = metric - LinearRegression().fit(x_dummies, metric).predict(x_dummies)
        if np.std(target_resid) == 0 or np.std(metric_resid) == 0:
            continue
        partial_r, partial_p = stats.pearsonr(target_resid, metric_resid)
        partial_rows.append(
            {
                "metric": column,
                "raw_r": round(raw_r, 4),
                "partial_r": round(float(partial_r), 4),
                "partial_p": round(float(partial_p), 4),
                "n": len(valid),
            }
        )

    if partial_rows:
        partial_df = add_fdr(pd.DataFrame(partial_rows), "partial_p")
        partial_df = partial_df.sort_values(
            "partial_r", key=lambda s: s.abs(), ascending=False
        )
        partial_df.to_csv(output_dir / "partial_correlations.csv", index=False)
        print("[style_controlled] wrote partial_correlations.csv (style-controlled)")
        print(partial_df.to_string(index=False))
    else:
        print("[style_controlled] not enough data for partial correlations.")


# --------------------------------------------------------------------------- #
# Idea 3: joint model over all composition metrics
# --------------------------------------------------------------------------- #
def joint_model(df: pd.DataFrame, output_dir: Path) -> None:
    metric_columns = composition_columns(df)
    if len(metric_columns) < 2:
        print("[joint_model] skipped: need >=2 composition metrics.")
        return

    data = df[[TARGET_COLUMN, *metric_columns]].dropna()
    # Drop zero-variance metrics (constant columns break standardization / VIF).
    usable = [c for c in metric_columns if data[c].nunique() > 1]
    if len(usable) < 2 or len(data) < len(usable) + 2:
        print("[joint_model] skipped: not enough rows/variance for a joint model.")
        return

    y = data[TARGET_COLUMN].to_numpy(dtype=float)
    x_raw = data[usable].to_numpy(dtype=float)
    x_std = (x_raw - x_raw.mean(axis=0)) / x_raw.std(axis=0)

    # OLS with standardized predictors -> comparable coefficients + per-feature p-values.
    x_design = sm.add_constant(x_std)
    ols = sm.OLS(y, x_design).fit()
    summary_text = ols.summary(xname=["const", *usable], yname=TARGET_COLUMN).as_text()
    with open(output_dir / "regression_summary.txt", "w") as fh:
        fh.write(summary_text)

    # VIF flags multicollinearity among the (standardized) predictors.
    vif_rows = []
    for i, column in enumerate(usable):
        vif_rows.append(
            {"metric": column, "vif": round(float(variance_inflation_factor(x_std, i)), 3)}
        )
    pd.DataFrame(vif_rows).sort_values("vif", ascending=False).to_csv(
        output_dir / "feature_vif.csv", index=False
    )

    # Random forest + permutation importance (more reliable than impurity importance).
    forest = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
    forest.fit(x_raw, y)
    perm = permutation_importance(forest, x_raw, y, n_repeats=20, random_state=0, n_jobs=-1)
    importance_df = pd.DataFrame(
        {
            "metric": usable,
            "impurity_importance": np.round(forest.feature_importances_, 4),
            "permutation_importance": np.round(perm.importances_mean, 4),
        }
    ).sort_values("permutation_importance", ascending=False)
    importance_df.to_csv(output_dir / "feature_importances.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 0.4 * len(usable) + 1.5))
    ordered = importance_df.iloc[::-1]
    ax.barh(ordered["metric"], ordered["permutation_importance"], color="#6b9fff")
    ax.set_xlabel("Permutation importance")
    ax.set_title(f"Composition drivers of {TARGET_COLUMN}")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importances.png", dpi=150)
    plt.close(fig)

    top = importance_df.head(3)["metric"].tolist()
    print(
        f"[joint_model] OLS R2={ols.rsquared:.3f} (adj {ols.rsquared_adj:.3f}), "
        f"RF R2={forest.score(x_raw, y):.3f}; top drivers: {', '.join(top)}"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)

    validate_score(df, output_dir)
    style_controlled_correlations(df, output_dir)
    joint_model(df, output_dir)
    print(f"\nWrote statistics outputs to {output_dir}")


if __name__ == "__main__":
    main()
