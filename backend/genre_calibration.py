"""Per-genre logistic calibration of ai-likeness from composition features.

Batch analysis of AI-ArtBench (analysis/) found that composition metrics relate
to ai-likeness differently per art genre, and sometimes in opposite directions
(e.g. cv_edge_density correlates negatively with ai-likeness in baroque/romanticism
but positively in post-impressionism). A single pooled correlation is misleading
(Simpson's paradox), so instead of one global adjustment, this module holds one
small logistic regression per genre, fit on 9 cv_* composition features (including
the fuzzy/axis-searched cv_symmetry_score_fuzzy_global alongside the original
strict cv_symmetry_score) against the human/AI-generated label from that genre's
AI-ArtBench sample (n=200 each, 5-fold CV accuracy 0.75-0.92 — see
backend/data/genre_calibration.json for the per-genre accuracy and
analysis/output/<genre>/ for the underlying data).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent / "data" / "genre_calibration.json"

# Zero-shot genre prompts, matching AI-ArtBench's 10 style labels.
GENRE_PROMPTS = {
    "renaissance": "a renaissance style painting",
    "baroque": "a baroque style painting",
    "romanticism": "a romanticism style painting",
    "realism": "a realist painting",
    "impressionism": "an impressionist painting",
    "post_impressionism": "a post-impressionist painting",
    "expressionism": "an expressionist painting",
    "surrealism": "a surrealist painting",
    "art_nouveau": "an art nouveau style artwork",
    "ukiyo_e": "a ukiyo-e woodblock print style artwork",
}
GENRE_KEYS = tuple(GENRE_PROMPTS.keys())

# Below this, a *predicted* genre label is more often wrong than right (e.g.
# "post_impressionism" predictions are only 9% correct - it collapses into
# "impressionism" almost every time - see the zero-shot genre eval in the
# genre-calibration session). Below the bar, we skip calibration rather than
# route to a calibration model trained for the wrong genre.
MIN_GENRE_DETECTION_PRECISION = 0.6


@dataclass(frozen=True)
class GenreCalibrationResult:
    genre: str
    genre_calibrated_ai_likeness_percent: float
    calibration_model_cv_accuracy: float
    calibration_model_n_train: int
    top_features: list[tuple[str, float]]

    def to_dict(self) -> dict:
        return {
            "genre": self.genre,
            "genre_calibrated_ai_likeness_percent": self.genre_calibrated_ai_likeness_percent,
            "calibration_model_cv_accuracy": self.calibration_model_cv_accuracy,
            "calibration_model_n_train": self.calibration_model_n_train,
            "top_features": [
                {"feature": name, "contribution": round(value, 4)}
                for name, value in self.top_features
            ],
        }


class GenreCalibrator:
    """Applies the fitted per-genre logistic model to a CvCompositionResult."""

    def __init__(self, data_path: Path = DATA_PATH) -> None:
        with open(data_path) as f:
            payload = json.load(f)
        self.features: list[str] = payload["features"]
        self.genres: dict[str, dict] = payload["genres"]

    def is_reliable(self, genre: str) -> bool:
        model = self.genres.get(genre)
        if model is None:
            return False
        return model["genre_detection_precision"] >= MIN_GENRE_DETECTION_PRECISION

    def calibrate(self, genre: str, cv_dict: dict) -> GenreCalibrationResult | None:
        """`cv_dict` is CvCompositionResult.to_dict() (keys without the cv_ prefix).

        Returns None if `genre` isn't known or its zero-shot detection precision
        is too low to trust routing to that genre's calibration model.
        """
        model = self.genres.get(genre)
        if model is None or not self.is_reliable(genre):
            return None

        raw = [cv_dict[name.removeprefix("cv_")] for name in self.features]
        contributions = []
        z = model["intercept"]
        for value, mean, scale, coef, name in zip(
            raw, model["mean"], model["scale"], model["coef"], self.features
        ):
            standardized = (value - mean) / scale if scale else 0.0
            contribution = standardized * coef
            z += contribution
            contributions.append((name, contribution))

        probability = 1.0 / (1.0 + pow(2.718281828459045, -z))
        contributions.sort(key=lambda item: abs(item[1]), reverse=True)

        return GenreCalibrationResult(
            genre=genre,
            genre_calibrated_ai_likeness_percent=round(probability * 100, 1),
            calibration_model_cv_accuracy=model["cv_accuracy"],
            calibration_model_n_train=model["n_train"],
            top_features=contributions[:3],
        )
