"""Stylistic AI-aesthetic scoring via SigLIP text–image similarity (not binary classification)."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

from backend.genre_calibration import GENRE_KEYS, GENRE_PROMPTS

MODEL_ID = "google/siglip-base-patch16-224"
MAX_EDGE = 512

# Text anchors describe *visual aesthetics*, not "was this file made by AI".
AI_AESTHETIC_PROMPTS = (
    "generic AI generated digital art with smooth polished rendering",
    "synthetic illustration typical of Midjourney or Stable Diffusion",
    "stock AI artwork with perfect lighting and generic composition",
)

ORIGINAL_AESTHETIC_PROMPTS = (
    "original hand-painted digital art with visible personal brushwork",
    "rough sketchy illustration with unique human imperfections",
    "authentic human-created artwork with distinctive stylistic choices",
)


@dataclass(frozen=True)
class AnalysisResult:
    ai_aesthetic_similarity: float
    original_aesthetic_similarity: float
    originality_score: float
    ai_likeness_percent: float
    dominant_aesthetic: str
    score_method: str
    model_id: str
    device: str
    detected_genre: str
    genre_confidence: float

    # Backward-compatible aliases for the Krita plugin and older clients.
    @property
    def ai_probability(self) -> float:
        return self.ai_aesthetic_similarity

    @property
    def human_probability(self) -> float:
        return self.original_aesthetic_similarity

    @property
    def predicted_label(self) -> str:
        return self.dominant_aesthetic


class ArtLikenessDetector:
    """Compares image embeddings to aesthetic text anchors — similarity, not classification."""

    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        force_cpu = os.getenv("FORCE_CPU", "1") != "0"
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self.temperature = float(os.getenv("SCORE_TEMPERATURE", "1.5"))

        torch.set_num_threads(max(1, os.cpu_count() or 1))

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = SiglipModel.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

        self._aesthetic_count = len(AI_AESTHETIC_PROMPTS) + len(ORIGINAL_AESTHETIC_PROMPTS)
        self._ai_count = len(AI_AESTHETIC_PROMPTS)
        self._genre_prompts = [GENRE_PROMPTS[key] for key in GENRE_KEYS]
        # Genre prompts ride along in the same forward pass (no extra vision-encoder cost).
        self._prompts = [*AI_AESTHETIC_PROMPTS, *ORIGINAL_AESTHETIC_PROMPTS, *self._genre_prompts]

    @staticmethod
    def _prepare_image(image_bytes: bytes) -> Image.Image:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        longest = max(width, height)
        if longest > MAX_EDGE:
            scale = MAX_EDGE / longest
            image = image.resize(
                (int(width * scale), int(height * scale)),
                Image.Resampling.LANCZOS,
            )
        return image

    def _similarity_weights(self, logits: torch.Tensor) -> tuple[float, float]:
        ai_mean = logits[: self._ai_count].mean()
        original_mean = logits[self._ai_count :].mean()
        weights = torch.softmax(
            torch.stack([ai_mean, original_mean]) / self.temperature,
            dim=0,
        )
        return float(weights[0].item()), float(weights[1].item())

    def _detect_genre(self, genre_logits: torch.Tensor) -> tuple[str, float]:
        probs = torch.softmax(genre_logits / self.temperature, dim=0)
        best = int(torch.argmax(probs).item())
        return GENRE_KEYS[best], float(probs[best].item())

    def analyze(self, image_bytes: bytes) -> AnalysisResult:
        image = self._prepare_image(image_bytes)
        inputs = self.processor(
            text=self._prompts,
            images=image,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits_per_image.squeeze(0)

        aesthetic_logits = logits[: self._aesthetic_count]
        genre_logits = logits[self._aesthetic_count :]

        ai_sim, original_sim = self._similarity_weights(aesthetic_logits)
        dominant = "original-aesthetic" if original_sim >= ai_sim else "ai-aesthetic"
        genre, genre_confidence = self._detect_genre(genre_logits)

        return AnalysisResult(
            ai_aesthetic_similarity=round(ai_sim, 4),
            original_aesthetic_similarity=round(original_sim, 4),
            originality_score=round(original_sim * 100, 1),
            ai_likeness_percent=round(ai_sim * 100, 1),
            dominant_aesthetic=dominant,
            score_method="stylistic_text_similarity",
            model_id=self.model_id,
            device=str(self.device),
            detected_genre=genre,
            genre_confidence=round(genre_confidence, 4),
        )
