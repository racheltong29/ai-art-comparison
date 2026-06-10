"""CPU-friendly AI-likeness scoring using a free Hugging Face classifier."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"
MAX_EDGE = 512


@dataclass(frozen=True)
class AnalysisResult:
    ai_probability: float
    human_probability: float
    predicted_label: str
    originality_score: float
    model_id: str
    device: str


class ArtLikenessDetector:
    def __init__(self, model_id: str = MODEL_ID) -> None:
        self.model_id = model_id
        force_cpu = os.getenv("FORCE_CPU", "1") != "0"
        if force_cpu or not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        torch.set_num_threads(max(1, os.cpu_count() or 1))

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = SiglipForImageClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()

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

    def analyze(self, image_bytes: bytes) -> AnalysisResult:
        image = self._prepare_image(image_bytes)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        labels = self.model.config.id2label
        probs_by_label = {
            labels[i].lower(): float(probabilities[i].item())
            for i in range(len(labels))
        }

        ai_prob = probs_by_label.get("ai", 0.0)
        human_prob = probs_by_label.get("hum", probs_by_label.get("human", 0.0))
        predicted_idx = int(torch.argmax(probabilities).item())
        predicted_label = labels[predicted_idx].lower()

        return AnalysisResult(
            ai_probability=round(ai_prob, 4),
            human_probability=round(human_prob, 4),
            predicted_label=predicted_label,
            originality_score=round(human_prob * 100, 1),
            model_id=self.model_id,
            device=str(self.device),
        )
