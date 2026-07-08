"""Composition analysis via classical CV saliency/contours and DL instance segmentation."""

from __future__ import annotations

import io
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
from PIL import Image

MAX_EDGE = 512
SALIENCY_THRESHOLD = 0.5
SEGMENTATION_SCORE_THRESHOLD = 0.5


@dataclass(frozen=True)
class CvCompositionResult:
    region_count: int
    largest_region_area_ratio: float
    subject_centroid_x: float
    subject_centroid_y: float
    rule_of_thirds_offset: float
    symmetry_score: float
    edge_density: float
    negative_space_ratio: float

    def to_dict(self) -> dict:
        return {
            "region_count": self.region_count,
            "largest_region_area_ratio": round(self.largest_region_area_ratio, 4),
            "subject_centroid_x": round(self.subject_centroid_x, 4),
            "subject_centroid_y": round(self.subject_centroid_y, 4),
            "rule_of_thirds_offset": round(self.rule_of_thirds_offset, 4),
            "symmetry_score": round(self.symmetry_score, 4),
            "edge_density": round(self.edge_density, 4),
            "negative_space_ratio": round(self.negative_space_ratio, 4),
        }


@dataclass(frozen=True)
class SegmentationCompositionResult:
    object_count: int
    subject_area_ratio: float
    bbox_spatial_spread: float
    class_diversity: int

    def to_dict(self) -> dict:
        return {
            "object_count": self.object_count,
            "subject_area_ratio": round(self.subject_area_ratio, 4),
            "bbox_spatial_spread": round(self.bbox_spatial_spread, 4),
            "class_diversity": self.class_diversity,
        }


@dataclass(frozen=True)
class CompositionResult:
    cv: CvCompositionResult
    segmentation: SegmentationCompositionResult | None = field(default=None)

    def to_dict(self) -> dict:
        result = {f"cv_{k}": v for k, v in self.cv.to_dict().items()}
        if self.segmentation is not None:
            result.update({f"seg_{k}": v for k, v in self.segmentation.to_dict().items()})
        return result


_THIRDS_POINTS = [(x, y) for x in (1 / 3, 2 / 3) for y in (1 / 3, 2 / 3)]


class CompositionAnalyzer:
    """Quantifies image composition via saliency-based bounding boxes and optional segmentation."""

    def __init__(self) -> None:
        self._segmentation_model = None
        self._segmentation_device: torch.device | None = None

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

    def analyze_cv(self, image_bytes: bytes) -> CvCompositionResult:
        image = self._prepare_image(image_bytes)
        rgb = np.array(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliency_map = saliency.computeSaliency(bgr)
        if not success:
            saliency_map = gray.astype("float32") / 255.0
        saliency_map = (saliency_map * 255).astype("uint8")

        _, thresholded = cv2.threshold(
            saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        total_area = float(width * height)
        region_count = len(contours)
        salient_area = 0.0
        largest_area = 0.0
        weighted_cx = 0.0
        weighted_cy = 0.0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = float(w * h)
            salient_area += area
            largest_area = max(largest_area, area)
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            weighted_cx += cx * area
            weighted_cy += cy * area

        if salient_area > 0:
            centroid_x = weighted_cx / salient_area
            centroid_y = weighted_cy / salient_area
        else:
            centroid_x, centroid_y = 0.5, 0.5

        rule_of_thirds_offset = min(
            ((centroid_x - tx) ** 2 + (centroid_y - ty) ** 2) ** 0.5
            for tx, ty in _THIRDS_POINTS
        )

        left_half = gray[:, : width // 2].astype("float32")
        right_half = cv2.flip(gray[:, width - width // 2 :], 1).astype("float32")
        min_w = min(left_half.shape[1], right_half.shape[1])
        horizontal_diff = float(
            np.mean(np.abs(left_half[:, :min_w] - right_half[:, :min_w]))
        )

        top_half = gray[: height // 2, :].astype("float32")
        bottom_half = cv2.flip(gray[height - height // 2 :, :], 0).astype("float32")
        min_h = min(top_half.shape[0], bottom_half.shape[0])
        vertical_diff = float(
            np.mean(np.abs(top_half[:min_h, :] - bottom_half[:min_h, :]))
        )

        symmetry_score = 1.0 - (horizontal_diff + vertical_diff) / (2 * 255.0)

        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.count_nonzero(edges)) / total_area

        negative_space_ratio = max(0.0, 1.0 - min(salient_area / total_area, 1.0))

        return CvCompositionResult(
            region_count=region_count,
            largest_region_area_ratio=largest_area / total_area,
            subject_centroid_x=centroid_x,
            subject_centroid_y=centroid_y,
            rule_of_thirds_offset=rule_of_thirds_offset,
            symmetry_score=symmetry_score,
            edge_density=edge_density,
            negative_space_ratio=negative_space_ratio,
        )

    def _get_segmentation_model(self):
        if self._segmentation_model is None:
            from torchvision.models.detection import (
                MaskRCNN_ResNet50_FPN_Weights,
                maskrcnn_resnet50_fpn,
            )

            force_cpu = os.getenv("FORCE_CPU", "1") != "0"
            self._segmentation_device = torch.device(
                "cuda" if (not force_cpu and torch.cuda.is_available()) else "cpu"
            )
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            model = maskrcnn_resnet50_fpn(weights=weights)
            model.to(self._segmentation_device)
            model.eval()
            self._segmentation_model = model
        return self._segmentation_model

    def analyze_segments(self, image_bytes: bytes) -> SegmentationCompositionResult:
        from torchvision.transforms.functional import to_tensor

        image = self._prepare_image(image_bytes)
        model = self._get_segmentation_model()
        tensor = to_tensor(image).to(self._segmentation_device)

        with torch.inference_mode():
            output = model([tensor])[0]

        scores = output["scores"].cpu().numpy()
        keep = scores >= SEGMENTATION_SCORE_THRESHOLD
        boxes = output["boxes"].cpu().numpy()[keep]
        masks = output["masks"].cpu().numpy()[keep]
        labels = output["labels"].cpu().numpy()[keep]

        width, height = image.size
        total_area = float(width * height)
        object_count = int(len(boxes))

        if object_count == 0:
            return SegmentationCompositionResult(
                object_count=0,
                subject_area_ratio=0.0,
                bbox_spatial_spread=0.0,
                class_diversity=0,
            )

        subject_mask = np.zeros((height, width), dtype=bool)
        for mask in masks:
            subject_mask |= mask[0] >= 0.5
        subject_area_ratio = float(np.count_nonzero(subject_mask)) / total_area

        centers_x = (boxes[:, 0] + boxes[:, 2]) / 2 / width
        centers_y = (boxes[:, 1] + boxes[:, 3]) / 2 / height
        bbox_spatial_spread = float(
            np.sqrt(np.var(centers_x) + np.var(centers_y))
        )

        class_diversity = int(len(set(labels.tolist())))

        return SegmentationCompositionResult(
            object_count=object_count,
            subject_area_ratio=subject_area_ratio,
            bbox_spatial_spread=bbox_spatial_spread,
            class_diversity=class_diversity,
        )

    def analyze(
        self, image_bytes: bytes, with_segmentation: bool = False
    ) -> CompositionResult:
        cv_result = self.analyze_cv(image_bytes)
        seg_result = self.analyze_segments(image_bytes) if with_segmentation else None
        return CompositionResult(cv=cv_result, segmentation=seg_result)
