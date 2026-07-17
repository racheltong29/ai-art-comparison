"""Loads AI-ArtBench artwork samples for the batch composition/ai-likeness analysis.

AI-ArtBench (Silva et al.) pairs human-painted artwork (from ArtBench-10 /
WikiArt) with AI-generated images (Latent Diffusion + Standard Diffusion)
across 10 matched art styles. It's distributed via Kaggle/IEEE DataPort as a
folder tree, and mirrored piecemeal on the HF Hub under various repo ids, so
this loader supports both:

- A local directory (the common case: download from Kaggle, unzip, point
  here) laid out as ``<root>/<split>/<style_dir>/*.jpg``, where
  ``style_dir`` is prefixed ``AI_SD_``/``AI_LD_`` for AI-generated images and
  unprefixed for human-painted images (the AI-ArtBench convention).
- A HuggingFace Hub dataset id, for whichever mirror you point at, as long
  as it exposes an image column plus a boolean/string field indicating
  AI-generated vs. human and a style/label field. Column names are
  configurable since mirrors vary.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

AI_STYLE_PREFIXES = ("AI_SD_", "AI_LD_")


@dataclass(frozen=True)
class ArtworkSample:
    image_bytes: bytes
    style: str
    is_ai_generated: bool
    source: str
    path: str | None = None


def iter_local_directory(
    root: str | Path, split: str | None = None, style_filter: str | None = None
) -> Iterator[ArtworkSample]:
    """Walks a local AI-ArtBench directory tree, yielding one sample per image file.

    `style_filter`, if given, restricts to style directories whose name (after
    stripping the AI_SD_/AI_LD_ prefix) matches, e.g. "renaissance" picks up
    `renaissance/`, `AI_SD_renaissance/`, and `AI_LD_renaissance/`.
    """
    root_path = Path(root)
    base = root_path / split if split else root_path
    if not base.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {base}")

    for style_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        is_ai = style_dir.name.startswith(AI_STYLE_PREFIXES)
        style = style_dir.name
        for prefix in AI_STYLE_PREFIXES:
            if style.startswith(prefix):
                style = style[len(prefix):]
                break
        if style_filter and style.lower().replace("-", "_") != style_filter.lower().replace("-", "_"):
            continue
        for image_path in sorted(style_dir.iterdir()):
            if image_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            yield ArtworkSample(
                image_bytes=image_path.read_bytes(),
                style=style,
                is_ai_generated=is_ai,
                source=style_dir.name,
                path=str(image_path),
            )


def iter_hf_dataset(
    repo_id: str,
    split: str = "train",
    image_column: str = "image",
    label_column: str = "label",
    style_column: str | None = None,
    ai_label_values: tuple[str, ...] = ("ai", "ai-generated", "generated", "1", "True"),
) -> Iterator[ArtworkSample]:
    """Streams a HuggingFace Hub dataset, yielding one sample per row."""
    from datasets import load_dataset

    ds = load_dataset(repo_id, split=split, streaming=True)
    for row in ds:
        image = row[image_column]
        buffer_bytes = _pil_to_bytes(image)
        label_value = str(row[label_column])
        is_ai = label_value.strip().lower() in {v.lower() for v in ai_label_values}
        style = str(row[style_column]) if style_column else "unknown"
        yield ArtworkSample(
            image_bytes=buffer_bytes,
            style=style,
            is_ai_generated=is_ai,
            source=repo_id,
        )


def _pil_to_bytes(image) -> bytes:
    import io

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG")
    return buffer.getvalue()


def sample_balanced(
    samples: Iterator[ArtworkSample], limit: int, seed: int = 0
) -> list[ArtworkSample]:
    """Reservoir-samples up to `limit` items, capping AI/human at half each when possible."""
    rng = random.Random(seed)
    ai_bucket: list[ArtworkSample] = []
    human_bucket: list[ArtworkSample] = []
    per_class_cap = max(1, limit // 2)

    for sample in samples:
        bucket = ai_bucket if sample.is_ai_generated else human_bucket
        if len(bucket) < per_class_cap:
            bucket.append(sample)
        elif rng.random() < per_class_cap / (len(bucket) + 1):
            bucket[rng.randrange(per_class_cap)] = sample
        if len(ai_bucket) >= per_class_cap and len(human_bucket) >= per_class_cap:
            break

    combined = ai_bucket + human_bucket
    rng.shuffle(combined)
    return combined[:limit]
