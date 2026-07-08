"""Batch-scores AI-ArtBench artwork samples for ai-likeness + composition metrics.

Usage:
    python -m analysis.run_analysis --local-dir /path/to/AI-ArtBench --limit 200
    python -m analysis.run_analysis --hf-dataset some/mirror --limit 200 --with-segmentation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from analysis.dataset import (
    ArtworkSample,
    iter_hf_dataset,
    iter_local_directory,
    sample_balanced,
)
from backend.composition import CompositionAnalyzer
from backend.detector import ArtLikenessDetector

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--local-dir", help="Path to a locally downloaded AI-ArtBench directory.")
    source.add_argument("--hf-dataset", help="HuggingFace Hub dataset repo id.")
    parser.add_argument("--split", default="train", help="Split/subfolder to read (default: train).")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--style-column", default=None)
    parser.add_argument("--limit", type=int, default=100, help="Max images to analyze.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--with-segmentation",
        action="store_true",
        help="Also run Mask R-CNN instance segmentation (slower).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "results.csv"),
        help="Where to write the results CSV.",
    )
    return parser.parse_args()


def build_samples(args: argparse.Namespace) -> list[ArtworkSample]:
    if args.local_dir:
        raw = iter_local_directory(args.local_dir, split=args.split)
    else:
        raw = iter_hf_dataset(
            args.hf_dataset,
            split=args.split,
            image_column=args.image_column,
            label_column=args.label_column,
            style_column=args.style_column,
        )
    return sample_balanced(raw, limit=args.limit, seed=args.seed)


def main() -> None:
    args = parse_args()
    samples = build_samples(args)
    if not samples:
        print("No samples found.", file=sys.stderr)
        sys.exit(1)

    detector = ArtLikenessDetector()
    composition_analyzer = CompositionAnalyzer()

    rows = []
    start = time.monotonic()
    for i, sample in enumerate(samples, start=1):
        try:
            ai_result = detector.analyze(sample.image_bytes)
            composition_result = composition_analyzer.analyze(
                sample.image_bytes, with_segmentation=args.with_segmentation
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(samples)}] skipped ({exc})", file=sys.stderr)
            continue

        row = {
            "style": sample.style,
            "is_ai_generated": sample.is_ai_generated,
            "source": sample.source,
            "ai_likeness_percent": ai_result.ai_likeness_percent,
            "originality_score": ai_result.originality_score,
            "dominant_aesthetic": ai_result.dominant_aesthetic,
        }
        row.update(composition_result.to_dict())
        rows.append(row)

        if i % 10 == 0 or i == len(samples):
            elapsed = time.monotonic() - start
            print(f"[{i}/{len(samples)}] {elapsed:.1f}s elapsed", file=sys.stderr)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
