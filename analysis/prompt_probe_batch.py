"""Tests a candidate two-pole prompt pair against a whole labeled genre sample,
instead of eyeballing one image at a time (see analysis/prompt_probe.py for that).

Reuses analysis/output/<genre>/results.csv (path + is_ai_generated columns,
produced by `python -m analysis.run_analysis --style <genre> ...`) as ground truth.

Usage:
    python -m analysis.prompt_probe_batch baroque \\
        --pole-a "a baroque painting with an ornate gilded frame" \\
        --pole-b "a baroque painting"

Reports the mean pole-A percentage for human vs. AI-generated images, and the
ROC AUC of pole-A percentage as a predictor of is_ai_generated - AUC of 0.5
means the wording has no discriminative power for this genre; further from
0.5 (in either direction) means it does.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from analysis.prompt_probe import MODEL_ID, TEMPERATURE, prepare_image

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("genre", help="Genre name matching analysis/output/<genre>/results.csv")
    parser.add_argument("--pole-a", nargs="+", required=True)
    parser.add_argument("--pole-b", nargs="+", required=True)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of images (default: all).")
    args = parser.parse_args()

    from transformers import AutoProcessor, SiglipModel

    csv_path = OUTPUT_DIR / args.genre / "results.csv"
    df = pd.read_csv(csv_path).dropna(subset=["path"])
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SiglipModel.from_pretrained(MODEL_ID).to(device).eval()

    prompts = [*args.pole_a, *args.pole_b]
    n_a = len(args.pole_a)

    percents = []
    for path in df["path"]:
        image = prepare_image(path)
        inputs = processor(text=prompts, images=image, padding="max_length", return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits_per_image.squeeze(0)

        a_mean = logits[:n_a].mean()
        b_mean = logits[n_a:].mean()
        weight_a = torch.softmax(torch.stack([a_mean, b_mean]) / args.temperature, dim=0)[0]
        percents.append(float(weight_a) * 100)

    df = df.copy()
    df["percent_a"] = percents

    human = df[~df["is_ai_generated"]]
    ai = df[df["is_ai_generated"]]
    auc = roc_auc_score(df["is_ai_generated"].astype(int), df["percent_a"])

    print(f"\ngenre: {args.genre}  (n={len(df)}, human={len(human)}, ai={len(ai)})")
    print(f"pole A: {args.pole_a}")
    print(f"pole B: {args.pole_b}")
    print(f"\nmean percent_a | human = {human['percent_a'].mean():.1f}   ai = {ai['percent_a'].mean():.1f}")
    print(f"ROC AUC (percent_a predicting is_ai_generated) = {auc:.3f}")
    print("(0.5 = no discriminative power; closer to 1.0 or 0.0 = stronger signal)")


if __name__ == "__main__":
    main()
