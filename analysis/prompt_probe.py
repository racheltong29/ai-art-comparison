"""Standalone tool for experimenting with SigLIP prompt wording on a single image.

Edit POLE_A / POLE_B below (or pass --pole-a / --pole-b) and run:

    python -m analysis.prompt_probe path/to/image.jpg

    python -m analysis.prompt_probe path/to/image.jpg \\
        --pole-a "a photo of a cat" \\
        --pole-b "a photo of a dog"

Each pole can have multiple prompts (like production's 3+3 setup) - they're
averaged before the final softmax, same as backend/detector.py does. Pole A's
softmax weight is reported as "percent_a" (production calls this ai_likeness).
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel

MODEL_ID = "google/siglip-base-patch16-224"
MAX_EDGE = 512
TEMPERATURE = 1.5

# Edit these directly for quick iteration instead of passing --pole-a/--pole-b every time.
POLE_A = [
    "generic AI generated digital art with smooth polished rendering",
    "synthetic illustration typical of Midjourney or Stable Diffusion",
    "stock AI artwork with perfect lighting and generic composition",
]
POLE_B = [
    "original hand-painted digital art with visible personal brushwork",
    "rough sketchy illustration with unique human imperfections",
    "authentic human-created artwork with distinctive stylistic choices",
]


def prepare_image(path: str) -> Image.Image:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest > MAX_EDGE:
        scale = MAX_EDGE / longest
        image = image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", help="Path to an image file.")
    parser.add_argument("--pole-a", nargs="+", default=None, help="One or more prompts for pole A (overrides POLE_A).")
    parser.add_argument("--pole-b", nargs="+", default=None, help="One or more prompts for pole B (overrides POLE_B).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    args = parser.parse_args()

    pole_a = args.pole_a or POLE_A
    pole_b = args.pole_b or POLE_B

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = SiglipModel.from_pretrained(MODEL_ID).to(device).eval()

    image = prepare_image(args.image)
    prompts = [*pole_a, *pole_b]
    inputs = processor(text=prompts, images=image, padding="max_length", return_tensors="pt").to(device)

    with torch.inference_mode():
        logits = model(**inputs).logits_per_image.squeeze(0)

    print(f"\nImage: {args.image}  (device={device})\n")
    print("Per-prompt raw logits:")
    for prompt, logit in zip(prompts, logits.tolist()):
        pole = "A" if prompt in pole_a else "B"
        print(f"  [{pole}] {logit:7.3f}  {prompt!r}")

    a_mean = logits[: len(pole_a)].mean()
    b_mean = logits[len(pole_a) :].mean()
    weights = torch.softmax(torch.stack([a_mean, b_mean]) / args.temperature, dim=0)

    print(f"\npole A mean logit: {float(a_mean):.3f}")
    print(f"pole B mean logit: {float(b_mean):.3f}")
    print(f"\n=> percent_a = {float(weights[0]) * 100:.1f}")
    print(f"=> percent_b = {float(weights[1]) * 100:.1f}")


if __name__ == "__main__":
    main()
