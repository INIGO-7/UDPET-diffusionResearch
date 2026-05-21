import argparse
import secrets
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import DDPMPipeline

from config import config


MODEL_DIR = Path(__file__).parent / config.output_dir


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(seed=None, num_inference_steps=None):
    if seed is None:
        seed = secrets.randbits(32)

    pipeline = DDPMPipeline.from_pretrained(MODEL_DIR)
    pipeline.to(pick_device())

    generator = torch.Generator(device="cpu").manual_seed(seed)
    kwargs = {"batch_size": 1, "generator": generator}
    if num_inference_steps is not None:
        kwargs["num_inference_steps"] = num_inference_steps

    image = pipeline(**kwargs).images[0]
    return image, seed


def show(image, seed, save_path=None):
    if save_path is not None:
        image.save(save_path)
        print(f"Saved butterfly to {save_path}")

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Generated butterfly (seed={seed})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a new butterfly with the trained DDPM.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible sampling.")
    parser.add_argument("--save", type=Path, default=None, help="Optional path to save the image.")
    parser.add_argument("--steps", type=int, default=None, help="Override the number of inference steps.")
    args = parser.parse_args()

    image, seed = generate(seed=args.seed, num_inference_steps=args.steps)
    print(f"seed={seed}")
    show(image, seed, save_path=args.save)


if __name__ == "__main__":
    main()
