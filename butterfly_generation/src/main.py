import glob
import os

import torch
from diffusers import DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup

from .config import config
from .data import build_dataloader
from .model import build_model, build_noise_scheduler
from .train import train_loop


def find_checkpoint(output_dir, resume_setting):
    if not os.path.isdir(output_dir):
        return None
    if resume_setting == "latest":
        checkpoints = sorted(glob.glob(os.path.join(output_dir, "checkpoint-epoch-*")))
        return checkpoints[-1] if checkpoints else None
    candidate = os.path.join(output_dir, resume_setting)
    return candidate if os.path.isdir(candidate) else None


def main():
    train_dataloader = build_dataloader()
    noise_scheduler = build_noise_scheduler()

    resume_from = None
    if config.resume_from_checkpoint:
        resume_from = find_checkpoint(config.output_dir, config.resume_from_checkpoint)

    if resume_from:
        print(f"Loading model weights from checkpoint: {resume_from}")
        model = DDPMPipeline.from_pretrained(resume_from).unet
    else:
        model = build_model()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=len(train_dataloader) * config.num_epochs,
    )

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, resume_from=resume_from)


if __name__ == "__main__":
    main()
