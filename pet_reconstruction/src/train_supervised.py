"""Pipeline A — supervised conditional v-DDPM training.

Builds the paired dataloader, the 2-channel-input UNet and the DDPM scheduler,
then delegates the v-prediction MSE training loop to `_train_engine`.

Usage (from pet_reconstruction/):
    python -m src.train_supervised
"""

import torch

from ._train_engine import resolve_resume_path, run_training
from .config import SupervisedConfig
from .data import build_paired_dataloader
from .model_supervised import build_model, build_noise_scheduler


def train(cfg: SupervisedConfig | None = None, resume_from: str | None = None) -> None:
    cfg = cfg or SupervisedConfig()
    output_dir = cfg.train.output_root / cfg.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_path = resolve_resume_path(output_dir, resume_from) if resume_from else None

    train_loader = build_paired_dataloader(
        cfg.data.cache_dir,
        cfg.data.splits_path,
        split="train",
        batch_size=cfg.train.train_batch_size,
    )
    model = build_model(cfg)
    noise_scheduler = build_noise_scheduler(cfg)

    def prepare_input(batch: dict, noisy_full: torch.Tensor) -> torch.Tensor:
        # Channel concatenation: [noisy x_t, low-dose y]
        return torch.cat([noisy_full, batch["low"]], dim=1)

    run_training(
        cfg,
        model,
        noise_scheduler,
        train_loader,
        output_dir,
        prepare_model_input=prepare_input,
        tracker_name="pet_supervised",
        resume_from=resume_path,
    )


if __name__ == "__main__":
    train()
