"""Pipeline B — unconditional v-DDPM prior training on full-dose slices.

No conditioning: the U-Net only sees the noisy full-dose slice. The low-dose
observation is reserved for inference-time DPS guidance.

Usage:
    python -m src.pet_reconstruction.train_unconditional
"""

import torch

from ._train_engine import run_training
from .config import UnconditionalConfig
from .data import build_unpaired_dataloader
from .model_unconditional import build_model, build_noise_scheduler


def train(cfg: UnconditionalConfig | None = None) -> None:
    cfg = cfg or UnconditionalConfig()
    output_dir = cfg.train.output_root / cfg.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_unpaired_dataloader(
        cfg.data.cache_dir,
        cfg.data.splits_path,
        split="train",
        batch_size=cfg.train.train_batch_size,
    )
    model = build_model(cfg)
    noise_scheduler = build_noise_scheduler(cfg)

    def prepare_input(batch: dict, noisy_full: torch.Tensor) -> torch.Tensor:
        # No conditioning: feed the noisy slice unchanged.
        return noisy_full

    run_training(
        cfg,
        model,
        noise_scheduler,
        train_loader,
        output_dir,
        prepare_model_input=prepare_input,
        tracker_name="pet_unconditional",
    )


if __name__ == "__main__":
    train()
