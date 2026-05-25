"""Pipeline A — supervised conditional v-DDPM training.

Builds the paired dataloader, the 2-channel-input UNet and the DDPM scheduler,
then delegates the v-prediction MSE training loop to `_train_engine`.

Usage (from pet_reconstruction/):
    python -m src.train_supervised
"""

import torch

from ._train_engine import resolve_resume_path, run_training
from .config import SupervisedConfig
from .data import PairedSliceDataset, build_paired_dataloader
from .model_supervised import build_model, build_noise_scheduler
from .reconstruct_supervised import _build_ddim_scheduler, ddim_sample_conditional
from .splits import load_splits


def _build_preview(cfg: SupervisedConfig, num_samples: int = 2):
    """Pick fixed val slices and return (sampler, references) for TensorBoard previews."""
    val_ids = load_splits(cfg.data.splits_path)["val"]
    ds = PairedSliceDataset(cfg.data.cache_dir, val_ids)
    if len(ds) == 0:
        return None, None

    # Evenly spaced indices so the previews cover different anatomical levels.
    step = max(len(ds) // (num_samples + 1), 1)
    indices = [min(step * (i + 1), len(ds) - 1) for i in range(num_samples)]

    sample_ids: list[str] = []
    full_list: list[torch.Tensor] = []
    low_list: list[torch.Tensor] = []
    references: dict[str, torch.Tensor] = {}
    for idx in indices:
        full_p, _ = ds.entries[idx]
        pid = full_p.parent.parent.name
        sid = f"{pid}__{full_p.stem}"
        item = ds[idx]
        sample_ids.append(sid)
        full_list.append(item["full"])
        low_list.append(item["low"])
        references[f"samples/{sid}/full"] = item["full"]
        references[f"samples/{sid}/low"] = item["low"]

    low_batch = torch.stack(low_list)  # (N, 1, H, W)
    ddim_scheduler = _build_ddim_scheduler(cfg)

    def sampler(unet: torch.nn.Module) -> dict[str, torch.Tensor]:
        device = next(unet.parameters()).device
        with torch.no_grad():
            recon = ddim_sample_conditional(
                unet,
                ddim_scheduler,
                low_batch.to(device),
                cfg.sample.num_inference_steps,
                cfg.sample.ddim_eta,
                device,
            ).cpu()
        return {f"samples/{sid}/recon": recon[i] for i, sid in enumerate(sample_ids)}

    return sampler, references


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
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
    )
    model = build_model(cfg)
    noise_scheduler = build_noise_scheduler(cfg)

    def prepare_input(batch: dict, noisy_full: torch.Tensor) -> torch.Tensor:
        # Channel concatenation: [noisy x_t, low-dose y]
        return torch.cat([noisy_full, batch["low"]], dim=1)

    preview_sampler, preview_refs = _build_preview(cfg)

    run_training(
        cfg,
        model,
        noise_scheduler,
        train_loader,
        output_dir,
        prepare_model_input=prepare_input,
        tracker_name="pet_supervised",
        resume_from=resume_path,
        preview_sampler=preview_sampler,
        preview_references=preview_refs,
    )


if __name__ == "__main__":
    train()
