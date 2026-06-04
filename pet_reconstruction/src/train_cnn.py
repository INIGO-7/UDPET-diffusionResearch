"""Baseline B — RED-CNN trained for PET reconstruction.

Reuses the shared regression engine (`_train_engine_regression`): same paired
dataloader, MSE objective, EMA, accumulation, checkpointing and previews as
Baseline A. The only differences are the model (RED-CNN instead of the U-Net),
its forward call (no timestep) and how weights are saved (a plain state dict).

Usage (from pet_reconstruction/):
    python -m src.train_cnn
"""

import torch

from ._train_engine import resolve_resume_path
from ._train_engine_regression import run_regression_training
from .config import CNNConfig
from .data import PairedSliceDataset, build_paired_dataloader
from .model_cnn import build_model, save_redcnn
from .splits import load_splits


def _forward(model: torch.nn.Module, model_input: torch.Tensor) -> torch.Tensor:
    """RED-CNN forward — a plain CNN, no timestep argument."""
    return model(model_input)


def _build_preview(cfg: CNNConfig, num_samples: int = 2):
    """Pick fixed val slices and return (sampler, references) for TensorBoard previews."""
    val_ids = load_splits(cfg.data.splits_path)["val"]
    ds = PairedSliceDataset(cfg.data.cache_dir, val_ids)
    if len(ds) == 0:
        return None, None

    # Evenly spaced indices so the previews cover different anatomical levels.
    step = max(len(ds) // (num_samples + 1), 1)
    indices = [min(step * (i + 1), len(ds) - 1) for i in range(num_samples)]

    sample_ids: list[str] = []
    low_list: list[torch.Tensor] = []
    references: dict[str, torch.Tensor] = {}
    for idx in indices:
        full_p, _ = ds.entries[idx]
        pid = full_p.parent.parent.name
        sid = f"{pid}__{full_p.stem}"
        item = ds[idx]
        sample_ids.append(sid)
        low_list.append(item["low"])
        references[f"samples/{sid}/full"] = item["full"]
        references[f"samples/{sid}/low"] = item["low"]

    low_batch = torch.stack(low_list)  # (N, 1, H, W)

    def sampler(model: torch.nn.Module) -> dict[str, torch.Tensor]:
        device = next(model.parameters()).device
        with torch.no_grad():
            recon = _forward(model, low_batch.to(device)).cpu()
        return {f"samples/{sid}/recon": recon[i] for i, sid in enumerate(sample_ids)}

    return sampler, references


def train(cfg: CNNConfig | None = None, resume_from: str | None = None) -> None:
    cfg = cfg or CNNConfig()
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

    def prepare_input(batch: dict) -> torch.Tensor:
        return batch["low"]

    preview_sampler, preview_refs = _build_preview(cfg)

    run_regression_training(
        cfg,
        model,
        train_loader,
        output_dir,
        prepare_model_input=prepare_input,
        model_forward=_forward,
        tracker_name="pet_cnn_redcnn",
        resume_from=resume_path,
        save_model_fn=save_redcnn,
        preview_sampler=preview_sampler,
        preview_references=preview_refs,
    )


if __name__ == "__main__":
    train()
