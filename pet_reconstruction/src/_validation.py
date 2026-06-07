"""Periodic validation metrics for convergence monitoring during training.

The training loss is a poor convergence signal — especially for diffusion, where
the v-prediction MSE is averaged over random timesteps and saturates within the
first few epochs while image quality keeps improving. The reliable signal is a
*downstream task* metric on a held-out validation subset: reconstruct a fixed set
of validation slices (exactly as at inference time) and measure them against the
full-dose target. Convergence = those metrics plateau.

This module builds a `validation_fn(model) -> {metric: value}` callback that both
training engines invoke at each checkpoint, under the same EMA-applied / eval()
context as the TensorBoard previews. Metrics live in the asinh-normalized
[-1, +1] model space (PSNR/SSIM/NRMSE, whole-image and foreground-only), which is
the same space training optimizes in — no per-volume M/k inversion is needed for a
convergence signal. Background maps to exactly -1 under asinh_normalize, so the
foreground mask is simply `target > -0.99`.
"""

from typing import Callable

import torch

from .data import PairedSliceDataset
from .metrics import aggregate_volume_metrics, slice_metrics
from .splits import load_splits

# Foreground threshold in model space: asinh_normalize maps 0 counts to exactly
# -1, so anything comfortably above -1 is foreground.
_FG_THRESHOLD = -0.99


def build_validation_fn(
    cfg,
    reconstruct_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
    num_slices: int = 16,
) -> Callable[[torch.nn.Module], dict[str, float]] | None:
    """Build a validation-metric callback over a fixed subset of val slices.

    Args:
        cfg: any config exposing `.data.splits_path` and `.data.cache_dir`.
        reconstruct_fn: maps (model, low_batch_on_device) -> predicted full-dose
            batch (N, 1, H, W) in model space. Must run the SAME reconstruction
            used at inference (DDIM for supervised, DPS for unconditional, a single
            forward for the regressors). It manages its own autograd context: the
            engine does NOT wrap the call in `no_grad`, so DPS (which needs grad
            through the U-Net) works unchanged.
        num_slices: how many evenly-spaced val slices to evaluate. Kept small so
            the per-checkpoint cost stays negligible relative to an epoch.

    Returns:
        A `validation_fn(model) -> dict[str, float]`, or None if the val split is
        empty (e.g. a smoke run with no validation patients).
    """
    val_ids = load_splits(cfg.data.splits_path)["val"]
    ds = PairedSliceDataset(cfg.data.cache_dir, val_ids)
    if len(ds) == 0:
        return None

    n = min(num_slices, len(ds))
    # Evenly spaced over the whole val set so we cover different anatomical levels.
    step = max(len(ds) // n, 1)
    indices = [min(step * i, len(ds) - 1) for i in range(n)]

    low_batch = torch.stack([ds[i]["low"] for i in indices])    # (N, 1, H, W)
    full_batch = torch.stack([ds[i]["full"] for i in indices]).float()

    def validation_fn(model: torch.nn.Module) -> dict[str, float]:
        device = next(model.parameters()).device
        pred = reconstruct_fn(model, low_batch.to(device)).detach().cpu().float()

        per_slice: list[dict[str, float]] = []
        for i in range(pred.shape[0]):
            p = pred[i, 0].numpy()
            t = full_batch[i, 0].numpy()
            mask = t > _FG_THRESHOLD
            per_slice.append(slice_metrics(p, t, mask if mask.any() else None))
        return aggregate_volume_metrics(per_slice)

    return validation_fn
