"""Image-quality + intensity-preservation metrics for PET reconstruction.

Inputs are 2D numpy arrays. Image-quality metrics operate in normalized
([-1, 1]) space; intensity-preservation works in ORIGINAL count space and
takes a foreground mask.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ---------------------------------------------------------------------------
# Generic image-quality (per slice)
# ---------------------------------------------------------------------------

def psnr(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    """PSNR in dB. data_range=2.0 for inputs normalized to [-1, +1]."""
    return float(peak_signal_noise_ratio(target, pred, data_range=data_range))


def ssim(pred: np.ndarray, target: np.ndarray, data_range: float = 2.0) -> float:
    """Mean SSIM between two 2D arrays."""
    return float(structural_similarity(target, pred, data_range=data_range))


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Normalized RMSE: sqrt(MSE) / (max(target) - min(target))."""
    rmse_val = float(np.sqrt(((pred - target) ** 2).mean()))
    rng = float(target.max() - target.min())
    return rmse_val / rng if rng > 1e-12 else 0.0


def slice_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    foreground_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """PSNR/SSIM/NRMSE on a single 2D slice, both whole-image and (optionally) foreground-only.

    For foreground SSIM we zero out the background of both arrays rather than
    extracting a 1D foreground vector, since SSIM relies on the 2D local windows.
    """
    out = {
        "psnr_whole": psnr(pred, target),
        "ssim_whole": ssim(pred, target),
        "nrmse_whole": nrmse(pred, target),
    }
    if foreground_mask is not None and foreground_mask.any():
        out["psnr_fg"] = psnr(pred[foreground_mask], target[foreground_mask])
        out["ssim_fg"] = ssim(pred * foreground_mask, target * foreground_mask)
        out["nrmse_fg"] = nrmse(pred[foreground_mask], target[foreground_mask])
    return out


# ---------------------------------------------------------------------------
# PET-specific intensity preservation (per volume, original counts space)
# ---------------------------------------------------------------------------

def intensity_preservation(
    pred_counts: np.ndarray,
    target_counts: np.ndarray,
    foreground_mask: np.ndarray,
) -> dict[str, float]:
    """Percent-error of foreground MEAN and MAX intensity in original count space.

    Acts as a proxy for SUV preservation when injected-dose / patient-weight
    metadata is unavailable.
    """
    if not foreground_mask.any():
        return {"mean_pct_err": 0.0, "max_pct_err": 0.0}
    fg_pred = pred_counts[foreground_mask]
    fg_target = target_counts[foreground_mask]
    mean_pred, mean_target = fg_pred.mean(), fg_target.mean()
    max_pred, max_target = fg_pred.max(), fg_target.max()
    return {
        "mean_pct_err": float(100.0 * (mean_pred - mean_target) / (mean_target + 1e-12)),
        "max_pct_err": float(100.0 * (max_pred - max_target) / (max_target + 1e-12)),
    }


# ---------------------------------------------------------------------------
# Volume-level aggregation
# ---------------------------------------------------------------------------

def aggregate_volume_metrics(per_slice: list[dict[str, float]]) -> dict[str, float]:
    """Mean across slices for each metric key present in the per-slice dicts."""
    if not per_slice:
        return {}
    keys = set().union(*per_slice)
    return {k: float(np.mean([s[k] for s in per_slice if k in s])) for k in keys}
