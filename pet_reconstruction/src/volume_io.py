"""NIfTI I/O, foreground bbox, axial resize, asinh normalization + inverse.

Every transformation that touches PET intensities lives here so the inverse
is colocated with the forward — important for inference, where the model
output has to be mapped back to original count space before metrics or NIfTI
output.
"""

from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


ArrayLike = Union[np.ndarray, torch.Tensor]


# ---------------------------------------------------------------------------
# NIfTI read / write
# ---------------------------------------------------------------------------

def load_volume(path: Path) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Read a NIfTI file as float32 (saves memory vs nibabel's float64 default)."""
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32), img.affine, img.header


def save_volume(array: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Write a NIfTI file preserving the affine matrix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(array.astype(np.float32), affine)
    nib.save(img, str(path))


# ---------------------------------------------------------------------------
# Foreground bounding box (per volume, computed on the full-dose scan)
# ---------------------------------------------------------------------------

def compute_foreground_bbox(
    volume: np.ndarray, threshold: float = 1.0
) -> tuple[slice, slice, slice]:
    """Smallest axis-aligned bbox enclosing all voxels above `threshold`.

    Returns three Python slices suitable for direct indexing: volume[bbox].
    """
    mask = volume > threshold
    if not mask.any():
        return tuple(slice(0, n) for n in volume.shape)  # degenerate: empty volume
    slices = []
    for ax in range(3):
        other_axes = tuple(j for j in range(3) if j != ax)
        idx = np.where(mask.any(axis=other_axes))[0]
        slices.append(slice(int(idx.min()), int(idx.max()) + 1))
    return tuple(slices)


def crop_with_bbox(volume: np.ndarray, bbox: tuple[slice, slice, slice]) -> np.ndarray:
    return volume[bbox]


# ---------------------------------------------------------------------------
# Axial resize (treats last dim as Z)
# ---------------------------------------------------------------------------

def resize_axial(volume: np.ndarray, target_size: int) -> np.ndarray:
    """Resize each axial slice (H, W) of a (H, W, Z) volume to (target, target, Z)."""
    H, W, Z = volume.shape
    # (Z, 1, H, W): treat Z as batch for 2D bilinear interpolation
    t = torch.from_numpy(volume).float().permute(2, 0, 1).unsqueeze(1)
    t = F.interpolate(t, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return t.squeeze(1).permute(1, 2, 0).numpy()


def resize_axial_back(volume: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Inverse-shape resize: (H', W', Z) -> (target_h, target_w, Z). Same algorithm."""
    H, W, Z = volume.shape
    t = torch.from_numpy(volume).float().permute(2, 0, 1).unsqueeze(1)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(1).permute(1, 2, 0).numpy()


# ---------------------------------------------------------------------------
# asinh normalization
# ---------------------------------------------------------------------------

def compute_norm_percentile(volume: np.ndarray, percentile: float = 99.5) -> float:
    """Normalization scale M = `percentile`-th percentile of FOREGROUND voxels.

    Foreground = strictly positive counts. Falls back to 1.0 for empty volumes.
    """
    fg = volume[volume > 0]
    if fg.size == 0:
        return 1.0
    return float(np.percentile(fg, percentile))


def asinh_normalize(x: ArrayLike, M: float, k: float = 10.0) -> ArrayLike:
    """Map non-negative counts to ~[-1, +1] via variance-stabilizing asinh.

        x' = 2 * arcsinh(x / k) / arcsinh(M / k) - 1

    For x ∈ [0, M] the output is monotonically in [-1, +1]; values above M map
    above +1 (kept rather than clipped, so the inverse stays exact).
    """
    scale = np.arcsinh(M / k)
    if isinstance(x, torch.Tensor):
        return 2.0 * torch.asinh(x / k) / scale - 1.0
    return 2.0 * np.arcsinh(x / k) / scale - 1.0


def asinh_denormalize(x: ArrayLike, M: float, k: float = 10.0) -> ArrayLike:
    """Exact inverse of asinh_normalize."""
    scale = np.arcsinh(M / k)
    if isinstance(x, torch.Tensor):
        return torch.sinh((x + 1.0) * 0.5 * scale) * k
    return np.sinh((x + 1.0) * 0.5 * scale) * k


# ---------------------------------------------------------------------------
# Inverse pipeline: kept normalized slices -> full original-grid count volume
# ---------------------------------------------------------------------------

def reassemble_to_original_grid(
    normalized_slices: np.ndarray,  # (kept_Z, image_size, image_size)
    kept_indices: list[int],
    bbox: tuple[slice, slice, slice],
    original_shape: tuple[int, int, int],
    M: float,
    k: float,
) -> np.ndarray:
    """Invert (asinh -> resize -> bbox crop -> slice filter) for a full volume.

    Slices not in `kept_indices` are filled with zero (out-of-body padding).
    """
    H_full, W_full, Z_full = original_shape
    H_bbox = bbox[0].stop - bbox[0].start
    W_bbox = bbox[1].stop - bbox[1].start

    # 1) inverse asinh -> counts space
    counts_slices = asinh_denormalize(normalized_slices, M, k)  # (kept_Z, S, S)

    # 2) resize back to bbox spatial extent (each kept slice independently)
    t = torch.from_numpy(counts_slices).float().unsqueeze(1)  # (kept_Z, 1, S, S)
    t = F.interpolate(t, size=(H_bbox, W_bbox), mode="bilinear", align_corners=False)
    bbox_slices = t.squeeze(1).numpy()  # (kept_Z, H_bbox, W_bbox)

    # 3) write into the full-grid volume at the bbox location + correct z
    full = np.zeros(original_shape, dtype=np.float32)
    for i, z in enumerate(kept_indices):
        full[bbox[0], bbox[1], z] = bbox_slices[i]
    return full
