"""Offline preprocessing: paired NIfTI volumes -> cached 2D slice tensors.

Runs once before training. Each output slice is a (image_size, image_size)
float16 tensor in normalized [-1, +1] asinh space, stored as a .pt file.

Cache layout under `cache_dir`:
    metadata.json                  per-volume (M, k, bbox, kept_indices, affine, ...)
    {patient_id}/full/{NNN}.pt     normalized full-dose slice at original z index NNN
    {patient_id}/low/{NNN}.pt      normalized low-dose slice at the same z index

Usage:
    python -m src.pet_reconstruction.preprocess
    python -m src.pet_reconstruction.preprocess --smoke --limit 50
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import DataConfig
from .splits import discover_patient_ids, ensure_splits, match_low_dose
from .volume_io import (
    asinh_normalize,
    compute_foreground_bbox,
    compute_norm_percentile,
    crop_with_bbox,
    load_volume,
    resize_axial,
)


def preprocess_volume(
    patient_id: str,
    full_dose_dir: Path,
    low_dose_dir: Path,
    cache_dir: Path,
    cfg: DataConfig,
) -> dict:
    """Preprocess one paired volume and write its per-slice .pt files.

    Returns the metadata dict describing this volume (M, k, bbox, kept_indices,
    affine, original_shape). The caller is responsible for accumulating these
    into the global metadata.json.
    """
    full_path = full_dose_dir / f"{patient_id}{cfg.full_suffix}"
    low_path = match_low_dose(patient_id, low_dose_dir, cfg.low_suffix_variants)

    full_vol, full_affine, _ = load_volume(full_path)
    low_vol, _, _ = load_volume(low_path)
    assert full_vol.shape == low_vol.shape, (
        f"Paired volume shape mismatch for {patient_id}: {full_vol.shape} vs {low_vol.shape}"
    )

    # 1) Foreground bbox computed on FULL dose only, applied identically to both.
    bbox = compute_foreground_bbox(full_vol, threshold=cfg.foreground_threshold)
    full_crop = crop_with_bbox(full_vol, bbox)
    low_crop = crop_with_bbox(low_vol, bbox)

    # 2) Per-slice axial resize to (image_size, image_size).
    full_resized = resize_axial(full_crop, cfg.image_size)  # (S, S, Z)
    low_resized = resize_axial(low_crop, cfg.image_size)

    # 3) Per-volume normalization scale M from FULL dose; same M applied to LOW dose.
    M = compute_norm_percentile(full_resized, cfg.norm_percentile)
    full_norm = asinh_normalize(full_resized, M, cfg.asinh_k).astype(np.float32)
    low_norm = asinh_normalize(low_resized, M, cfg.asinh_k).astype(np.float32)

    # 4) Slice filter: keep slices whose foreground fraction (in original count space,
    #    measured on the cropped+resized full-dose) is above the threshold.
    fg_frac = (full_resized > cfg.foreground_threshold).mean(axis=(0, 1))  # (Z,)

    full_cache = cache_dir / patient_id / "full"
    low_cache = cache_dir / patient_id / "low"
    full_cache.mkdir(parents=True, exist_ok=True)
    low_cache.mkdir(parents=True, exist_ok=True)

    kept_indices: list[int] = []
    Z = full_norm.shape[2]
    for z in range(Z):
        if float(fg_frac[z]) < cfg.min_foreground_fraction:
            continue
        idx_str = f"{z:04d}"
        torch.save(
            torch.from_numpy(full_norm[:, :, z]).to(torch.float16),
            full_cache / f"{idx_str}.pt",
        )
        torch.save(
            torch.from_numpy(low_norm[:, :, z]).to(torch.float16),
            low_cache / f"{idx_str}.pt",
        )
        kept_indices.append(z)

    return {
        "patient_id": patient_id,
        "M": float(M),
        "k": float(cfg.asinh_k),
        "bbox": [(int(s.start), int(s.stop)) for s in bbox],
        "affine": full_affine.tolist(),
        "kept_indices": kept_indices,
        "original_shape": [int(s) for s in full_vol.shape],
        "image_size": int(cfg.image_size),
    }


def preprocess_all(cfg: DataConfig | None = None, limit: int | None = None) -> None:
    """Preprocess every paired volume in the dataset (or first `limit` for smoke tests).

    Also (re-)builds the patient-level split JSON when called for the first time,
    so the cache and the splits are guaranteed to refer to the same patient pool.
    """
    cfg = cfg or DataConfig()
    full_dose_dir = cfg.raw_dataset_dir / cfg.full_dose_subdir
    low_dose_dir = cfg.raw_dataset_dir / cfg.low_dose_subdir
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    patient_ids = discover_patient_ids(full_dose_dir, cfg.full_suffix)
    if limit is not None:
        patient_ids = patient_ids[:limit]

    metadata: dict[str, dict] = {}
    for pid in tqdm(patient_ids, desc="Preprocessing volumes"):
        try:
            metadata[pid] = preprocess_volume(pid, full_dose_dir, low_dose_dir, cfg.cache_dir, cfg)
        except FileNotFoundError as exc:
            print(f"[skip] {pid}: {exc}")

    (cfg.cache_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Cached {len(metadata)} volumes to {cfg.cache_dir}")

    # Splits are derived from the FULL patient pool (not just `limit`d subset) so the
    # 80/10/10 ratios reflect the real dataset even during smoke tests.
    splits = ensure_splits(
        cfg.splits_path,
        full_dose_dir,
        cfg.full_suffix,
        cfg.train_frac,
        cfg.val_frac,
        cfg.test_frac,
        cfg.split_seed,
    )
    print(
        f"Splits at {cfg.splits_path}: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess PET volumes into a slice cache.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N volumes.")
    parser.add_argument(
        "--smoke", action="store_true", help="Use 128x128 resolution (smoke test variant)."
    )
    args = parser.parse_args()

    cfg = DataConfig()
    if args.smoke:
        cfg.image_size = 128
    preprocess_all(cfg, limit=args.limit)
