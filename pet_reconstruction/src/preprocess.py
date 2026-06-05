"""Offline preprocessing: paired NIfTI volumes -> cached 2D slice tensors.

Runs once before training. Each output slice is a (image_size, image_size)
float16 tensor in normalized [-1, +1] asinh space, stored as a .pt file.

Cache layout under `cache_dir`:
    metadata.json                  per-volume (M, k, bbox, kept_indices, affine, ...)
    {patient_id}/full/{NNN}.pt     normalized full-dose slice at original z index NNN
    {patient_id}/low/{NNN}.pt      normalized low-dose slice at the same z index

Usage (from pet_reconstruction/):
    python -m src.preprocess
    python -m src.preprocess --smoke --limit 50
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


def prepare_low_dose(low_vol: np.ndarray, cfg: DataConfig) -> dict:
    """Derive the normalization geometry from a LOW-dose volume alone.

    Single source of truth shared by offline preprocessing and runtime inference,
    so the frame the model is trained in matches exactly what inference can
    reconstruct from an unseen low-dose scan. Nothing here touches the full dose:

        bbox          foreground bounding box (low-dose)
        low_resized   cropped + axially resized stack, (S, S, Z)
        M             asinh percentile scale (low-dose)
        kept_indices  axial slices passing the foreground-fraction filter

    All four are later applied identically to the paired full-dose target during
    preprocessing, and recomputed on the fly during reconstruction.
    """
    bbox = compute_foreground_bbox(low_vol, threshold=cfg.foreground_threshold)
    low_resized = resize_axial(crop_with_bbox(low_vol, bbox), cfg.image_size)  # (S, S, Z)
    M = compute_norm_percentile(low_resized, cfg.norm_percentile)
    fg_frac = (low_resized > cfg.foreground_threshold).mean(axis=(0, 1))  # (Z,)
    kept_indices = [
        z for z in range(low_resized.shape[2])
        if float(fg_frac[z]) >= cfg.min_foreground_fraction
    ]
    return {"bbox": bbox, "low_resized": low_resized, "M": float(M), "kept_indices": kept_indices}


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

    # Bbox, normalization scale M and the slice filter are ALL derived from the
    # LOW dose alone (the only volume available at inference) and applied
    # identically to the full-dose target. prepare_low_dose() is the single source
    # of truth shared with the reconstruction path, so train/inference match.
    prep = prepare_low_dose(low_vol, cfg)
    bbox = prep["bbox"]
    M = prep["M"]
    kept_indices = prep["kept_indices"]
    low_resized = prep["low_resized"]

    # Crop + resize the full-dose target with the SAME low-derived bbox, then
    # normalize both volumes with the SAME low-derived M.
    full_resized = resize_axial(crop_with_bbox(full_vol, bbox), cfg.image_size)  # (S, S, Z)
    full_norm = asinh_normalize(full_resized, M, cfg.asinh_k).astype(np.float32)
    low_norm = asinh_normalize(low_resized, M, cfg.asinh_k).astype(np.float32)

    full_cache = cache_dir / patient_id / "full"
    low_cache = cache_dir / patient_id / "low"
    full_cache.mkdir(parents=True, exist_ok=True)
    low_cache.mkdir(parents=True, exist_ok=True)

    for z in kept_indices:
        idx_str = f"{z:04d}"
        full_pt = full_cache / f"{idx_str}.pt"
        low_pt = low_cache / f"{idx_str}.pt"
        if full_pt.exists() and low_pt.exists():
            continue
        torch.save(torch.from_numpy(full_norm[:, :, z]).to(torch.float16), full_pt)
        torch.save(torch.from_numpy(low_norm[:, :, z]).to(torch.float16), low_pt)

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

    # Resume support: load any metadata from a previous (possibly interrupted) run
    # and skip volumes already fully cached. metadata.json is rewritten after every
    # volume so an interrupted run never loses completed work.
    metadata_path = cfg.cache_dir / "metadata.json"
    metadata: dict[str, dict] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print(f"Resuming: {len(metadata)} volumes already in metadata.json")

    for pid in tqdm(patient_ids, desc="Preprocessing volumes"):
        if pid in metadata:
            continue
        try:
            metadata[pid] = preprocess_volume(pid, full_dose_dir, low_dose_dir, cfg.cache_dir, cfg)
            metadata_path.write_text(json.dumps(metadata, indent=2))
        except FileNotFoundError as exc:
            print(f"[skip] {pid}: {exc}")

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
