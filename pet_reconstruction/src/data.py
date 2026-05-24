"""PyTorch Datasets and DataLoaders backed by the preprocessed slice cache.

Two dataset variants:
    PairedSliceDataset   -> {"full", "low"}   for the supervised pipeline
    FullDoseOnlyDataset  -> {"full"}          for the unconditional prior

Both yield slices as float32 tensors of shape (1, H, W). The slice cache is
stored as float16; we upcast on load for numerical headroom in the training
forward pass.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .splits import load_splits


def _list_cache_slices(cache_dir: Path, patient_id: str, kind: str) -> list[Path]:
    """All cached slices for one patient, sorted by z-index filename."""
    return sorted((cache_dir / patient_id / kind).glob("*.pt"))


def _load_slice(path: Path) -> torch.Tensor:
    """Load a cached fp16 slice and return it as fp32 with a channel dim, shape (1, H, W)."""
    t = torch.load(path, weights_only=True)
    return t.to(torch.float32).unsqueeze(0)


class PairedSliceDataset(Dataset):
    """Yields {"full": (1,H,W), "low": (1,H,W)} pairs across the listed patients."""

    def __init__(self, cache_dir: Path, patient_ids: list[str]):
        self.entries: list[tuple[Path, Path]] = []
        for pid in patient_ids:
            for full_p in _list_cache_slices(cache_dir, pid, "full"):
                low_p = cache_dir / pid / "low" / full_p.name
                if low_p.exists():
                    self.entries.append((full_p, low_p))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        full_p, low_p = self.entries[idx]
        return {"full": _load_slice(full_p), "low": _load_slice(low_p)}


class FullDoseOnlyDataset(Dataset):
    """Yields {"full": (1,H,W)} for the unconditional prior pipeline."""

    def __init__(self, cache_dir: Path, patient_ids: list[str]):
        self.entries: list[Path] = []
        for pid in patient_ids:
            self.entries.extend(_list_cache_slices(cache_dir, pid, "full"))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"full": _load_slice(self.entries[idx])}


def build_paired_dataloader(
    cache_dir: Path,
    splits_path: Path,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    ids = load_splits(splits_path)[split]
    ds = PairedSliceDataset(cache_dir, ids)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False
    )


def build_unpaired_dataloader(
    cache_dir: Path,
    splits_path: Path,
    split: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    ids = load_splits(splits_path)[split]
    ds = FullDoseOnlyDataset(cache_dir, ids)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=False
    )
