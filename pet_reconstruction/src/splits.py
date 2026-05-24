"""Patient-level 80/10/10 split shared by both pipelines.

The split is deterministic given the seed and persisted to JSON, so the
supervised and unconditional pipelines are evaluated on EXACTLY the same
test volumes — a precondition for the headline A vs B comparison.
"""

import json
import random
from pathlib import Path
from typing import Iterable


def discover_patient_ids(full_dose_dir: Path, full_suffix: str) -> list[str]:
    """Enumerate patient IDs from the full-dose folder.

    A patient ID is the filename with `full_suffix` stripped, e.g.
        '01122021_1_20211201_164050_Full_dose.nii.gz' -> '01122021_1_20211201_164050'.
    """
    ids = []
    for path in sorted(full_dose_dir.glob(f"*{full_suffix}")):
        ids.append(path.name[: -len(full_suffix)])
    return ids


def match_low_dose(patient_id: str, low_dose_dir: Path, suffix_variants: tuple) -> Path:
    """Find the low-dose NIfTI matching a patient ID, trying each suffix variant."""
    for sfx in suffix_variants:
        candidate = low_dose_dir / f"{patient_id}{sfx}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No low-dose match for patient {patient_id!r} in {low_dose_dir} "
        f"(tried suffixes {suffix_variants})"
    )


def build_splits(
    patient_ids: Iterable[str],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    test_frac: float = 0.10,
    seed: int = 0,
) -> dict[str, list[str]]:
    """Deterministic random 3-way split by patient ID."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, (
        f"Fractions must sum to 1: got {train_frac}+{val_frac}+{test_frac}"
    )
    ids = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": ids[:n_train],
        "val": ids[n_train : n_train + n_val],
        "test": ids[n_train + n_val :],
    }


def save_splits(splits: dict[str, list[str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2))


def load_splits(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text())


def ensure_splits(
    path: Path,
    full_dose_dir: Path,
    full_suffix: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, list[str]]:
    """Idempotent: load existing splits or build, save, and return new ones."""
    if path.exists():
        return load_splits(path)
    ids = discover_patient_ids(full_dose_dir, full_suffix)
    splits = build_splits(ids, train_frac, val_frac, test_frac, seed)
    save_splits(splits, path)
    return splits
