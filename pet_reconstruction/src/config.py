"""Dataclass-based configuration for the PET reconstruction MVP.

All numerical hyperparameters live here; the source files import a single
config object and never hard-code values. See memoria/info.md "Arquitectura
y paradigma de entrenamiento" for the rationale of every value.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Repo root = parent of the `pet_reconstruction/` package directory.
# Anchoring data paths here makes the CLI work from any CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataConfig:
    # --- Raw inputs ---
    # Paths are anchored to the repo root (parent of pet_reconstruction/) so the
    # CLI works regardless of CWD.
    raw_dataset_dir: Path = _REPO_ROOT / "res/dataset/compressed_PET"
    full_dose_subdir: str = "dose_Full"
    low_dose_subdir: str = "dose_20"
    full_suffix: str = "_Full_dose.nii.gz"
    # The low-dose filenames in this dataset come with two different separators
    # between the patient id and the "1-20 dose" tag — either a space or an underscore.
    low_suffix_variants: tuple = (" 1-20 dose.nii.gz", "_1-20 dose.nii.gz")

    # --- Cached preprocessed slices ---
    cache_dir: Path = _REPO_ROOT / "data/pet_cache"
    splits_path: Path = _REPO_ROOT / "data/splits.json"

    # --- Crop & resize ---
    image_size: int = 256
    # Threshold (in PET counts) above which a voxel counts as foreground for
    # bounding-box and slice-keep decisions. Empirically PET background is exact zero.
    foreground_threshold: float = 1.0

    # --- asinh normalization (variance-stabilizing, per-volume) ---
    asinh_k: float = 10.0
    norm_percentile: float = 99.5

    # --- Slice filter ---
    # Minimum fraction of foreground pixels (within the cropped bbox) required
    # to keep an axial slice in the training set.
    min_foreground_fraction: float = 0.01

    # --- Splits ---
    train_frac: float = 0.80
    val_frac: float = 0.10
    test_frac: float = 0.10
    split_seed: int = 0


@dataclass
class ModelConfig:
    block_out_channels: tuple = (128, 128, 256, 256, 512, 512)
    layers_per_block: int = 2
    # Attention block at the 5th level (index 4), down + up — i.e., on the
    # 16x16 feature map when the input is 256x256.


@dataclass
class TrainConfig:
    # --- Diffusion paradigm ---
    num_train_timesteps: int = 1000
    prediction_type: str = "v_prediction"
    # "squaredcos_cap_v2" is diffusers' Nichol & Dhariwal cosine schedule with the
    # numerical-stability cap on beta_t.
    beta_schedule: str = "squaredcos_cap_v2"

    # --- Optimization budget ---
    # Tuned for a single 32 GB CUDA GPU (RTX PRO 4500 Blackwell). On MPS, drop
    # train_batch_size to 4, set gradient_accumulation_steps to 4, and switch
    # mixed_precision back to "no".
    train_batch_size: int = 8
    gradient_accumulation_steps: int = 4  # effective batch size = 32
    num_epochs: int = 100
    # Scaled ~sqrt(32/16) from the original 1e-4 baseline to track the larger batch.
    learning_rate: float = 1.4e-4
    lr_warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    # --- Stabilization ---
    use_ema: bool = True
    ema_decay: float = 0.9999

    # bf16 on Blackwell tensor cores: ~2x throughput vs fp32 with no NaN risk.
    mixed_precision: str = "bf16"

    # --- DataLoader ---
    num_workers: int = 12
    pin_memory: bool = True

    # --- Checkpointing / logging ---
    save_model_epochs: int = 10
    output_root: Path = Path("checkpoints")
    seed: int = 0


@dataclass
class SampleConfig:
    num_inference_steps: int = 50
    ddim_eta: float = 0.0  # deterministic

    # --- Pipeline B (DPS) only ---
    dps_omega: float = 1.0  # measurement-consistency guidance scale


@dataclass
class SupervisedConfig:
    pipeline: str = "supervised"
    in_channels: int = 2   # noisy full-dose ⊕ low-dose
    out_channels: int = 1
    output_subdir: str = "supervised"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


@dataclass
class UnconditionalConfig:
    pipeline: str = "unconditional"
    in_channels: int = 1
    out_channels: int = 1
    output_subdir: str = "unconditional"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)


def apply_smoke_overrides(cfg) -> None:
    """Shrink a config in-place for a fast end-to-end smoke test.

    Smoke variant per memoria: 128² resolution, 5 epochs, save every epoch.
    Patient pool is reduced separately by passing --limit to preprocess.
    """
    cfg.data.image_size = 128
    cfg.train.num_epochs = 5
    cfg.train.save_model_epochs = 1
