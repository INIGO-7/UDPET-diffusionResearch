"""Dataclass-based configuration for the PET reconstruction MVP.

All numerical hyperparameters live here; the source files import a single
config object and never hard-code values. See memoria/info.md "Arquitectura
y paradigma de entrenamiento" for the rationale of every value.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    # --- Raw inputs ---
    # Paths are resolved from the pet_reconstruction/ directory (run scripts with
    # `cd pet_reconstruction && python -m src.<module>`); the shared res/ and data/
    # directories live one level up at the repo root.
    raw_dataset_dir: Path = Path("../res/dataset/compressed_PET")
    full_dose_subdir: str = "dose_Full"
    low_dose_subdir: str = "dose_20"
    full_suffix: str = "_Full_dose.nii.gz"
    # The low-dose filenames in this dataset come with two different separators
    # between the patient id and the "1-20 dose" tag — either a space or an underscore.
    low_suffix_variants: tuple = (" 1-20 dose.nii.gz", "_1-20 dose.nii.gz")

    # --- Cached preprocessed slices ---
    cache_dir: Path = Path("../data/pet_cache")
    splits_path: Path = Path("../data/splits.json")

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
    train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch size = 16
    num_epochs: int = 30
    learning_rate: float = 1e-4
    lr_warmup_steps: int = 500
    grad_clip_norm: float = 1.0

    # --- Stabilization ---
    use_ema: bool = True
    ema_decay: float = 0.9999

    # fp32 chosen for MPS reliability (fp16 has historical NaN risk on MPS).
    mixed_precision: str = "no"

    # --- Checkpointing / logging ---
    save_model_epochs: int = 5
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
