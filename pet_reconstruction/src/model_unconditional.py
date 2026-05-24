"""Pipeline B — unconditional UNet prior + DDPM scheduler.

Single-channel input: only the noisy full-dose slice. The low-dose y is
NOT used during training; it enters only via DPS guidance at inference time.
"""

from diffusers import DDPMScheduler, UNet2DModel

from ._unet_builder import build_unet
from .config import UnconditionalConfig


def build_model(cfg: UnconditionalConfig) -> UNet2DModel:
    return build_unet(
        image_size=cfg.data.image_size,
        in_channels=cfg.in_channels,    # 1: noisy full-dose only
        out_channels=cfg.out_channels,  # 1
        model_cfg=cfg.model,
    )


def build_noise_scheduler(cfg: UnconditionalConfig) -> DDPMScheduler:
    """Identical scheduler to Pipeline A — v-prediction + cosine + 1000 timesteps."""
    return DDPMScheduler(
        num_train_timesteps=cfg.train.num_train_timesteps,
        beta_schedule=cfg.train.beta_schedule,
        prediction_type=cfg.train.prediction_type,
    )
