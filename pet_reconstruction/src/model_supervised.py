"""Pipeline A — supervised conditional UNet + DDPM scheduler.

Input layout for the U-Net: 2 channels = [noisy full-dose, paired low-dose].
"""

from diffusers import DDPMScheduler, UNet2DModel

from ._unet_builder import build_unet
from .config import SupervisedConfig


def build_model(cfg: SupervisedConfig) -> UNet2DModel:
    return build_unet(
        image_size=cfg.data.image_size,
        in_channels=cfg.in_channels,    # 2: noisy ⊕ low-dose conditioning
        out_channels=cfg.out_channels,  # 1: v prediction over the full-dose channel
        model_cfg=cfg.model,
    )


def build_noise_scheduler(cfg: SupervisedConfig) -> DDPMScheduler:
    """v-prediction + cosine β-schedule + 1000 train timesteps."""
    return DDPMScheduler(
        num_train_timesteps=cfg.train.num_train_timesteps,
        beta_schedule=cfg.train.beta_schedule,
        prediction_type=cfg.train.prediction_type,
    )
