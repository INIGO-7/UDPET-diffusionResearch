"""Baseline A — the supervised U-Net trained as a direct regressor (no diffusion).

Reuses the *exact* diffusion architecture (`build_unet`) so the only thing that
changes versus Pipeline A is the training paradigm — this is the controlled
ablation of "does the diffusion process itself help, at equal architecture?".

Two deltas versus the diffusion model, neither of which touches the network code:
  - `in_channels=1`: the second channel of Pipeline A was the noisy x_t; here it
    is gone, the input is just the low-dose slice.
  - No `DDPMScheduler` is constructed: there is no forward noising process.

The UNet2DModel forward signature requires a `timestep`. We always pass a single
constant value (`REGRESSION_TIMESTEP`); with a constant timestep the sinusoidal
time embedding and its per-block projections collapse to a fixed, learnable bias
added inside every ResNet block. The specific value is irrelevant (the embedding
MLP adapts to it during training), so it is NOT a hyperparameter.
"""

from diffusers import UNet2DModel

from ._unet_builder import build_unet
from .config import RegressionUNetConfig

# Constant timestep fed to the U-Net on every forward pass. Any fixed value works;
# 0 is the natural choice. See module docstring for why this is inert.
REGRESSION_TIMESTEP = 0


def build_model(cfg: RegressionUNetConfig) -> UNet2DModel:
    return build_unet(
        image_size=cfg.data.image_size,
        in_channels=cfg.in_channels,    # 1: low-dose only
        out_channels=cfg.out_channels,  # 1: direct full-dose estimate
        model_cfg=cfg.model,
    )
