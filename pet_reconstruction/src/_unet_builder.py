"""Shared UNet builder for both pipelines.

The architecture is the butterfly-tutorial UNet2DModel exactly, with only
the input/output channels parameterized so the same builder serves both
pipeline A (in_channels=2) and pipeline B (in_channels=1).
"""

from diffusers import UNet2DModel

from .config import ModelConfig


def build_unet(
    image_size: int,
    in_channels: int,
    out_channels: int,
    model_cfg: ModelConfig,
) -> UNet2DModel:
    """Build a UNet2DModel with attention at the 5th level only.

    Matches the butterfly_generation architecture for direct comparability;
    only the in/out channels differ between pipelines.
    """
    return UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=model_cfg.layers_per_block,
        block_out_channels=model_cfg.block_out_channels,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
