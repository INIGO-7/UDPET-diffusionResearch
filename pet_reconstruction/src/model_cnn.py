"""Baseline B — RED-CNN, a classic CNN denoiser for low-dose CT/PET.

RED-CNN (Chen et al. 2017, "Low-Dose CT with a Residual Encoder-Decoder
Convolutional Neural Network") is the reference discriminative baseline for the
low-dose -> standard-dose task. It is a fully-convolutional residual
encoder-decoder: 5 convolutions then 5 deconvolutions, 96 channels, 5x5 kernels,
stride 1, NO pooling (full resolution throughout), with three symmetric residual
skip connections. ~1.8 M parameters — roughly 50x smaller than the diffusion
U-Net — which is precisely the point of this ablation.

One deliberate deviation from the original architecture
-------------------------------------------------------
The original RED-CNN ends with a ReLU because it was built for intensities
normalized to [0, 1]. Here the model space is the *signed* asinh [-1, +1] range
(zero counts map to -1, so the background sits at -1). A terminal ReLU would
clamp away the entire negative half of the signal, so it is removed and the
output is linear. All interior ReLUs are kept.

Padding is 0 (the faithful original), so every conv shrinks the map by
`kernel_size - 1` and every deconv grows it back symmetrically; the three skip
connections therefore align exactly.

This is not a diffusers model, so checkpoints are plain state dicts (`model.pt`)
rather than the `unet/` `save_pretrained` layout used by the diffusion pipelines.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .config import CNNConfig


class REDCNN(nn.Module):
    def __init__(
        self,
        num_filters: int = 96,
        kernel_size: int = 5,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        ch, k = num_filters, kernel_size
        self.conv1 = nn.Conv2d(in_channels, ch, k, stride=1, padding=0)
        self.conv2 = nn.Conv2d(ch, ch, k, stride=1, padding=0)
        self.conv3 = nn.Conv2d(ch, ch, k, stride=1, padding=0)
        self.conv4 = nn.Conv2d(ch, ch, k, stride=1, padding=0)
        self.conv5 = nn.Conv2d(ch, ch, k, stride=1, padding=0)
        self.tconv1 = nn.ConvTranspose2d(ch, ch, k, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(ch, ch, k, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(ch, ch, k, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(ch, ch, k, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(ch, out_channels, k, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder, stashing residuals at the input, after conv2 and after conv4.
        residual_1 = x
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # Decoder, adding the symmetric residuals back in.
        out = self.tconv1(out)
        out = out + residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out = out + residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out = out + residual_1  # global skip; NO terminal ReLU (asinh space is signed)
        return out


def build_model(cfg: CNNConfig) -> REDCNN:
    return REDCNN(
        num_filters=cfg.model.num_filters,
        kernel_size=cfg.model.kernel_size,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
    )


def save_redcnn(model: nn.Module, save_dir: Path) -> None:
    """`save_model_fn` for the regression engine: persist a plain state dict.

    RED-CNN is not a diffusers model, so it has no `save_pretrained`; weights go
    to `model.pt` inside the checkpoint dir instead of an `unet/` subdir.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "model.pt")


def load_redcnn(cfg: CNNConfig, checkpoint_dir: Path, device: str) -> REDCNN:
    """Rebuild RED-CNN from cfg and load the checkpoint's `model.pt` weights."""
    model = build_model(cfg)
    state = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()
