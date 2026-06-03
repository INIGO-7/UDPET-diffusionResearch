"""Pipeline A inference: reconstruct test volumes with the supervised checkpoint.

For each volume:
    - load the raw low-dose NIfTI and derive bbox / asinh M / kept slices from it
      AT RUNTIME (prepare_low_dose) -- no full-dose info, no preprocess cache
    - run DDIM-50 sampling with the low-dose slice as the second input channel
    - assemble a (image_size, image_size, kept_Z) normalized recon
    - invert (asinh -> resize -> bbox) back to the original-grid count volume
    - save as a NIfTI with the original affine

Usage (from pet_reconstruction/):
    python -m src.reconstruct_supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029
    # ...or reconstruct a single unseen low-dose volume directly:
    python -m src.reconstruct_supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
        --low-volume /path/to/unseen_1-20_dose.nii.gz
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DModel
from tqdm.auto import tqdm

from .config import SupervisedConfig
from .preprocess import prepare_low_dose
from .splits import load_splits, match_low_dose
from .volume_io import (
    asinh_normalize,
    load_volume,
    reassemble_to_original_grid,
    save_volume,
)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_ddim_scheduler(cfg: SupervisedConfig) -> DDIMScheduler:
    """Construct a DDIM scheduler that mirrors the training-time DDPM config."""
    return DDIMScheduler(
        num_train_timesteps=cfg.train.num_train_timesteps,
        beta_schedule=cfg.train.beta_schedule,
        prediction_type=cfg.train.prediction_type,
    )


@torch.no_grad()
def ddim_sample_conditional(
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    low_batch: torch.Tensor,  # (B, 1, H, W), normalized
    num_inference_steps: int,
    eta: float,
    device: str,
) -> torch.Tensor:
    """DDIM reverse process with channel-concatenated low-dose conditioning."""
    scheduler.set_timesteps(num_inference_steps, device=device)
    B, _, H, W = low_batch.shape
    x = torch.randn((B, 1, H, W), device=device)
    for t in scheduler.timesteps:
        model_input = torch.cat([x, low_batch], dim=1)
        v_pred = unet(model_input, t, return_dict=False)[0]
        x = scheduler.step(v_pred, t, x, eta=eta, return_dict=False)[0]
    return x


def reconstruct_volume(
    low_path: Path,
    cfg: SupervisedConfig,
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    device: str,
    inference_batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct one volume directly from a raw low-dose NIfTI.

    Normalization geometry (bbox, asinh scale M, kept slices) is derived from the
    low-dose volume AT RUNTIME via prepare_low_dose(), so this works on any unseen
    scan with no preprocess cache or metadata entry. Returns (count-space recon on
    the original grid, affine).
    """
    low_vol, affine, _ = load_volume(low_path)
    prep = prepare_low_dose(low_vol, cfg.data)
    bbox, low_resized = prep["bbox"], prep["low_resized"]
    M, kept_indices = prep["M"], prep["kept_indices"]
    if not kept_indices:
        raise ValueError(f"No foreground slices found in {low_path}")

    low_norm = asinh_normalize(low_resized, M, cfg.data.asinh_k).astype(np.float32)
    kept_stack = np.stack([low_norm[:, :, z] for z in kept_indices])  # (kept_Z, S, S)

    recon_chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(kept_stack), inference_batch_size), desc=f"Recon {low_path.stem}"):
        low_batch = (
            torch.from_numpy(kept_stack[i : i + inference_batch_size]).unsqueeze(1).to(device)
        )
        recon = ddim_sample_conditional(
            unet,
            scheduler,
            low_batch,
            cfg.sample.num_inference_steps,
            cfg.sample.ddim_eta,
            device,
        )
        recon_chunks.append(recon.cpu().float())

    normalized = torch.cat(recon_chunks, dim=0).squeeze(1).numpy()  # (kept_Z, S, S)
    volume = reassemble_to_original_grid(
        normalized_slices=normalized,
        kept_indices=kept_indices,
        bbox=bbox,
        original_shape=low_vol.shape,
        M=M,
        k=cfg.data.asinh_k,
    )
    return volume, affine


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct volumes with Pipeline A.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint dir containing unet/ and scheduler/.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructions/supervised"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Reconstruct only first N volumes.")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument(
        "--low-volume",
        type=Path,
        default=None,
        help="Reconstruct a single raw low-dose NIfTI directly (an unseen volume); "
        "bypasses the split and any preprocess cache.",
    )
    args = parser.parse_args()

    cfg = SupervisedConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device()
    unet = UNet2DModel.from_pretrained(args.checkpoint / "unet").to(device).eval()
    scheduler = _build_ddim_scheduler(cfg)

    # Build the (output_name, low_path) work list.
    if args.low_volume is not None:
        jobs = [(args.low_volume.name.split(".")[0], args.low_volume)]
    else:
        low_dose_dir = cfg.data.raw_dataset_dir / cfg.data.low_dose_subdir
        patient_ids = load_splits(cfg.data.splits_path)[args.split]
        if args.limit is not None:
            patient_ids = patient_ids[: args.limit]
        jobs = []
        for pid in patient_ids:
            try:
                jobs.append((pid, match_low_dose(pid, low_dose_dir, cfg.data.low_suffix_variants)))
            except FileNotFoundError as exc:
                print(f"[skip] {pid}: {exc}")

    for name, low_path in jobs:
        recon, affine = reconstruct_volume(
            low_path, cfg, unet, scheduler, device, args.inference_batch_size
        )
        save_volume(recon, affine, args.output_dir / f"{name}_recon.nii.gz")
        print(f"Saved {name}_recon.nii.gz")


if __name__ == "__main__":
    main()
