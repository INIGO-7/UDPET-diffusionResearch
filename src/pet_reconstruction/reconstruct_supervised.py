"""Pipeline A inference: reconstruct test volumes with the supervised checkpoint.

For each patient in the requested split:
    - load every cached low-dose slice
    - run DDIM-50 sampling with the low-dose slice as the second input channel
    - assemble a (image_size, image_size, kept_Z) normalized recon
    - invert (asinh -> resize -> bbox) back to the original-grid count volume
    - save as a NIfTI with the original affine

Usage:
    python -m src.pet_reconstruction.reconstruct_supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DModel
from tqdm.auto import tqdm

from .config import SupervisedConfig
from .splits import load_splits
from .volume_io import reassemble_to_original_grid, save_volume


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


def reconstruct_patient(
    patient_id: str,
    cfg: SupervisedConfig,
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    metadata: dict,
    device: str,
    inference_batch_size: int = 4,
) -> np.ndarray:
    """Reconstruct one full patient volume in original-grid count space."""
    low_dir = cfg.data.cache_dir / patient_id / "low"
    slice_paths = sorted(low_dir.glob("*.pt"))
    if not slice_paths:
        raise FileNotFoundError(f"No cached low-dose slices for patient {patient_id}")

    recon_chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(slice_paths), inference_batch_size), desc=f"Recon {patient_id}"):
        chunk_paths = slice_paths[i : i + inference_batch_size]
        low_batch = torch.stack(
            [torch.load(p, weights_only=True).to(torch.float32) for p in chunk_paths]
        ).unsqueeze(1).to(device)
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

    pmeta = metadata[patient_id]
    bbox = tuple(slice(s, e) for s, e in pmeta["bbox"])
    return reassemble_to_original_grid(
        normalized_slices=normalized,
        kept_indices=pmeta["kept_indices"],
        bbox=bbox,
        original_shape=tuple(pmeta["original_shape"]),
        M=pmeta["M"],
        k=pmeta["k"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct test volumes with Pipeline A.")
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
    args = parser.parse_args()

    cfg = SupervisedConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device()
    unet = UNet2DModel.from_pretrained(args.checkpoint / "unet").to(device).eval()
    scheduler = _build_ddim_scheduler(cfg)

    metadata = json.loads((cfg.data.cache_dir / "metadata.json").read_text())
    patient_ids = load_splits(cfg.data.splits_path)[args.split]
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]

    for pid in patient_ids:
        if pid not in metadata:
            print(f"[skip] {pid}: not in preprocess metadata")
            continue
        recon = reconstruct_patient(
            pid, cfg, unet, scheduler, metadata, device, args.inference_batch_size
        )
        affine = np.array(metadata[pid]["affine"])
        save_volume(recon, affine, args.output_dir / f"{pid}_recon.nii.gz")
        print(f"Saved {pid}_recon.nii.gz")


if __name__ == "__main__":
    main()
