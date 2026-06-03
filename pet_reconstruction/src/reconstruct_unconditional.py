"""Pipeline B inference: unconditional prior + DPS guidance (identity operator).

At each DDIM step:
    1. UNet predicts v_t from x_t (no conditioning).
    2. Tweedie estimate of x_0:  x̂_0 = α_t x_t − σ_t v̂_t.
    3. DPS gradient step toward y (identity operator, uniform σ in the asinh
       normalized space):
           r = x̂_0 − y
           x_t ← x_t − ω · ∇_{x_t} ‖r‖²,
       with ω/‖r‖ as the effective per-step scale (Chung et al 2023).
    4. Standard DDIM step on the corrected x_t with the original v̂_t.

Like Pipeline A, normalization geometry (bbox / asinh M / kept slices) is derived
from the raw low-dose NIfTI AT RUNTIME (prepare_low_dose) -- no preprocess cache
or metadata needed, so an unseen low-dose volume works via --low-volume.

Usage (from pet_reconstruction/):
    python -m src.reconstruct_unconditional \
        --checkpoint checkpoints/unconditional/checkpoint-epoch-029
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DModel
from tqdm.auto import tqdm

from .config import UnconditionalConfig
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


def _build_ddim_scheduler(cfg: UnconditionalConfig) -> DDIMScheduler:
    return DDIMScheduler(
        num_train_timesteps=cfg.train.num_train_timesteps,
        beta_schedule=cfg.train.beta_schedule,
        prediction_type=cfg.train.prediction_type,
    )


def ddim_dps_sample(
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    y: torch.Tensor,  # (B, 1, H, W), normalized low-dose
    num_inference_steps: int,
    eta: float,
    omega: float,
    device: str,
) -> torch.Tensor:
    """DDIM reverse process guided by DPS measurement consistency."""
    scheduler.set_timesteps(num_inference_steps, device=device)
    alphas = scheduler.alphas_cumprod.to(device)  # (T,)

    B, _, H, W = y.shape
    x = torch.randn((B, 1, H, W), device=device)

    for t in scheduler.timesteps:
        # Compute the gradient through the UNet -> Tweedie -> ‖x̂_0 − y‖² path.
        x = x.detach().requires_grad_(True)
        v_pred = unet(x, t, return_dict=False)[0]

        # Tweedie x_0 for the v-parameterization: x̂_0 = α x_t − σ v̂_t
        alpha_t = alphas[t].sqrt()
        sigma_t = (1.0 - alphas[t]).sqrt()
        x0_hat = alpha_t * x - sigma_t * v_pred

        residual = x0_hat - y
        loss = (residual * residual).sum()
        grad = torch.autograd.grad(loss, x)[0]
        # Per-step scale: ω / ‖residual‖   (Chung et al 2023, eq. 13)
        residual_norm = torch.linalg.vector_norm(residual.detach()) + 1e-8

        with torch.no_grad():
            x_corrected = x.detach() - (omega / residual_norm) * grad
            x = scheduler.step(
                v_pred.detach(), t, x_corrected, eta=eta, return_dict=False
            )[0]

    return x.detach()


def reconstruct_volume(
    low_path: Path,
    cfg: UnconditionalConfig,
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
        y_batch = (
            torch.from_numpy(kept_stack[i : i + inference_batch_size]).unsqueeze(1).to(device)
        )
        recon = ddim_dps_sample(
            unet,
            scheduler,
            y_batch,
            cfg.sample.num_inference_steps,
            cfg.sample.ddim_eta,
            cfg.sample.dps_omega,
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
    parser = argparse.ArgumentParser(
        description="Reconstruct volumes with Pipeline B (unconditional + DPS)."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint dir containing unet/ and scheduler/.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructions/unconditional"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument(
        "--low-volume",
        type=Path,
        default=None,
        help="Reconstruct a single raw low-dose NIfTI directly (an unseen volume); "
        "bypasses the split and any preprocess cache.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Override the DPS guidance scale (default: SampleConfig.dps_omega).",
    )
    args = parser.parse_args()

    cfg = UnconditionalConfig()
    if args.omega is not None:
        cfg.sample.dps_omega = args.omega
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
