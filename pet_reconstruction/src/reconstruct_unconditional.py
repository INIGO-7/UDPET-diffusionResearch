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

Usage (from pet_reconstruction/):
    python -m src.reconstruct_unconditional \
        --checkpoint checkpoints/unconditional/checkpoint-epoch-029
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DModel
from tqdm.auto import tqdm

from .config import UnconditionalConfig
from .splits import load_splits
from .volume_io import reassemble_to_original_grid, save_volume


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


def reconstruct_patient(
    patient_id: str,
    cfg: UnconditionalConfig,
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
        y_batch = torch.stack(
            [torch.load(p, weights_only=True).to(torch.float32) for p in chunk_paths]
        ).unsqueeze(1).to(device)
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
    parser = argparse.ArgumentParser(
        description="Reconstruct test volumes with Pipeline B (unconditional + DPS)."
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
