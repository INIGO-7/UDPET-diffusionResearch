"""Baseline B inference: reconstruct test volumes with the RED-CNN.

Same bookkeeping as `reconstruct_regression` (cached low-dose slices in,
count-space NIfTI out, original affine preserved) — only the model differs.

Usage (from pet_reconstruction/):
    python -m src.reconstruct_cnn \
        --checkpoint checkpoints/cnn_redcnn/checkpoint-epoch-099
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import CNNConfig
from .model_cnn import REDCNN, load_redcnn
from .splits import load_splits
from .volume_io import reassemble_to_original_grid, save_volume


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def redcnn_batch(model: REDCNN, low_batch: torch.Tensor) -> torch.Tensor:
    """Single forward pass: low-dose (B, 1, H, W) -> full-dose estimate (B, 1, H, W)."""
    return model(low_batch)


def reconstruct_patient(
    patient_id: str,
    cfg: CNNConfig,
    model: REDCNN,
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
        recon = redcnn_batch(model, low_batch)
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
    parser = argparse.ArgumentParser(description="Reconstruct test volumes with Baseline B (RED-CNN).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint dir containing model.pt.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reconstructions/cnn_redcnn"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Reconstruct only first N volumes.")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    args = parser.parse_args()

    cfg = CNNConfig()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device()
    model = load_redcnn(cfg, args.checkpoint, device)

    metadata = json.loads((cfg.data.cache_dir / "metadata.json").read_text())
    patient_ids = load_splits(cfg.data.splits_path)[args.split]
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]

    for pid in patient_ids:
        if pid not in metadata:
            print(f"[skip] {pid}: not in preprocess metadata")
            continue
        recon = reconstruct_patient(
            pid, cfg, model, metadata, device, args.inference_batch_size
        )
        affine = np.array(metadata[pid]["affine"])
        save_volume(recon, affine, args.output_dir / f"{pid}_recon.nii.gz")
        print(f"Saved {pid}_recon.nii.gz")


if __name__ == "__main__":
    main()
