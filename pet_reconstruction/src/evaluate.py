"""End-to-end evaluation of a trained checkpoint on a held-out split.

For each patient in the requested split:
    1. Load the cached normalized full-dose slices (ground truth).
    2. Load the cached normalized low-dose slices (conditioning / DPS target).
    3. Run DDIM-50 sampling — channel-concat for Pipeline A, DPS-guided for B.
    4. Compute per-slice PSNR / SSIM / NRMSE (whole + foreground) in NORMALIZED
       space (the space the model was trained in).
    5. Invert the asinh and compute intensity-preservation in ORIGINAL COUNT space.
    6. Save a 4-panel figure for a handful of representative patients.

Outputs under `--output-dir`:
    per_slice.csv      one row per (patient, slice) with every metric
    per_volume.csv     one row per patient, mean across slices
    summary.json       dataset-wide aggregate, plus the run config
    figures/{pid}_slice{NNNN}.png

Usage (from pet_reconstruction/):
    python -m src.evaluate --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
        --output-dir evaluations/supervised/
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import UNet2DModel
from tqdm.auto import tqdm

from .config import SupervisedConfig, UnconditionalConfig
from .metrics import (
    aggregate_volume_metrics,
    intensity_preservation,
    slice_metrics,
)
from .reconstruct_supervised import (
    _build_ddim_scheduler as _build_sched_sup,
    ddim_sample_conditional,
)
from .reconstruct_unconditional import (
    _build_ddim_scheduler as _build_sched_unc,
    ddim_dps_sample,
)
from .splits import load_splits
from .visualize import four_panel_figure
from .volume_io import asinh_denormalize


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_slice(path: Path) -> torch.Tensor:
    """Load a cached fp16 slice as a CPU fp32 2D tensor."""
    return torch.load(path, weights_only=True).to(torch.float32)


def evaluate_patient(
    patient_id: str,
    cfg,
    unet: UNet2DModel,
    scheduler,
    metadata: dict,
    device: str,
    pipeline: str,
    inference_batch_size: int = 4,
    figure_slice_idx: int | None = None,
    figure_dir: Path | None = None,
) -> list[dict]:
    """Run inference + per-slice metrics for one patient.

    If `figure_slice_idx` is provided AND `figure_dir` is set, also saves a
    4-panel figure for that single slice index (counted within the patient's
    kept slices, 0-based).
    """
    full_paths = sorted((cfg.data.cache_dir / patient_id / "full").glob("*.pt"))
    low_paths = sorted((cfg.data.cache_dir / patient_id / "low").glob("*.pt"))
    assert len(full_paths) == len(low_paths), (
        f"Mismatched cache for {patient_id}: {len(full_paths)} full vs {len(low_paths)} low"
    )

    M = metadata[patient_id]["M"]
    k = metadata[patient_id]["k"]

    per_slice: list[dict] = []

    for i in tqdm(
        range(0, len(full_paths), inference_batch_size),
        desc=f"Eval {patient_id}",
        leave=False,
    ):
        chunk_full = full_paths[i : i + inference_batch_size]
        chunk_low = low_paths[i : i + inference_batch_size]

        low_batch = (
            torch.stack([_load_slice(p) for p in chunk_low]).unsqueeze(1).to(device)
        )
        full_batch_cpu = torch.stack([_load_slice(p) for p in chunk_full])  # (B, H, W) on CPU

        if pipeline == "supervised":
            recon = ddim_sample_conditional(
                unet, scheduler, low_batch,
                cfg.sample.num_inference_steps, cfg.sample.ddim_eta, device,
            )
        else:
            recon = ddim_dps_sample(
                unet, scheduler, low_batch,
                cfg.sample.num_inference_steps, cfg.sample.ddim_eta,
                cfg.sample.dps_omega, device,
            )

        recon_np = recon.cpu().float().squeeze(1).numpy()  # (B, H, W) normalized
        full_np = full_batch_cpu.numpy()
        low_np = low_batch.cpu().float().squeeze(1).numpy()

        for j in range(recon_np.shape[0]):
            slice_local_idx = i + j
            r, f, l = recon_np[j], full_np[j], low_np[j]

            # Foreground mask derived from the ground-truth full-dose in COUNT space.
            f_counts = asinh_denormalize(f, M, k)
            fg_mask = f_counts > cfg.data.foreground_threshold

            # Image-quality metrics in NORMALIZED space, whole + foreground.
            metrics = slice_metrics(r, f, foreground_mask=fg_mask)

            # Intensity preservation in ORIGINAL count space (PET-flavoured proxy for SUV).
            r_counts = asinh_denormalize(r, M, k)
            metrics.update(intensity_preservation(r_counts, f_counts, fg_mask))

            metrics["patient_id"] = patient_id
            metrics["slice_local_idx"] = slice_local_idx
            per_slice.append(metrics)

            # Snapshot the figure-worthy slice (if requested)
            if (
                figure_slice_idx is not None
                and figure_dir is not None
                and slice_local_idx == figure_slice_idx
            ):
                figure_path = figure_dir / f"{patient_id}_slice{slice_local_idx:04d}.png"
                four_panel_figure(
                    low=l,
                    full=f,
                    recon=r,
                    save_path=figure_path,
                    title=f"{patient_id} — slice {slice_local_idx} (normalized space)",
                )

    return per_slice


# Keys that look numeric but are identifiers / bookkeeping, not metrics.
_NON_METRIC_KEYS = {"slice_local_idx", "n_slices"}


def _metrics_only(row: dict) -> dict:
    """Strip identifier and bookkeeping fields, keep only numeric metrics."""
    return {
        k: v for k, v in row.items()
        if isinstance(v, (int, float)) and k not in _NON_METRIC_KEYS
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a held-out split.")
    parser.add_argument("--pipeline", choices=["supervised", "unconditional"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None, help="Evaluate first N patients only.")
    parser.add_argument(
        "--n-figures",
        type=int,
        default=5,
        help="How many evenly-spaced patients in the split get a 4-panel figure.",
    )
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Override DPS guidance scale (Pipeline B only).",
    )
    args = parser.parse_args()

    # ----- Build config / scheduler / model -----
    if args.pipeline == "supervised":
        cfg = SupervisedConfig()
        scheduler = _build_sched_sup(cfg)
    else:
        cfg = UnconditionalConfig()
        if args.omega is not None:
            cfg.sample.dps_omega = args.omega
        scheduler = _build_sched_unc(cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    device = _pick_device()
    print(f"Loading checkpoint from {args.checkpoint} (device={device})")
    unet = UNet2DModel.from_pretrained(args.checkpoint / "unet").to(device).eval()

    metadata = json.loads((cfg.data.cache_dir / "metadata.json").read_text())
    patient_ids = load_splits(cfg.data.splits_path)[args.split]
    if args.limit is not None:
        patient_ids = patient_ids[: args.limit]
    print(f"Evaluating {len(patient_ids)} patients on split={args.split}")

    # Pick evenly-spaced patients for figures
    figure_patients: set[str] = set()
    if args.n_figures > 0 and patient_ids:
        idx = np.linspace(0, len(patient_ids) - 1, num=min(args.n_figures, len(patient_ids)), dtype=int)
        figure_patients = {patient_ids[i] for i in idx}

    # ----- Run -----
    all_slice_rows: list[dict] = []
    per_volume_rows: list[dict] = []

    for pid in patient_ids:
        if pid not in metadata:
            print(f"[skip] {pid}: not in cache metadata")
            continue

        # Use the middle kept slice for the figure
        n_kept = len(metadata[pid]["kept_indices"])
        fig_slice = n_kept // 2 if pid in figure_patients else None

        slice_rows = evaluate_patient(
            pid, cfg, unet, scheduler, metadata, device,
            pipeline=args.pipeline,
            inference_batch_size=args.inference_batch_size,
            figure_slice_idx=fig_slice,
            figure_dir=figure_dir if pid in figure_patients else None,
        )
        all_slice_rows.extend(slice_rows)

        # Aggregate this volume
        vol_agg = aggregate_volume_metrics([_metrics_only(r) for r in slice_rows])
        vol_agg["patient_id"] = pid
        vol_agg["n_slices"] = len(slice_rows)
        per_volume_rows.append(vol_agg)
        print(f"  {pid}: {len(slice_rows)} slices  |  " + "  ".join(
            f"{k}={v:.3f}" for k, v in vol_agg.items()
            if k in {"psnr_fg", "ssim_fg", "nrmse_fg", "mean_pct_err"}
        ))

    # ----- Aggregate + write outputs -----
    _write_csv(all_slice_rows, args.output_dir / "per_slice.csv")
    _write_csv(per_volume_rows, args.output_dir / "per_volume.csv")

    aggregate = aggregate_volume_metrics([_metrics_only(r) for r in per_volume_rows])
    summary = {
        "pipeline": args.pipeline,
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_patients": len(per_volume_rows),
        "n_slices": len(all_slice_rows),
        "config": {
            "image_size": cfg.data.image_size,
            "num_inference_steps": cfg.sample.num_inference_steps,
            "ddim_eta": cfg.sample.ddim_eta,
            "dps_omega": cfg.sample.dps_omega if args.pipeline == "unconditional" else None,
        },
        "aggregate_metrics": aggregate,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # ----- Headline print -----
    print("\n=== Headline (mean across volumes) ===")
    headline_order = [
        "psnr_whole", "ssim_whole", "nrmse_whole",
        "psnr_fg", "ssim_fg", "nrmse_fg",
        "mean_pct_err", "max_pct_err",
    ]
    for k in headline_order:
        if k in aggregate:
            print(f"  {k:14s}: {aggregate[k]:8.4f}")
    print(f"\nWrote per-slice, per-volume, summary, and figures to {args.output_dir}")


if __name__ == "__main__":
    main()
