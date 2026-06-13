"""Single-slice reconstruction exporter for memoria comparison figures.

Given a model (pipeline + checkpoint), a patient and an axial slice index, this
runs inference on that one slice and writes THREE clean PNGs (raw pixels only,
no axes/titles/colorbar) at the original scan resolution into a fresh timestamped
folder under `reports/slice_recon/`:

    full_dose.png                  raw full-dose slice, original grid, untouched
    low_dose.png                   raw low-dose slice, original grid, untouched
    recon_<pipeline>_<ckpt>.png    model reconstruction, mapped back to the
                                   original grid (asinh^-1 -> resize -> bbox)

All three PNGs share ONE intensity window (vmin=0, vmax = 99.5th percentile of
the full-dose foreground) so they are directly comparable: the low-dose appears
genuinely dim/noisy (~1/20 counts) and a good reconstruction matches the full
dose. A `meta.json` records every parameter (patient, slice, model, M, k,
window, split membership, ...) so a figure can be traced back exactly.

Unlike `preview`, this accepts ANY patient (train/val/test) — it is a reporting
tool, not a blinded qualitative check — but the split each patient belongs to is
recorded in meta.json so you stay aware of what you are showing.

Everything (bbox, asinh scale M, kept slices, normalization) is derived at
RUNTIME from the raw low-dose NIfTI via prepare_low_dose(), exactly like
`reconstruct`, so no preprocess cache is required and the result matches the
training frame.

Usage (from pet_reconstruction/):
    python -m src.slice_recon \
        --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-099 \
        --patient-id 01122021_1_20211201_164050 \
        --slice-idx 120

    # Omit --slice-idx to use the middle kept slice; --list-slices to inspect.
    python -m src.slice_recon --pipeline cnn \
        --checkpoint checkpoints/cnn_redcnn/checkpoint-epoch-059 \
        --patient-id <pid> --list-slices
"""

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import torch
from diffusers import UNet2DModel

from .config import (
    CNNConfig,
    RegressionUNetConfig,
    SupervisedConfig,
    UnconditionalConfig,
)
from .model_cnn import load_redcnn
from .preprocess import prepare_low_dose
from .reconstruct_cnn import redcnn_batch
from .reconstruct_regression import regress_batch
from .reconstruct_supervised import (
    _build_ddim_scheduler as _build_sched_sup,
    ddim_sample_conditional,
)
from .reconstruct_unconditional import (
    _build_ddim_scheduler as _build_sched_unc,
    ddim_dps_sample,
)
from .splits import load_splits, match_low_dose
from .volume_io import asinh_normalize, load_volume, reassemble_to_original_grid


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _make_config(pipeline: str, omega: float | None):
    if pipeline == "supervised":
        return SupervisedConfig()
    if pipeline == "regression":
        return RegressionUNetConfig()
    if pipeline == "cnn":
        return CNNConfig()
    cfg = UnconditionalConfig()
    if omega is not None:
        cfg.sample.dps_omega = omega
    return cfg


def _load_model(pipeline: str, checkpoint: Path, cfg, device: str):
    """Return (model, scheduler). scheduler is None for the non-diffusion paths."""
    if pipeline == "cnn":
        return load_redcnn(cfg, checkpoint, device), None
    model = UNet2DModel.from_pretrained(checkpoint / "unet").to(device).eval()
    if pipeline == "supervised":
        return model, _build_sched_sup(cfg)
    if pipeline == "regression":
        return model, None
    return model, _build_sched_unc(cfg)


def _which_split(patient_id: str, splits_path: Path) -> str:
    """Report which split a patient belongs to (or 'unknown' if absent)."""
    try:
        splits = load_splits(splits_path)
    except FileNotFoundError:
        return "unknown"
    for name, ids in splits.items():
        if patient_id in ids:
            return name
    return "unknown"


def _reconstruct_norm_slice(pipeline, model, scheduler, cfg, low_norm_slice, device) -> np.ndarray:
    """Run the model on one normalized low-dose slice -> normalized recon (S, S)."""
    low_batch = torch.from_numpy(low_norm_slice).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,S,S)
    if pipeline == "supervised":
        recon = ddim_sample_conditional(
            model, scheduler, low_batch,
            cfg.sample.num_inference_steps, cfg.sample.ddim_eta, device,
        )
    elif pipeline == "regression":
        with torch.no_grad():
            recon = regress_batch(model, low_batch)
    elif pipeline == "cnn":
        recon = redcnn_batch(model, low_batch)
    else:  # unconditional + DPS
        recon = ddim_dps_sample(
            model, scheduler, low_batch,
            cfg.sample.num_inference_steps, cfg.sample.ddim_eta,
            cfg.sample.dps_omega, device,
        )
    return recon.cpu().float().squeeze().numpy()  # (S, S)


def _save_png(array2d: np.ndarray, path: Path, vmin: float, vmax: float, cmap: str) -> None:
    """Write a 2D array as a clean PNG at exactly one pixel per array element."""
    mpimg.imsave(str(path), array2d, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export full / low / reconstruction PNGs for one slice (memoria figures)."
    )
    parser.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="Checkpoint dir (with unet/ or model.pt). Required unless --list-slices.",
    )
    parser.add_argument(
        "--patient-id",
        required=True,
        help="Any patient ID present in the raw dataset (train/val/test).",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=None,
        help="Original-grid axial z index. Must be a kept (foreground) slice. "
        "Defaults to the middle kept slice. Use --list-slices to inspect.",
    )
    parser.add_argument(
        "--list-slices",
        action="store_true",
        help="Print the kept (reconstructable) slice indices for this patient and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/slice_recon"),
        help="Base dir; a fresh <HHMM>_<DD-MM-YYYY>_<uuid8> subfolder is created inside.",
    )
    parser.add_argument(
        "--cmap", default="gray", help="Matplotlib colormap for the PNGs (default: gray)."
    )
    parser.add_argument(
        "--rot90",
        type=int,
        default=0,
        help="Rotate every panel 90*N degrees counter-clockwise (display orientation only).",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Override DPS guidance scale (unconditional pipeline only).",
    )
    args = parser.parse_args()

    cfg = _make_config(args.pipeline, args.omega)

    # Resolve the raw NIfTI pair for this patient (full-dose ground truth + low-dose input).
    full_dose_dir = cfg.data.raw_dataset_dir / cfg.data.full_dose_subdir
    low_dose_dir = cfg.data.raw_dataset_dir / cfg.data.low_dose_subdir
    full_path = full_dose_dir / f"{args.patient_id}{cfg.data.full_suffix}"
    if not full_path.exists():
        raise SystemExit(f"No full-dose NIfTI for patient {args.patient_id!r} at {full_path}")
    try:
        low_path = match_low_dose(args.patient_id, low_dose_dir, cfg.data.low_suffix_variants)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    # Runtime normalization geometry from the low dose alone (no cache, no leakage).
    low_vol, _, _ = load_volume(low_path)
    prep = prepare_low_dose(low_vol, cfg.data)
    bbox, low_resized = prep["bbox"], prep["low_resized"]
    M, kept_indices = prep["M"], prep["kept_indices"]
    if not kept_indices:
        raise SystemExit(f"No foreground (reconstructable) slices in {low_path}")

    if args.list_slices:
        print(
            f"{args.patient_id}: {len(kept_indices)} kept slices, "
            f"z in [{kept_indices[0]}..{kept_indices[-1]}] "
            f"(split={_which_split(args.patient_id, cfg.data.splits_path)})"
        )
        print("  valid --slice-idx values:", ", ".join(str(z) for z in kept_indices))
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --list-slices is used.")

    z = args.slice_idx if args.slice_idx is not None else kept_indices[len(kept_indices) // 2]
    if z not in kept_indices:
        raise SystemExit(
            f"Slice {z} is not a kept (foreground) slice for {args.patient_id}. "
            f"Valid z in [{kept_indices[0]}..{kept_indices[-1]}]; run --list-slices for the full set."
        )

    # Raw original-resolution slices, untouched, for the two ground-truth panels.
    full_vol, _, _ = load_volume(full_path)
    if full_vol.shape != low_vol.shape:
        raise SystemExit(
            f"Paired shape mismatch for {args.patient_id}: full {full_vol.shape} vs low {low_vol.shape}"
        )
    full_slice = full_vol[:, :, z]
    low_slice = low_vol[:, :, z]

    # Reconstruct just the requested slice, then invert back onto the original grid.
    device = _pick_device()
    print(f"Loading {args.pipeline} checkpoint {args.checkpoint} (device={device})")
    model, scheduler = _load_model(args.pipeline, args.checkpoint, cfg, device)

    low_norm = asinh_normalize(low_resized[:, :, z], M, cfg.data.asinh_k).astype(np.float32)
    print(f"Reconstructing slice z={z} of {args.patient_id} ({args.pipeline})...")
    recon_norm = _reconstruct_norm_slice(args.pipeline, model, scheduler, cfg, low_norm, device)
    recon_slice = reassemble_to_original_grid(
        normalized_slices=recon_norm[None],  # (1, S, S)
        kept_indices=[z],
        bbox=bbox,
        original_shape=low_vol.shape,
        M=M,
        k=cfg.data.asinh_k,
    )[:, :, z]  # (H, W), count space, zeros outside bbox

    # One shared display window from the full-dose foreground, applied to all three.
    fg = full_slice[full_slice > 0]
    vmin = 0.0
    vmax = float(np.percentile(fg, cfg.data.norm_percentile)) if fg.size else float(full_slice.max() or 1.0)

    if args.rot90:
        full_slice = np.rot90(full_slice, args.rot90)
        low_slice = np.rot90(low_slice, args.rot90)
        recon_slice = np.rot90(recon_slice, args.rot90)

    # Output folder: all runs of the same patient+slice share a parent, and each
    # model run gets its own <model>_<uuid> subfolder inside it.
    ckpt_tag = args.checkpoint.name
    model_tag = f"{args.pipeline}_{ckpt_tag}"
    folder = args.output_dir / f"{args.patient_id}_{z}" / f"{model_tag}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()

    recon_name = f"recon_{args.pipeline}_{ckpt_tag}.png"
    _save_png(full_slice, folder / "full_dose.png", vmin, vmax, args.cmap)
    _save_png(low_slice, folder / "low_dose.png", vmin, vmax, args.cmap)
    _save_png(recon_slice, folder / recon_name, vmin, vmax, args.cmap)

    meta = {
        "patient_id": args.patient_id,
        "split": _which_split(args.patient_id, cfg.data.splits_path),
        "slice_idx": int(z),
        "pipeline": args.pipeline,
        "checkpoint": str(args.checkpoint),
        "omega": cfg.sample.dps_omega if args.pipeline == "unconditional" else None,
        "M": float(M),
        "k": float(cfg.data.asinh_k),
        "image_size": int(cfg.data.image_size),
        "bbox": [(int(s.start), int(s.stop)) for s in bbox],
        "original_shape": [int(s) for s in low_vol.shape],
        "window": {"vmin": vmin, "vmax": vmax, "source": "full_dose p99.5"},
        "cmap": args.cmap,
        "rot90": int(args.rot90),
        "files": {"full": "full_dose.png", "low": "low_dose.png", "recon": recon_name},
        "timestamp": now.isoformat(timespec="seconds"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved 3 PNGs + meta.json to {folder}")


if __name__ == "__main__":
    main()
