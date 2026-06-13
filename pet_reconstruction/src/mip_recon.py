"""Maximum-Intensity-Projection (MIP) exporter for memoria figures.

Companion to `slice_recon`, but instead of a single axial slice this projects the
WHOLE volume onto one anatomical plane (coronal / sagittal / axial) with a
maximum-intensity projection — the canonical whole-body PET reading view. It
writes THREE clean PNGs (raw pixels, no axes/colorbar):

    full_dose_<plane>.png            MIP of the raw full-dose volume
    low_dose_<plane>.png             MIP of the raw low-dose volume
    recon_<pipeline>_<ckpt>_<plane>.png   MIP of the model's reconstructed volume

into  reports/MIP_recon/<patient_id>/<pipeline>_<ckpt>_<uuid8>/  plus a meta.json.

The reconstruction is run at RUNTIME over the whole low-dose volume (same
prepare_low_dose geometry as `reconstruct`), so it needs no preprocess cache and
works for any patient and any of the four pipelines. NOTE: for the diffusion
pipelines (supervised / unconditional) this samples every kept slice with
DDIM-50, so a full volume is slow; the cnn / regression baselines are a single
forward pass and fast.

All three MIPs share ONE intensity window derived from the full-dose MIP
(vmin=0, vmax = 99.5th percentile of its foreground), so they are directly
comparable. The projection axis is chosen from the NIfTI affine (aff2axcodes),
and the 2D image is oriented Superior-up (coronal/sagittal) or Anterior-up
(axial). Use --rot90/--flipud/--fliplr to fine-tune display orientation.

Usage (from pet_reconstruction/):
    python -m src.mip_recon \
        --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-099 \
        --patient-id 01122021_1_20211201_164050 \
        --plane coronal
"""

import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm.auto import tqdm

from .preprocess import prepare_low_dose
from .reconstruct_supervised import ddim_sample_conditional
from .reconstruct_unconditional import ddim_dps_sample
from .reconstruct_regression import regress_batch
from .reconstruct_cnn import redcnn_batch
from .slice_recon import (
    _load_model,
    _make_config,
    _pick_device,
    _save_png,
    _which_split,
)
from .splits import match_low_dose
from .volume_io import asinh_normalize, load_volume, reassemble_to_original_grid


# Which array axis to project for each plane is decided from the affine, but the
# anatomical convention (what ends up vertical, and which direction is "up") is fixed:
_PLANE_PROJECT = {"coronal": "AP", "sagittal": "LR", "axial": "SI"}
_PLANE_VERTICAL = {"coronal": ("SI", "S"), "sagittal": ("SI", "S"), "axial": ("AP", "A")}


def _reconstruct_volume_runtime(
    pipeline: str, model, scheduler, cfg, low_path: Path, device: str, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a whole volume in original-grid count space, model-agnostic.

    Mirrors reconstruct_{supervised,unconditional}.reconstruct_volume but branches
    on the pipeline so the cnn/regression baselines also run purely from the raw
    low-dose NIfTI (no cache). Returns (count-space volume, affine).
    """
    low_vol, affine, _ = load_volume(low_path)
    prep = prepare_low_dose(low_vol, cfg.data)
    bbox, low_resized = prep["bbox"], prep["low_resized"]
    M, kept_indices = prep["M"], prep["kept_indices"]
    if not kept_indices:
        raise SystemExit(f"No foreground slices found in {low_path}")

    low_norm = asinh_normalize(low_resized, M, cfg.data.asinh_k).astype(np.float32)
    kept_stack = np.stack([low_norm[:, :, z] for z in kept_indices])  # (kept_Z, S, S)

    recon_chunks: list[torch.Tensor] = []
    for i in tqdm(range(0, len(kept_stack), batch_size), desc=f"Recon {low_path.stem}"):
        low_batch = torch.from_numpy(kept_stack[i : i + batch_size]).unsqueeze(1).to(device)
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


def _anatomical_axes(affine: np.ndarray) -> dict[str, tuple[int, str]]:
    """Map 'LR'/'AP'/'SI' -> (array axis, axis code) using the NIfTI affine."""
    axcodes = nib.orientations.aff2axcodes(affine)
    out: dict[str, tuple[int, str]] = {}
    for ax, code in enumerate(axcodes):
        if code in ("L", "R"):
            out["LR"] = (ax, code)
        elif code in ("A", "P"):
            out["AP"] = (ax, code)
        elif code in ("S", "I"):
            out["SI"] = (ax, code)
    return out


def _oriented_mip(volume: np.ndarray, affine: np.ndarray, plane: str) -> np.ndarray:
    """Maximum-intensity projection onto `plane`, oriented for radiological display.

    Projection axis is taken from the affine; the result is arranged so the
    Superior (coronal/sagittal) or Anterior (axial) direction points up.
    """
    anat = _anatomical_axes(affine)
    proj_axis = anat[_PLANE_PROJECT[plane]][0]
    mip = volume.max(axis=proj_axis)  # 2D over the two remaining axes (ascending order)

    remaining = [a for a in range(3) if a != proj_axis]
    vert_label, vert_pos = _PLANE_VERTICAL[plane]
    vert_axis = anat[vert_label][0]
    # Move the vertical anatomical axis to rows(0); the other becomes columns.
    mip = np.moveaxis(mip, remaining.index(vert_axis), 0)
    # Row index increases along the axis code; if that code already points to the
    # "up" direction, the up end sits at the bottom -> flip so it sits at the top.
    if anat[vert_label][1] == vert_pos:
        mip = np.flipud(mip)
    return np.ascontiguousarray(mip)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export full / low / recon MIP PNGs over one plane (memoria figures)."
    )
    parser.add_argument(
        "--pipeline",
        choices=["supervised", "unconditional", "regression", "cnn"],
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint dir (with unet/ or model.pt).",
    )
    parser.add_argument(
        "--patient-id",
        required=True,
        help="Any patient ID present in the raw dataset (train/val/test).",
    )
    parser.add_argument(
        "--plane",
        choices=["coronal", "sagittal", "axial"],
        required=True,
        help="Anatomical plane to project the MIP onto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/MIP_recon"),
        help="Base dir; a <pipeline>_<ckpt>_<uuid8> subfolder is created under <patient_id>/.",
    )
    parser.add_argument("--cmap", default="gray", help="Matplotlib colormap (default: gray).")
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument(
        "--rot90", type=int, default=0, help="Extra 90*N CCW rotation of every panel (display)."
    )
    parser.add_argument("--flipud", action="store_true", help="Flip every panel vertically (display).")
    parser.add_argument("--fliplr", action="store_true", help="Flip every panel horizontally (display).")
    parser.add_argument(
        "--omega", type=float, default=None, help="Override DPS guidance scale (unconditional only)."
    )
    args = parser.parse_args()

    cfg = _make_config(args.pipeline, args.omega)

    # Resolve the raw NIfTI pair (full-dose ground truth + low-dose input).
    full_dose_dir = cfg.data.raw_dataset_dir / cfg.data.full_dose_subdir
    low_dose_dir = cfg.data.raw_dataset_dir / cfg.data.low_dose_subdir
    full_path = full_dose_dir / f"{args.patient_id}{cfg.data.full_suffix}"
    if not full_path.exists():
        raise SystemExit(f"No full-dose NIfTI for patient {args.patient_id!r} at {full_path}")
    try:
        low_path = match_low_dose(args.patient_id, low_dose_dir, cfg.data.low_suffix_variants)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    device = _pick_device()
    print(f"Loading {args.pipeline} checkpoint {args.checkpoint} (device={device})")
    model, scheduler = _load_model(args.pipeline, args.checkpoint, cfg, device)

    print(f"Reconstructing whole volume for {args.patient_id} ({args.pipeline})...")
    recon_vol, affine = _reconstruct_volume_runtime(
        args.pipeline, model, scheduler, cfg, low_path, device, args.inference_batch_size
    )
    full_vol, full_affine, _ = load_volume(full_path)
    low_vol, _, _ = load_volume(low_path)

    def _mip(vol: np.ndarray) -> np.ndarray:
        m = _oriented_mip(vol, full_affine, args.plane)
        if args.rot90:
            m = np.rot90(m, args.rot90)
        if args.flipud:
            m = np.flipud(m)
        if args.fliplr:
            m = np.fliplr(m)
        return np.ascontiguousarray(m)

    full_mip = _mip(full_vol)
    low_mip = _mip(low_vol)
    recon_mip = _mip(recon_vol)

    # One shared window from the full-dose MIP foreground, applied to all three.
    fg = full_mip[full_mip > 0]
    vmin = 0.0
    vmax = float(np.percentile(fg, cfg.data.norm_percentile)) if fg.size else float(full_mip.max() or 1.0)

    # Output folder: per-patient parent, per-model-run subfolder.
    ckpt_tag = args.checkpoint.name
    model_tag = f"{args.pipeline}_{ckpt_tag}"
    folder = args.output_dir / args.patient_id / f"{model_tag}_{uuid.uuid4().hex[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    now = datetime.now()

    recon_name = f"recon_{args.pipeline}_{ckpt_tag}_{args.plane}.png"
    _save_png(full_mip, folder / f"full_dose_{args.plane}.png", vmin, vmax, args.cmap)
    _save_png(low_mip, folder / f"low_dose_{args.plane}.png", vmin, vmax, args.cmap)
    _save_png(recon_mip, folder / recon_name, vmin, vmax, args.cmap)

    meta = {
        "patient_id": args.patient_id,
        "split": _which_split(args.patient_id, cfg.data.splits_path),
        "plane": args.plane,
        "projection": "max-intensity, whole volume",
        "pipeline": args.pipeline,
        "checkpoint": str(args.checkpoint),
        "omega": cfg.sample.dps_omega if args.pipeline == "unconditional" else None,
        "axcodes": "".join(nib.orientations.aff2axcodes(full_affine)),
        "original_shape": [int(s) for s in low_vol.shape],
        "window": {"vmin": vmin, "vmax": vmax, "source": f"full_dose {args.plane} MIP p99.5"},
        "cmap": args.cmap,
        "display": {"rot90": int(args.rot90), "flipud": bool(args.flipud), "fliplr": bool(args.fliplr)},
        "files": {
            "full": f"full_dose_{args.plane}.png",
            "low": f"low_dose_{args.plane}.png",
            "recon": recon_name,
        },
        "timestamp": now.isoformat(timespec="seconds"),
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Saved 3 {args.plane} MIP PNGs + meta.json to {folder}")


if __name__ == "__main__":
    main()
