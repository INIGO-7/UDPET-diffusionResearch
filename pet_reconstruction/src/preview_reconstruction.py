"""Single-slice qualitative preview: low-dose / reconstruction / full-dose.

Picks one cached slice from a TEST-split patient, runs DDIM-50 sampling with
the chosen checkpoint, and renders a 3-panel comparison figure. Useful for
eyeballing a model without running the full evaluate.py pipeline.

Picking is restricted to the test split — at training time the model never
saw these volumes, so the comparison is honest.

Usage (from pet_reconstruction/):
    # List test patient IDs (and their kept-slice counts) and exit.
    python -m src.preview_reconstruction \
        --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
        --list

    # Render one slice (defaults to the middle kept slice).
    python -m src.preview_reconstruction \
        --pipeline supervised \
        --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
        --patient-id 01122021_1_20211201_164050 \
        --slice-idx 120 \
        --save-path previews/sup_01122021_1_s120.png
"""

import argparse
import json
from pathlib import Path

import torch
from diffusers import UNet2DModel

from .config import SupervisedConfig, UnconditionalConfig
from .reconstruct_supervised import (
    _build_ddim_scheduler as _build_sched_sup,
    ddim_sample_conditional,
)
from .reconstruct_unconditional import (
    _build_ddim_scheduler as _build_sched_unc,
    ddim_dps_sample,
)
from .splits import load_splits
from .visualize import three_panel_figure


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_slice(path: Path) -> torch.Tensor:
    return torch.load(path, weights_only=True).to(torch.float32)


def _list_test_patients(cfg) -> list[tuple[str, int]]:
    """Return [(patient_id, n_cached_slices)] for every test patient that has a cache."""
    test_ids = load_splits(cfg.data.splits_path)["test"]
    out: list[tuple[str, int]] = []
    for pid in test_ids:
        full_dir = cfg.data.cache_dir / pid / "full"
        if not full_dir.exists():
            continue
        out.append((pid, len(list(full_dir.glob("*.pt")))))
    return out


def _resolve_slice_path(slice_dir: Path, slice_idx: int) -> Path:
    """Map a slice index to the cached .pt path, accepting either filename style."""
    # Cache filenames are 4-digit zero-padded, e.g. 0123.pt.
    candidate = slice_dir / f"{slice_idx:04d}.pt"
    if candidate.exists():
        return candidate
    # Fallback: position-based index into the sorted listing.
    slices = sorted(slice_dir.glob("*.pt"))
    if not slices:
        raise FileNotFoundError(f"No cached slices in {slice_dir}")
    if not 0 <= slice_idx < len(slices):
        raise IndexError(
            f"Slice index {slice_idx} out of range; this patient has {len(slices)} cached slices "
            f"(valid indices 0..{len(slices) - 1})."
        )
    return slices[slice_idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a single low / recon / full comparison for one test slice."
    )
    parser.add_argument("--pipeline", choices=["supervised", "unconditional"], required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=False,
        help="Path to a checkpoint dir containing unet/. Required unless --list is used.",
    )
    parser.add_argument(
        "--patient-id",
        default=None,
        help="Patient ID from the test split. Use --list to see available IDs.",
    )
    parser.add_argument(
        "--slice-idx",
        type=int,
        default=None,
        help="Index into the patient's cached slices. Defaults to the middle slice.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="If given, save the figure here instead of showing it interactively.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the test-split patient IDs (with cached-slice counts) and exit.",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=None,
        help="Override DPS guidance scale (unconditional pipeline only).",
    )
    args = parser.parse_args()

    if args.pipeline == "supervised":
        cfg = SupervisedConfig()
    else:
        cfg = UnconditionalConfig()
        if args.omega is not None:
            cfg.sample.dps_omega = args.omega

    if args.list:
        rows = _list_test_patients(cfg)
        if not rows:
            print("No cached test patients found. Did you run preprocess?")
            return
        print(f"Test-split patients with cached slices ({len(rows)}):")
        for pid, n in rows:
            print(f"  {pid}   ({n} slices, valid --slice-idx 0..{n - 1})")
        return

    if args.checkpoint is None:
        parser.error("--checkpoint is required unless --list is used.")
    if args.patient_id is None:
        parser.error("--patient-id is required (use --list to see options).")

    # Restrict to the test split: refuse anything else so previews are honest.
    test_ids = set(load_splits(cfg.data.splits_path)["test"])
    if args.patient_id not in test_ids:
        raise SystemExit(
            f"Refusing to preview {args.patient_id!r}: not in the test split. "
            f"Use --list to see eligible test patients."
        )

    full_dir = cfg.data.cache_dir / args.patient_id / "full"
    low_dir = cfg.data.cache_dir / args.patient_id / "low"
    if not full_dir.exists() or not low_dir.exists():
        raise SystemExit(
            f"No cached slices for {args.patient_id}. Did you preprocess this patient?"
        )

    all_slices = sorted(full_dir.glob("*.pt"))
    slice_idx = args.slice_idx if args.slice_idx is not None else len(all_slices) // 2

    full_path = _resolve_slice_path(full_dir, slice_idx)
    low_path = low_dir / full_path.name
    if not low_path.exists():
        raise SystemExit(f"Low-dose counterpart missing for slice {full_path.name}")

    device = _pick_device()
    print(f"Loading checkpoint from {args.checkpoint} (device={device})")
    unet = UNet2DModel.from_pretrained(args.checkpoint / "unet").to(device).eval()
    scheduler = (
        _build_sched_sup(cfg) if args.pipeline == "supervised" else _build_sched_unc(cfg)
    )

    low_t = _load_slice(low_path)
    full_t = _load_slice(full_path)
    low_batch = low_t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    print(
        f"Sampling slice {full_path.stem} of {args.patient_id} "
        f"with {cfg.sample.num_inference_steps} DDIM steps..."
    )
    if args.pipeline == "supervised":
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

    recon_np = recon.cpu().float().squeeze().numpy()
    low_np = low_t.numpy()
    full_np = full_t.numpy()

    title = (
        f"{args.patient_id} — slice {full_path.stem} "
        f"({args.pipeline}, normalized space)"
    )
    three_panel_figure(
        low=low_np, recon=recon_np, full=full_np,
        save_path=args.save_path, title=title,
    )
    if args.save_path is not None:
        print(f"Saved figure to {args.save_path}")


if __name__ == "__main__":
    main()
