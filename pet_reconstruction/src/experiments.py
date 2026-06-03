"""Thesis-supporting analysis experiments.

This is the home for the small, *read-only* studies that justify design choices
in the thesis but are NOT part of the train/reconstruct/evaluate pipeline — e.g.
"what would change if we picked a different normalization percentile?". They run
on the raw NIfTI dataset (no training, no checkpoint) and emit a printed summary
plus figures, so they're cheap to run and re-run while writing.

Add a new experiment by writing a `def exp_<name>(args) -> None` function and
registering it in EXPERIMENTS. Each becomes a subcommand:

    python -m src.experiments <name> [flags]

Experiments:
    norm-percentile   How much real foreground signal gets pushed above +1 as the
                      asinh normalization percentile (M) is lowered. Motivates the
                      norm_percentile=99.5 default in config.py.
"""

import argparse
from pathlib import Path

import numpy as np

from .config import DataConfig
from .splits import discover_patient_ids, match_low_dose
from .volume_io import (
    compute_foreground_bbox,
    compute_norm_percentile,
    crop_with_bbox,
    load_volume,
    resize_axial,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _select_patient_ids(cfg: DataConfig, limit: int | None, patient_ids: list[str] | None):
    """Return the list of patient ids to analyze.

    Explicit --patient-id wins; otherwise the first `limit` discovered ids.
    """
    full_dose_dir = cfg.raw_dataset_dir / cfg.full_dose_subdir
    if patient_ids:
        return patient_ids
    discovered = discover_patient_ids(full_dose_dir, cfg.full_suffix)
    return discovered[:limit] if limit is not None else discovered


def _preprocessed_volume(cfg: DataConfig, patient_id: str, is_low_dose: bool = False) -> np.ndarray:
    """Full/low-dose volume after the exact spatial steps preprocessing applies before M.

    bbox (on volume) -> crop -> axial resize. Returns the (S, S, Z) array on which
    compute_norm_percentile would run, so any M we compute here matches training.
    """
    dose_folder = cfg.low_dose_subdir if is_low_dose else cfg.full_dose_subdir
    dose_dir = cfg.raw_dataset_dir / dose_folder
    path = ""

    if is_low_dose:
        path = match_low_dose(patient_id, dose_dir, cfg.low_suffix_variants)
    else:
        path = dose_dir / f"{patient_id}{cfg.full_suffix}"

    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Volume not found: {path}")

    volume, _, _ = load_volume(path)
    bbox = compute_foreground_bbox(volume, threshold=cfg.foreground_threshold)
    return resize_axial(crop_with_bbox(volume, bbox), cfg.image_size)


# ---------------------------------------------------------------------------
# Experiment: norm-percentile sweep
# ---------------------------------------------------------------------------

def exp_norm_percentile(args: argparse.Namespace) -> None:
    """Quantify the cost of lowering the asinh normalization percentile.

    For each candidate percentile p we compute M_p, then measure how many FOREGROUND
    voxels exceed M_p (these map above +1, i.e. out of the diffusion model's assumed
    [-1, +1] range). Crucially we also report, relative to the default percentile,
    the *additional* voxels pushed above +1 and *where* they sit along the axial axis
    (z-slice index, a head->feet proxy for anatomy since we have no segmentation).
    """
    cfg = DataConfig()
    default_p = cfg.norm_percentile
    percentiles = sorted(set(args.percentiles + [default_p]))
    pids = _select_patient_ids(cfg, args.limit, args.patient_id)
    is_low_dose = args.low_dose

    print(f"asinh normalization percentile sweep (default = {default_p})")
    print(f"volumes analyzed: {len(pids)}  |  image_size={cfg.image_size}\n")
    print(f"Volume dosage analyzed: {"LOW" if is_low_dose else "FULL"} dose volumes.")

    # Per-volume z-profiles of the voxels newly clipped when going default -> lowest p,
    # accumulated as a list of (Z,) arrays normalized to slice position in [0, 1].
    additional_z_profiles: list[tuple[np.ndarray, np.ndarray]] = []
    agg_rows: dict[float, list[float]] = {p: [] for p in percentiles}

    lowest_p = min(percentiles)

    for pid in pids:
        try:
            volume_resized = _preprocessed_volume(cfg, pid, is_low_dose)
        except FileNotFoundError as exc:
            print(f"[skip] {pid}: {exc}")
            continue

        fg_mask = volume_resized > 0
        fg = volume_resized[fg_mask]
        if fg.size == 0:
            print(f"[skip] {pid}: empty foreground")
            continue

        M = {p: compute_norm_percentile(volume_resized, p) for p in percentiles}
        M_default = M[default_p]

        print(f"  {pid}  (foreground voxels: {fg.size:,})")
        for p in percentiles:
            above = float((fg > M[p]).mean()) * 100.0
            agg_rows[p].append(above)
            tag = "  <- default" if p == default_p else ""
            print(
                f"    p={p:>5}  M={M[p]:10.2f}  "
                f"%foreground above +1: {above:5.2f}%{tag}"
            )
        # Voxels with M_lowest < value <= M_default: these are IN-range today but get
        # pushed above +1 if we adopt the lowest percentile. This is the "real signal lost".
        if lowest_p < default_p:
            newly = (volume_resized > M[lowest_p]) & (volume_resized <= M_default)
            z_counts = newly.sum(axis=(0, 1)).astype(np.float64)  # (Z,)
            z_pos = np.linspace(0.0, 1.0, num=z_counts.shape[0])
            additional_z_profiles.append((z_pos, z_counts))
            print(
                f"    going {default_p} -> {lowest_p}: "
                f"{int(newly.sum()):,} voxels ({newly.sum() / fg.size * 100:.2f}% of fg) "
                f"newly pushed above +1\n"
            )
        else:
            print()

    # Aggregate summary
    print("Aggregate %foreground mapped above +1 (mean +/- std across volumes):")
    for p in percentiles:
        vals = np.array(agg_rows[p])
        if vals.size:
            tag = "  <- default" if p == default_p else ""
            print(f"  p={p:>5}:  {vals.mean():5.2f}% +/- {vals.std():4.2f}%{tag}")

    if args.plot and additional_z_profiles:
        _plot_norm_percentile(additional_z_profiles, default_p, lowest_p, args.output_dir)


def _plot_norm_percentile(profiles, default_p, lowest_p, output_dir: Path) -> None:
    """Axial (head->feet) distribution of voxels pushed above +1 by lowering p."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for z_pos, z_counts in profiles:
        ax.plot(z_pos, z_counts, color="tab:blue", alpha=0.25, linewidth=1)
    ax.set_xlabel("axial position (0 = first kept slice, 1 = last)  ~ head -> feet")
    ax.set_ylabel("voxels newly pushed above +1")
    ax.set_title(
        f"Real foreground clipped when lowering norm percentile {default_p} -> {lowest_p}\n"
        f"(each line = one volume)"
    )
    fig.tight_layout()
    out = output_dir / f"norm_percentile_{default_p}_to_{lowest_p}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\nSaved z-distribution figure to {out}")


# ---------------------------------------------------------------------------
# Registry + CLI
# ---------------------------------------------------------------------------

EXPERIMENTS = {
    "norm-percentile": exp_norm_percentile,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis-supporting analysis experiments.")
    sub = parser.add_subparsers(dest="experiment", required=True)

    p_norm = sub.add_parser(
        "norm-percentile",
        help="How much real foreground gets pushed above +1 as M's percentile is lowered.",
    )
    p_norm.add_argument("--low-dose", action="store_true", help="Use the low dose images instead of default full-dose")
    p_norm.add_argument(
        "--percentiles",
        type=float,
        nargs="+",
        default=[95.0, 99.5, 99.9],
        help="Candidate percentiles to compare. The config default is always included.",
    )
    p_norm.add_argument("--limit", type=int, default=5, help="Analyze the first N volumes.")
    p_norm.add_argument(
        "--patient-id", nargs="+", default=None, help="Analyze these specific ids instead."
    )
    p_norm.add_argument("--plot", action="store_true", help="Save the axial-distribution figure.")
    p_norm.add_argument(
        "--output-dir", type=Path, default=Path("experiments_out"), help="Where to save figures."
    )

    args = parser.parse_args()
    EXPERIMENTS[args.experiment](args)


if __name__ == "__main__":
    main()
