"""Qualitative 4-panel figure: low / full / reconstruction / |residual|.

Designed for the test-set reporting figures referenced in memoria/info.md
section "Evaluación". Operates on normalized 2D slices (or count slices —
the function does not interpret the intensity scale).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def four_panel_figure(
    low: np.ndarray,
    full: np.ndarray,
    recon: np.ndarray,
    save_path: Path | None = None,
    title: str | None = None,
    cmap: str = "gray",
    residual_cmap: str = "hot",
) -> None:
    """Plot low / full / reconstruction / absolute residual side by side.

    All three input arrays must share intensity scale. The residual is
    rendered with its own colormap+colorbar so its range is not coupled
    to the other three panels.
    """
    residual = np.abs(recon - full)
    vmin = float(min(low.min(), full.min(), recon.min()))
    vmax = float(max(low.max(), full.max(), recon.max()))

    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    panels = [
        ("Low dose", low, cmap, vmin, vmax),
        ("Full dose", full, cmap, vmin, vmax),
        ("Reconstruction", recon, cmap, vmin, vmax),
        ("|Recon − Full|", residual, residual_cmap, None, None),
    ]
    for ax, (lbl, img, cm, vlo, vhi) in zip(axs, panels):
        im = ax.imshow(img, cmap=cm, vmin=vlo, vmax=vhi)
        ax.set_title(lbl, fontsize=10)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def three_panel_figure(
    low: np.ndarray,
    recon: np.ndarray,
    full: np.ndarray,
    save_path: Path | None = None,
    title: str | None = None,
    cmap: str = "gray",
) -> None:
    """Plot low-dose / reconstruction / full-dose side by side on a shared scale."""
    vmin = float(min(low.min(), recon.min(), full.min()))
    vmax = float(max(low.max(), recon.max(), full.max()))

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [
        ("Low dose (input)", low),
        ("Reconstruction", recon),
        ("Full dose (ground truth)", full),
    ]
    for ax, (lbl, img) in zip(axs, panels):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(lbl, fontsize=10)
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
