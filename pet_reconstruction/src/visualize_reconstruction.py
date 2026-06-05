"""
Visualizador de RECONSTRUCCIONES de un experimento (PET NIfTI).

Dado el nombre de un experimento (el subdirectorio bajo `reconstructions/`,
p.ej. "supervised_epoch99"), empareja cada volumen reconstruido
`<prefijo>_recon.nii.gz` con su dosis 1/20 (entrada) y su dosis completa
(referencia) del dataset crudo, y los muestra en tres filas
(1/20 dose / Reconstrucción / Full dose) con los tres planos ortogonales y
sliders sincronizados.

Además calcula métricas de calidad (PSNR / SSIM / NRMSE) sobre el volumen,
comparando contra la dosis completa:
    - baseline : 1/20 dose  vs  full dose   (sin reconstruir)
    - recon    : reconstrucción vs full dose

Uso (desde pet_reconstruction/):
    python -m src.visualize_reconstruction supervised_epoch99
    python -m src.visualize_reconstruction supervised_epoch99 <prefijo>

Dependencias:
    pip install nibabel matplotlib numpy scikit-image
"""

import os
import sys
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from .visualize_dataset import (
    DOSE_FULL_DIR,
    DOSE_20_DIR,
    PLANE_AXES,
    PLANE_LABELS,
    _prefix,
    _slice,
    load_clim,
    load_volume,
)
from .metrics import nrmse, psnr, ssim

# pet_reconstruction/ es dos niveles por encima de este archivo (src/ -> pet_reconstruction/),
# que es donde viven las reconstrucciones (a diferencia del dataset crudo, en TFM/res/).
_PET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECON_ROOT = os.path.join(_PET_ROOT, "reconstructions")


# ---------------------------------------------------------------------------
# Emparejamiento experimento -> (recon, full, low)
# ---------------------------------------------------------------------------

def _index_dose(dose_dir: str) -> dict[str, str]:
    return {_prefix(f): f for f in glob.glob(os.path.join(dose_dir, "*.nii*"))}


def list_triplets(experiment: str) -> list[tuple[str, str, str, str]]:
    """Devuelve [(prefijo, ruta_low, ruta_recon, ruta_full)] del experimento.

    Cada reconstrucción `<prefijo>_recon.nii*` se empareja con su dosis 1/20 y
    su dosis completa por prefijo temporal compartido.
    """
    exp_dir = os.path.join(RECON_ROOT, experiment)
    if not os.path.isdir(exp_dir):
        available = sorted(
            d for d in os.listdir(RECON_ROOT)
            if os.path.isdir(os.path.join(RECON_ROOT, d))
        ) if os.path.isdir(RECON_ROOT) else []
        raise FileNotFoundError(
            f"No existe el experimento '{experiment}' en {RECON_ROOT}.\n"
            f"Experimentos disponibles: {available or '(ninguno)'}"
        )

    recon_files = sorted(glob.glob(os.path.join(exp_dir, "*_recon.nii*")))
    if not recon_files:
        raise FileNotFoundError(f"No hay reconstrucciones (*_recon.nii*) en {exp_dir}")

    full_index = _index_dose(DOSE_FULL_DIR)
    low_index = _index_dose(DOSE_20_DIR)

    triplets: list[tuple[str, str, str, str]] = []
    for recon in recon_files:
        p = _prefix(recon)
        full = full_index.get(p)
        low = low_index.get(p)
        if full is None or low is None:
            print(f"[skip] {p}: falta {'full' if full is None else ''} "
                  f"{'low' if low is None else ''} dose en el dataset crudo")
            continue
        triplets.append((p, low, recon, full))

    if not triplets:
        raise FileNotFoundError(
            f"Ninguna reconstrucción de '{experiment}' pudo emparejarse con "
            f"dose_Full / dose_20 en el dataset crudo."
        )
    return triplets


def resolve_triplets(experiment: str, prefix_query: str | None) -> list[tuple[str, str, str, str]]:
    triplets = list_triplets(experiment)
    if prefix_query is None:
        return triplets
    matches = [t for t in triplets if prefix_query in t[0]]
    if not matches:
        raise FileNotFoundError(
            f"Ningún volumen del experimento '{experiment}' coincide con "
            f"el prefijo '{prefix_query}'."
        )
    # El consultado primero, pero conservamos el resto para los botones prev/next.
    rest = [t for t in triplets if t not in matches]
    return matches + rest


# ---------------------------------------------------------------------------
# Métricas a nivel de volumen (espacio de cuentas original)
# ---------------------------------------------------------------------------

def volume_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """PSNR / SSIM / NRMSE de un volumen 3D frente a la referencia (full dose).

    `data_range` se deriva de la referencia (full dose) para que baseline y
    recon sean comparables en la misma escala de cuentas.
    """
    pred = np.nan_to_num(pred.astype(np.float64))
    target = np.nan_to_num(target.astype(np.float64))
    data_range = float(target.max() - target.min())
    if data_range < 1e-12:
        data_range = 1.0
    return {
        "psnr": psnr(pred, target, data_range=data_range),
        "ssim": ssim(pred, target, data_range=data_range),
        "nrmse": nrmse(pred, target),
    }


def _format_metrics(prefix: str, low: np.ndarray, recon: np.ndarray, full: np.ndarray) -> str:
    base = volume_metrics(low, full)
    rec = volume_metrics(recon, full)
    return (
        f"{prefix}\n"
        f"{'':>10}  {'PSNR(dB)':>9}  {'SSIM':>6}  {'NRMSE':>7}\n"
        f"{'baseline':>10}  {base['psnr']:>9.2f}  {base['ssim']:>6.3f}  {base['nrmse']:>7.4f}   (1/20 vs full)\n"
        f"{'recon':>10}  {rec['psnr']:>9.2f}  {rec['ssim']:>6.3f}  {rec['nrmse']:>7.4f}   (recon vs full)"
    )


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

ROW_LABELS = ["1/20 dose", "Reconstrucción", "Full dose"]


def show_reconstructions(experiment: str, triplets: list[tuple[str, str, str, str]]) -> None:
    state = {"idx": 0}

    def paths(idx: int) -> tuple[str, str, str, str]:
        return triplets[idx]

    prefix0, low0, recon0, full0 = paths(0)
    vols = [load_volume(low0), load_volume(recon0), load_volume(full0)]
    clims = [load_clim(low0), load_clim(recon0), load_clim(full0)]

    nx, ny, nz = vols[0].shape
    init = [nx // 2, ny // 2, nz // 2]

    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle(f"Experimento: {experiment}", fontsize=11, y=0.99)

    ims = []
    for row, (vol, cl, rlabel) in enumerate(zip(vols, clims, ROW_LABELS)):
        row_ims = []
        for col, (axis, title) in enumerate(zip(PLANE_AXES, PLANE_LABELS)):
            ax = axes[row][col]
            im = ax.imshow(_slice(vol, axis, init[axis]), cmap="hot", vmin=cl[0], vmax=cl[1])
            if row == 0:
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(rlabel, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            row_ims.append(im)
        ims.append(row_ims)

    plt.subplots_adjust(bottom=0.30, top=0.92, hspace=0.05, wspace=0.04)

    # Caja de métricas (texto monoespaciado, debajo de las imágenes).
    metrics_box = fig.text(
        0.5, 0.255, "", ha="center", va="top", family="monospace", fontsize=9,
    )

    def _refresh_metrics() -> None:
        prefix, _, _, _ = paths(state["idx"])
        metrics_box.set_text(
            f"[{state['idx'] + 1}/{len(triplets)}]\n"
            + _format_metrics(prefix, vols[0], vols[1], vols[2])
        )

    _refresh_metrics()

    ax_sx = plt.axes([0.12, 0.17, 0.76, 0.022])
    ax_sy = plt.axes([0.12, 0.13, 0.76, 0.022])
    ax_sz = plt.axes([0.12, 0.09, 0.76, 0.022])
    s_x = Slider(ax_sx, "X (Sag)", 0, nx - 1, valinit=init[0], valfmt="%d")
    s_y = Slider(ax_sy, "Y (Cor)", 0, ny - 1, valinit=init[1], valfmt="%d")
    s_z = Slider(ax_sz, "Z (Ax)",  0, nz - 1, valinit=init[2], valfmt="%d")

    def update_slices(_):
        xi, yi, zi = int(s_x.val), int(s_y.val), int(s_z.val)
        for row, vol in enumerate(vols):
            ims[row][0].set_data(_slice(vol, 0, xi))
            ims[row][1].set_data(_slice(vol, 1, yi))
            ims[row][2].set_data(_slice(vol, 2, zi))
        fig.canvas.draw_idle()

    s_x.on_changed(update_slices)
    s_y.on_changed(update_slices)
    s_z.on_changed(update_slices)

    def load_triplet(idx: int) -> None:
        _, low_p, recon_p, full_p = paths(idx)
        vols[0] = load_volume(low_p)
        vols[1] = load_volume(recon_p)
        vols[2] = load_volume(full_p)
        clims[0], clims[1], clims[2] = load_clim(low_p), load_clim(recon_p), load_clim(full_p)
        nxi, nyi, nzi = vols[0].shape
        centers = [nxi // 2, nyi // 2, nzi // 2]
        s_x.valmax = nxi - 1; s_x.set_val(centers[0])
        s_y.valmax = nyi - 1; s_y.set_val(centers[1])
        s_z.valmax = nzi - 1; s_z.set_val(centers[2])
        for row, (vol, cl) in enumerate(zip(vols, clims)):
            for col, axis in enumerate(PLANE_AXES):
                ims[row][col].set_data(_slice(vol, axis, centers[axis]))
                ims[row][col].set_clim(cl[0], cl[1])
        _refresh_metrics()
        fig.canvas.draw_idle()

    if len(triplets) > 1:
        ax_prev = plt.axes([0.30, 0.02, 0.16, 0.04])
        ax_next = plt.axes([0.54, 0.02, 0.16, 0.04])
        btn_prev = Button(ax_prev, "◀  Anterior")
        btn_next = Button(ax_next, "Siguiente  ▶")

        def on_prev(_):
            state["idx"] = (state["idx"] - 1) % len(triplets)
            load_triplet(state["idx"])

        def on_next(_):
            state["idx"] = (state["idx"] + 1) % len(triplets)
            load_triplet(state["idx"])

        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)
        # Mantener referencias vivas frente al recolector de basura.
        fig._recon_buttons = (btn_prev, btn_next)

    plt.show()


def main() -> None:
    args = sys.argv[1:]
    if not args:
        raise SystemExit(
            "Uso: python -m src.visualize_reconstruction <experimento> [prefijo]\n"
            "  ej.: python -m src.visualize_reconstruction supervised_epoch99"
        )
    experiment = args[0]
    prefix_query = args[1] if len(args) > 1 else None
    triplets = resolve_triplets(experiment, prefix_query)
    show_reconstructions(experiment, triplets)


if __name__ == "__main__":
    main()
