"""
Visualizador comparativo de dosis completa vs dosis 1/20 (PET NIfTI).

Muestra los tres planos ortogonales de ambos volúmenes en paralelo con
sliders sincronizados para navegar por los cortes.

Uso (desde pet_reconstruction/):
    python -m src.visualize_nifti                         # primer par del dataset
    python -m src.visualize_nifti <prefijo>               # ej. "01122021_1_20211201_164050"
    python -m src.visualize_nifti <ruta_full> <ruta_20>   # rutas absolutas/relativas

Dependencias:
    pip install nibabel matplotlib numpy
"""

import os
import sys
import glob
import numpy as np
from functools import lru_cache
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Raíz del proyecto: dos niveles por encima de este archivo (pet_reconstruction/src/
# -> pet_reconstruction/ -> TFM/), independientemente del cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOSE_FULL_DIR = os.path.join(_PROJECT_ROOT, "res", "dataset", "compressed_PET", "dose_Full")
DOSE_20_DIR   = os.path.join(_PROJECT_ROOT, "res", "dataset", "compressed_PET", "dose_20")


# ---------------------------------------------------------------------------
# Emparejamiento de archivos
# ---------------------------------------------------------------------------

def _prefix(path: str) -> str:
    """Extrae el prefijo temporal compartido por ambas dosis."""
    name = os.path.basename(path)
    for suffix in ("_Full_dose.nii.gz", "_1-20 dose.nii.gz", "_Full_dose.nii", "_1-20 dose.nii"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def list_pairs() -> list[tuple[str, str]]:
    """Devuelve todos los pares (full, 20) ordenados por prefijo."""
    full_files   = {_prefix(f): f for f in glob.glob(os.path.join(DOSE_FULL_DIR, "*.nii*"))}
    dose20_files = {_prefix(f): f for f in glob.glob(os.path.join(DOSE_20_DIR,   "*.nii*"))}
    common = sorted(set(full_files) & set(dose20_files))
    if not common:
        raise FileNotFoundError(
            f"No se encontraron pares coincidentes en:\n  {DOSE_FULL_DIR}\n  {DOSE_20_DIR}"
        )
    return [(full_files[p], dose20_files[p]) for p in common]


def resolve_paths(args: list[str]) -> tuple[str, str]:
    if len(args) == 2:
        a, b = args
        if not os.path.isfile(a):
            raise FileNotFoundError(f"No existe: {a}")
        if not os.path.isfile(b):
            raise FileNotFoundError(f"No existe: {b}")
        return a, b

    pairs = list_pairs()

    if len(args) == 1:
        prefix_query = args[0]
        matches = [(f, d) for f, d in pairs if prefix_query in _prefix(f)]
        if not matches:
            raise FileNotFoundError(f"No se encontró ningún par con prefijo '{prefix_query}'")
        return matches[0]

    return pairs[0]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

@lru_cache(maxsize=20)
def load_volume(path: str) -> np.ndarray:
    img  = nib.load(path)
    data = np.squeeze(img.get_fdata())
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Se esperaba volumen 3D, shape={data.shape} en {path}")
    print(f"{'Full' if 'Full' in path else '1/20':5s}  shape={data.shape}  "
          f"range=[{np.nanmin(data):.1f}, {np.nanmax(data):.1f}]  {os.path.basename(path)}")
    return data


@lru_cache(maxsize=20)
def load_clim(path: str) -> tuple[float, float]:
    lo, hi = np.nanpercentile(load_volume(path), [1, 99])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

PLANE_AXES   = [0, 1, 2]
PLANE_LABELS = ["Sagital (X)", "Coronal (Y)", "Axial (Z)"]

def _slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    s = [slice(None)] * 3
    s[axis] = idx
    return np.rot90(vol[tuple(s)])



# ---------------------------------------------------------------------------
# Entrada
# ---------------------------------------------------------------------------

def show_comparison(path_full: str, path_20: str, pairs: list[tuple[str, str]] | None = None) -> None:
    vol_full = load_volume(path_full)
    vol_20   = load_volume(path_20)

    nx, ny, nz = vol_full.shape
    init = [nx // 2, ny // 2, nz // 2]

    clim_full = load_clim(path_full)
    clim_20   = load_clim(path_20)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    title_obj = fig.suptitle("", fontsize=9)

    def _update_title(pf: str) -> None:
        idx_str = ""
        if pairs:
            idx = next((i for i, (f, _) in enumerate(pairs) if f == pf), None)
            if idx is not None:
                idx_str = f"  [{idx + 1}/{len(pairs)}]"
        title_obj.set_text(
            f"Full dose  (arriba)   vs   1/20 dose  (abajo){idx_str}\n{_prefix(pf)}"
        )

    _update_title(path_full)

    vols       = [vol_full, vol_20]
    clims      = [clim_full, clim_20]
    row_labels = ["Full dose", "1/20 dose"]
    ims        = []

    for row, (vol, cl, rlabel) in enumerate(zip(vols, clims, row_labels)):
        row_ims = []
        for col, (axis, title) in enumerate(zip(PLANE_AXES, PLANE_LABELS)):
            ax = axes[row][col]
            im = ax.imshow(_slice(vol, axis, init[axis]), cmap="hot", vmin=cl[0], vmax=cl[1])
            if row == 0:
                ax.set_title(title, fontsize=8)
            if col == 0:
                ax.set_ylabel(rlabel, fontsize=8)
            ax.axis("off")
            row_ims.append(im)
        ims.append(row_ims)

    plt.subplots_adjust(bottom=0.28, hspace=0.05, wspace=0.04)

    ax_sx = plt.axes([0.12, 0.18, 0.76, 0.025])
    ax_sy = plt.axes([0.12, 0.13, 0.76, 0.025])
    ax_sz = plt.axes([0.12, 0.08, 0.76, 0.025])
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

    # Botones anterior / siguiente (solo si hay lista de pares)
    if pairs:
        state = {"idx": next((i for i, (f, _) in enumerate(pairs) if f == path_full), 0)}

        ax_prev = plt.axes([0.30, 0.02, 0.16, 0.04])
        ax_next = plt.axes([0.54, 0.02, 0.16, 0.04])
        btn_prev = Button(ax_prev, "◀  Anterior")
        btn_next = Button(ax_next, "Siguiente  ▶")

        def load_pair(idx: int) -> None:
            pf, p20 = pairs[idx]
            v_full = load_volume(pf)
            v_20   = load_volume(p20)
            cf = load_clim(pf)
            c20 = load_clim(p20)
            nxi, nyi, nzi = v_full.shape
            # Reinicia sliders al corte central del nuevo volumen
            s_x.valmax = nxi - 1; s_x.set_val(nxi // 2)
            s_y.valmax = nyi - 1; s_y.set_val(nyi // 2)
            s_z.valmax = nzi - 1; s_z.set_val(nzi // 2)
            vols[0], vols[1] = v_full, v_20
            clims[0], clims[1] = cf, c20
            for row, (vol, cl) in enumerate(zip(vols, clims)):
                for col, axis in enumerate(PLANE_AXES):
                    ims[row][col].set_data(_slice(vol, axis, [nxi//2, nyi//2, nzi//2][axis]))
                    ims[row][col].set_clim(cl[0], cl[1])
            _update_title(pf)
            fig.canvas.draw_idle()

        def on_prev(_):
            state["idx"] = (state["idx"] - 1) % len(pairs)
            load_pair(state["idx"])

        def on_next(_):
            state["idx"] = (state["idx"] + 1) % len(pairs)
            load_pair(state["idx"])

        btn_prev.on_clicked(on_prev)
        btn_next.on_clicked(on_next)

    plt.show()


def main() -> None:
    pairs = list_pairs()
    path_full, path_20 = resolve_paths(sys.argv[1:])
    show_comparison(path_full, path_20, pairs=pairs)


if __name__ == "__main__":
    main()
