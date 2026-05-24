"""
butterfly_image_info.py
Información básica sobre las imágenes del dataset smithsonian_butterflies_subset.

Uso (desde butterfly_generation/):
    python -m src.dataset_inspect --data_dir <ruta_a_data/butterflies>

El directorio debe contener el archivo .arrow (ej: data-00000-of-00001.arrow).
"""

import argparse
import io
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from PIL import Image


def get_arrow_file(data_dir: str) -> Path:
    base = Path(data_dir)
    candidates = sorted(base.glob("*.arrow"))
    if not candidates:
        raise FileNotFoundError(f"No se encontraron archivos .arrow en: {data_dir}")
    return candidates[0]


def analyze_images(data_dir: str):
    arrow_file = get_arrow_file(data_dir)

    print(f"\n📁 Directorio : {data_dir}")
    print(f"📄 Archivo    : {arrow_file.name}")

    with ipc.open_stream(arrow_file) as reader:
        table = reader.read_all()

    print(f"\n🗂️  Esquema del dataset:")
    for field in table.schema:
        print(f"   • {field.name:<20} {field.type}")

    print(f"\n⏳ Procesando {len(table)} imágenes...\n")

    widths, heights, modes, sizes_kb = [], [], [], []
    errors = 0

    image_col = table.column("image")

    for item in image_col:
        raw_value = item.as_py()
        try:
            if isinstance(raw_value, bytes):
                raw = raw_value
            elif isinstance(raw_value, dict):
                raw = raw_value.get("bytes")
            else:
                errors += 1
                continue

            img = Image.open(io.BytesIO(raw))
            widths.append(img.width)
            heights.append(img.height)
            modes.append(img.mode)
            sizes_kb.append(len(raw) / 1024)

        except Exception as e:
            errors += 1
            print(f"  ⚠️  Error: {e}")

    if not widths:
        print("❌ No se pudieron procesar imágenes.")
        return

    areas = [w * h for w, h in zip(widths, heights)]
    min_idx = areas.index(min(areas))
    max_idx = areas.index(max(areas))

    print("=" * 55)
    print("       INFORMACIÓN BÁSICA DEL DATASET DE IMÁGENES")
    print("=" * 55)

    print(f"\n📊 GENERAL")
    print(f"  Total imágenes procesadas : {len(widths)}")
    print(f"  Errores al procesar       : {errors}")

    print(f"\n📐 DIMENSIONES (píxeles)")
    print(f"  Ancho  — mín: {min(widths):>6}  máx: {max(widths):>6}  promedio: {sum(widths)/len(widths):>8.1f}")
    print(f"  Alto   — mín: {min(heights):>6}  máx: {max(heights):>6}  promedio: {sum(heights)/len(heights):>8.1f}")
    print(f"\n  Imagen más pequeña : {widths[min_idx]} x {heights[min_idx]} px  ({min(areas):,} px²)")
    print(f"  Imagen más grande  : {widths[max_idx]} x {heights[max_idx]} px  ({max(areas):,} px²)")

    print(f"\n🎨 MODOS DE COLOR")
    mode_counts = {}
    for m in modes:
        mode_counts[m] = mode_counts.get(m, 0) + 1
    for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
        print(f"  {mode:<8} : {count:>5} imágenes  ({count/len(modes)*100:.1f}%)")

    print(f"\n💾 TAMAÑO EN DISCO (bytes crudos)")
    print(f"  Mínimo   : {min(sizes_kb):>8.1f} KB")
    print(f"  Máximo   : {max(sizes_kb):>8.1f} KB")
    print(f"  Promedio : {sum(sizes_kb)/len(sizes_kb):>8.1f} KB")
    print(f"  Total    : {sum(sizes_kb)/1024:>8.1f} MB")

    print(f"\n📏 PROPORCIÓN (aspect ratio)")
    ratios = [w / h for w, h in zip(widths, heights)]
    cuadradas   = sum(1 for r in ratios if abs(r - 1.0) < 0.05)
    horizontales = sum(1 for r in ratios if r > 1.05)
    verticales   = sum(1 for r in ratios if r < 0.95)
    print(f"  Cuadradas (~1:1)  : {cuadradas:>5} ({cuadradas/len(ratios)*100:.1f}%)")
    print(f"  Horizontales (>1) : {horizontales:>5} ({horizontales/len(ratios)*100:.1f}%)")
    print(f"  Verticales   (<1) : {verticales:>5} ({verticales/len(ratios)*100:.1f}%)")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Análisis básico de imágenes del dataset de mariposas.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Ruta al directorio que contiene el .arrow (ej: data/butterflies)",
    )
    args = parser.parse_args()
    analyze_images(args.data_dir)