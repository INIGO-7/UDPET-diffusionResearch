# pet_reconstruction — Manual de usuario

Reconstrucción de PET de dosis completa a partir de PET de baja dosis (1/20) mediante modelos de difusión 2D por cortes axiales. Pipelines:

- **`supervised`** (Pipeline A): UNet condicionado por concatenación de canales (entrada = ruido ⊕ corte de baja dosis), muestreo DDIM.
- **`unconditional`** (Pipeline B): UNet incondicional + guiado por **DPS** (Diffusion Posterior Sampling) usando el corte de baja dosis como medida.
- **`regression`** (Línea base A): la *misma* UNet del pipeline supervisado pero **sin difusión** — entrada = corte de baja dosis (1 canal), objetivo MSE en el espacio asinh `[-1,1]`, inferencia de una sola pasada. Ablación controlada del paradigma de difusión: misma arquitectura, mismos datos y presupuesto que Pipeline A, sólo cambia el proceso generativo.
- **`cnn`** (Línea base B): **RED-CNN** (Chen et al. 2017), la CNN de referencia para *denoising* de CT/PET de baja dosis (~1.8 M parámetros, ~50× menor que la UNet). Mismo entrenamiento que la línea base A; aísla la *familia arquitectónica* en vez del paradigma. Salida lineal (sin la ReLU final del original, ya que el espacio asinh es con signo y el fondo vive en −1).

Todos los comandos se ejecutan **desde la carpeta `pet_reconstruction/`** y usan el dispatcher `python -m src.main <subcomando>` (o directamente el módulo correspondiente cuando se quieren más flags).

---

## 0. Requisitos previos

- Dataset en `res/dataset/compressed_PET/dose_Full/` y `res/dataset/compressed_PET/dose_20/` (relativo a la raíz del repo).
- Dependencias del repo (`pip install -r ../requirements.txt`).
- GPU recomendada (CUDA). En MPS/CPU funciona pero los hiperparámetros por defecto están pensados para una GPU de 32 GB; ver nota en `src/config.py` para ajustes.

Todas las rutas de datos están ancladas a la raíz del repo, así que no importa desde dónde llames mientras el CWD sea `pet_reconstruction/`.

---

## 1. Flujo típico de extremo a extremo

```bash
cd pet_reconstruction/

# 1) Preprocesar (una sola vez, varias horas)
python -m src.main preprocess

# 2) Entrenar uno de los dos pipelines
python -m src.main train --pipeline supervised
# o
python -m src.main train --pipeline unconditional

# 3) Reconstruir el split de test con un checkpoint
python -m src.main reconstruct --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-029

# 4) Evaluar (métricas + figuras) sobre el split de test
python -m src.main evaluate --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
    --output-dir evaluations/supervised/

# 5) Vista previa cualitativa de un único corte
python -m src.main preview --pipeline supervised --list
python -m src.main preview --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
    --patient-id <test_pid> --slice-idx 120
```

Para una validación rápida del pipeline completo (~1 h), añade `--smoke` a `preprocess` y `train` (resolución 128², 5 epochs, guardar cada epoch).

---

## 2. Comandos del dispatcher (`src.main`)

### 2.1 `preprocess` — construir la caché de cortes

Convierte cada par NIfTI (full / 1-20) en cortes 2D normalizados (asinh, recortados al bbox de foreground, redimensionados a `image_size`²) guardados como `.pt` float16. Escribe también `metadata.json` con bbox, índices conservados, afín original y constantes de normalización por volumen.

**Importante (sin fuga de full-dose):** la geometría y la normalización —bbox de foreground, escala asinh `M` (percentil 99.5) y filtro de cortes— se derivan **únicamente del volumen de baja dosis** (lo único disponible en inferencia real) y se aplican idénticamente al target de dosis completa. La lógica vive en `prepare_low_dose()` (en `preprocess.py`), que es la **única fuente de verdad** compartida con la reconstrucción, de modo que la normalización de entrenamiento y la de inferencia coinciden exactamente.

```bash
python -m src.main preprocess              # todo el dataset
python -m src.main preprocess --limit 50   # solo los primeros 50 pacientes
python -m src.main preprocess --smoke      # variante 128² (más rápida)
```

Flags:
- `--limit N`: procesar solo los primeros N pacientes (útil para iterar).
- `--smoke`: reduce la resolución a 128². Combinable con `--limit`.

Salida: `data/pet_cache/{patient_id}/{full,low}/NNNN.pt` y `data/pet_cache/metadata.json`. Los splits train/val/test se generan en `data/splits.json`.

### 2.2 `train` — entrenar un pipeline

```bash
python -m src.main train --pipeline supervised
python -m src.main train --pipeline unconditional
python -m src.main train --pipeline regression          # línea base A (UNet sin difusión)
python -m src.main train --pipeline cnn                  # línea base B (RED-CNN)
python -m src.main train --pipeline supervised --resume-from latest
python -m src.main train --pipeline supervised --resume-from checkpoint-epoch-019
python -m src.main train --pipeline cnn --smoke
```

Flags:
- `--pipeline {supervised, unconditional, regression, cnn}` (obligatorio).
- `--smoke`: shrink rápido (resolución 128², 5 epochs, checkpoint por epoch).
- `--resume-from`: reanuda entrenamiento. `latest` selecciona el `checkpoint-epoch-*` más reciente bajo el directorio del pipeline; alternativamente, el nombre exacto del subdirectorio.

Salida: `checkpoints/{supervised|unconditional|regression_unet|cnn_redcnn}/checkpoint-epoch-NNN/`. Los pipelines de difusión guardan subcarpetas `unet/` y `scheduler/`; la línea base `regression` guarda sólo `unet/`; la línea base `cnn` guarda un `model.pt` (state dict; RED-CNN no es un modelo `diffusers`). Cada `save_model_epochs` epochs se guarda también una previsualización de reconstrucciones de ejemplo en TensorBoard.

Configuración por defecto (en `src/config.py`): batch 32, 100 epochs, `v_prediction`, schedule cosine, EMA 0.9999, bf16, lr 1.4e-4 con warmup. Si entrenas en MPS/CPU, ajusta `train_batch_size`, `gradient_accumulation_steps` y `mixed_precision` siguiendo las notas del fichero.

### 2.3 `reconstruct` — generar volúmenes NIfTI a partir de un checkpoint

Lee el **NIfTI de baja dosis crudo** de cada paciente del split solicitado y deriva en **tiempo de ejecución** (vía `prepare_low_dose`) el bbox, la escala asinh `M` y los cortes conservados a partir de ese volumen —**no usa la caché ni `metadata.json`**—. Muestrea con DDIM-50 y reensambla un volumen en la rejilla original (deshace asinh → resize inverso → vuelve al bbox), guardándolo como NIfTI con el afín del propio volumen de entrada.

Como todo se calcula en runtime desde la baja dosis, `reconstruct` funciona también sobre un **volumen no visto** que no haya pasado por el preprocesado, mediante `--low-volume`.

```bash
python -m src.main reconstruct --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-029

python -m src.main reconstruct --pipeline unconditional \
    --checkpoint checkpoints/unconditional/checkpoint-epoch-029 \
    --output-dir reconstructions/unconditional/ \
    --split test --limit 5 --inference-batch-size 8 --omega 1.5

# Reconstruir directamente un volumen de baja dosis no visto (sin caché ni split):
python -m src.main reconstruct --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
    --low-volume /ruta/a/paciente_1-20\ dose.nii.gz
```

Flags pass-through al script interno:
- `--checkpoint PATH` (obligatorio): directorio que contiene `unet/`.
- `--output-dir PATH` (default `reconstructions/{pipeline}`).
- `--split {train,val,test}` (default `test`).
- `--limit N`: solo los primeros N pacientes.
- `--low-volume PATH`: reconstruye un único NIfTI de baja dosis crudo (volumen no visto); ignora `--split`, `--limit` y la caché.
- `--inference-batch-size` (default 4).
- `--omega FLOAT` (solo `unconditional`): sobreescribe la escala de guiado DPS.

Salida: `<output-dir>/<patient_id>_recon.nii.gz`. **Requiere los NIfTI de baja dosis crudos** disponibles (en `res/dataset/...` para el modo por split, o la ruta dada en `--low-volume`).

### 2.4 `evaluate` — métricas + figuras sobre un split

Para cada paciente del split: corre inferencia, calcula PSNR / SSIM / NRMSE (whole y foreground) en espacio normalizado, invierte el asinh y calcula métricas de preservación de intensidad en espacio de cuentas, y guarda figuras de 4 paneles para una muestra representativa de pacientes. La escala `M` para invertir el asinh se estima **en runtime** del volumen de baja dosis crudo (no se lee de `metadata.json`), igual que en `reconstruct`; por tanto evaluate necesita tanto la caché (ground-truth de dosis completa) como los NIfTI de baja dosis crudos.

```bash
python -m src.main evaluate --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
    --output-dir evaluations/supervised/

python -m src.main evaluate --pipeline unconditional \
    --checkpoint checkpoints/unconditional/checkpoint-epoch-029 \
    --output-dir evaluations/unconditional/ \
    --split val --n-figures 10 --inference-batch-size 8 --omega 1.0
```

Flags:
- `--checkpoint PATH` (obligatorio).
- `--output-dir PATH` (obligatorio).
- `--split {train,val,test}` (default `test`).
- `--limit N`: evaluar solo los primeros N pacientes.
- `--n-figures N` (default 5): número de pacientes equiespaciados que reciben figura de 4 paneles.
- `--inference-batch-size` (default 4).
- `--omega FLOAT` (solo `unconditional`).

Salida bajo `--output-dir`:
- `per_slice.csv`: una fila por (paciente, corte) con todas las métricas.
- `per_volume.csv`: media por paciente.
- `summary.json`: agregados globales y config de la ejecución.
- `figures/<pid>_slice<NNNN>.png`: figuras 4-panel.

Al terminar imprime un resumen con PSNR/SSIM/NRMSE whole+fg y errores porcentuales de intensidad.

### 2.5 `preview` — comparar un único corte (cualitativo)

Pinta un panel low / recon / full para un corte de un paciente del **test split** (solo test, para que la comparación sea honesta).

```bash
# Listar pacientes de test disponibles y cuántos cortes tienen cacheados
python -m src.main preview --pipeline supervised --list

# Renderizar un corte (default: el central)
python -m src.main preview --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-029 \
    --patient-id 01122021_1_20211201_164050 \
    --slice-idx 120 \
    --save-path previews/sup_01122021_1_s120.png
```

Flags:
- `--list`: imprime IDs de test válidos y sale (no necesita `--checkpoint`).
- `--checkpoint PATH`: obligatorio salvo con `--list`.
- `--patient-id ID`: obligatorio. Debe pertenecer al split test.
- `--slice-idx N`: índice (acepta el nombre 4-dígitos del cache o un índice posicional). Default: corte central.
- `--save-path PATH`: guarda la figura en disco; si se omite, se abre interactivamente.
- `--omega FLOAT`: solo `unconditional`.

### 2.6 `slice-recon` — exportar PNGs limpios de un corte (figuras de la memoria)

Hace inferencia sobre **un solo corte** de un paciente con un modelo dado y guarda **tres PNG limpios** (solo píxeles, sin ejes/título/colorbar) a la **resolución original** del escáner. Las ejecuciones se organizan por paciente+corte: `reports/slice_recon/<patient_id>_<slice>/<pipeline>_<checkpoint>_<uuid8>/`, de modo que todas las reconstrucciones del mismo corte (distintos modelos/épocas) conviven bajo el mismo padre. Dentro de cada carpeta de ejecución:

- `full_dose.png` — corte full-dose crudo, grid original, sin tocar.
- `low_dose.png` — corte low-dose equivalente, grid original, sin tocar.
- `recon_<pipeline>_<checkpoint>.png` — reconstrucción del modelo remapeada al grid original (asinh⁻¹ → resize → bbox).

Los tres comparten **una única ventana de intensidad** (`vmin=0`, `vmax`=percentil 99.5 del foreground del full-dose) para que sean directamente comparables: el low-dose se ve genuinamente tenue/ruidoso (~1/20 de cuentas) y una buena reconstrucción iguala al full-dose. Se escribe además un `meta.json` con todos los parámetros (paciente, corte, modelo, `M`, `k`, ventana, split al que pertenece el paciente, ...).

A diferencia de `preview`, **acepta cualquier paciente** (train/val/test) por ser una herramienta de reporte; el split se registra en `meta.json`. Toda la geometría (bbox / `M` / cortes conservados) se deriva **en runtime** del NIfTI de baja dosis crudo (como `reconstruct`), así que no necesita la caché de preprocesado.

```bash
# Listar los cortes reconstruibles (kept slices) de un paciente y salir
python -m src.main slice-recon --pipeline supervised \
    --patient-id 01122021_1_20211201_164050 --list-slices

# Exportar los 3 PNG de un corte (omite --slice-idx para el corte central)
python -m src.main slice-recon --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-099 \
    --patient-id 01122021_1_20211201_164050 --slice-idx 320
```

Flags:
- `--pipeline {supervised,unconditional,regression,cnn}` (obligatorio).
- `--checkpoint PATH`: obligatorio salvo con `--list-slices`.
- `--patient-id ID` (obligatorio): cualquier paciente presente en el dataset crudo.
- `--slice-idx N`: índice axial z en el grid original; debe ser un corte conservado (foreground). Default: el corte central conservado.
- `--list-slices`: imprime los índices z reconstruibles del paciente y sale.
- `--output-dir PATH` (default `reports/slice_recon`): dentro se crea `<patient_id>_<slice>/<pipeline>_<checkpoint>_<uuid8>/`.
- `--cmap NAME` (default `gray`): colormap de matplotlib para los PNG.
- `--rot90 N`: rota cada panel 90·N grados (solo orientación de display).
- `--omega FLOAT`: solo `unconditional`.

### 2.7 `mip-recon` — MIP de volumen completo sobre un plano (figuras de la memoria)

Hermano de `slice-recon`, pero en vez de un corte proyecta el **volumen completo** sobre un plano anatómico (`coronal` / `sagittal` / `axial`) con una **proyección de máxima intensidad** (MIP) — la vista de lectura PET clásica de cuerpo entero. Guarda tres PNG limpios en `reports/MIP_recon/<patient_id>/<pipeline>_<checkpoint>_<uuid8>/`:

- `full_dose_<plane>.png` — MIP del volumen full-dose crudo.
- `low_dose_<plane>.png` — MIP del volumen low-dose crudo.
- `recon_<pipeline>_<checkpoint>_<plane>.png` — MIP del volumen reconstruido por el modelo.

La reconstrucción se corre **en runtime sobre todo el volumen** (misma geometría `prepare_low_dose` que `reconstruct`), así que no necesita caché y vale para cualquier paciente y cualquiera de los 4 pipelines. **Nota de coste:** en los pipelines de difusión (`supervised`/`unconditional`) esto muestrea cada corte con DDIM-50, por lo que un volumen entero es lento; las líneas base `cnn`/`regression` son una sola pasada y van rápidas.

Los tres MIP comparten **una única ventana** derivada del MIP de full-dose (`vmin=0`, `vmax`=percentil 99.5 de su foreground). El eje de proyección se elige del affine del NIfTI (`aff2axcodes`) y la imagen se orienta con Superior arriba (coronal/sagital) o Anterior arriba (axial). Se escribe `meta.json` con todos los parámetros.

```bash
python -m src.main mip-recon --pipeline supervised \
    --checkpoint checkpoints/supervised/checkpoint-epoch-099 \
    --patient-id 01122021_1_20211201_164050 --plane coronal
```

Flags:
- `--pipeline {supervised,unconditional,regression,cnn}` (obligatorio).
- `--checkpoint PATH` (obligatorio).
- `--patient-id ID` (obligatorio): cualquier paciente del dataset crudo.
- `--plane {coronal,sagittal,axial}` (obligatorio): plano de proyección del MIP.
- `--output-dir PATH` (default `reports/MIP_recon`): dentro se crea `<patient_id>/<pipeline>_<checkpoint>_<uuid8>/`.
- `--inference-batch-size` (default 4).
- `--cmap NAME` (default `gray`).
- `--rot90 N` / `--flipud` / `--fliplr`: ajuste fino de la orientación de display.
- `--omega FLOAT`: solo `unconditional`.

---

## 3. Llamada directa a los scripts (más control)

El dispatcher `src.main` reenvía todos los flags al script subyacente, así que puedes llamarlos directamente si lo prefieres:

```bash
python -m src.preprocess [--limit N] [--smoke]
python -m src.train_supervised        # equivalente al dispatcher pero sin smoke/resume helpers
python -m src.train_unconditional
python -m src.train_regression            # línea base A (UNet sin difusión)
python -m src.train_cnn                    # línea base B (RED-CNN)
python -m src.reconstruct_supervised      --checkpoint ... [flags]
python -m src.reconstruct_unconditional   --checkpoint ... [flags]
python -m src.reconstruct_regression      --checkpoint ... [flags]
python -m src.reconstruct_cnn             --checkpoint ... [flags]
python -m src.evaluate                    --pipeline ... --checkpoint ... --output-dir ...
python -m src.preview_reconstruction      --pipeline ... [flags]
python -m src.slice_recon                 --pipeline ... --patient-id ... [flags]
python -m src.mip_recon                   --pipeline ... --patient-id ... --plane ... [flags]
```

---

## 4. Utilidades extra: visualizadores NIfTI

**Dataset crudo** — inspeccionar pares dose_Full / dose_20 con sliders sincronizados en los tres planos ortogonales:

```bash
python -m src.visualize_dataset                              # primer par del dataset
python -m src.visualize_dataset <prefijo>                    # ej. 01122021_1_20211201_164050
python -m src.visualize_dataset <ruta_full> <ruta_20>        # rutas explícitas
```

**Reconstrucciones de un experimento** — comparar 1/20 dose / reconstrucción / full dose (tres filas, tres planos) y métricas de volumen (PSNR/SSIM/NRMSE) de la reconstrucción frente al baseline (1/20 sin reconstruir), ambos contra la dosis completa. El argumento es el nombre del experimento, es decir el subdirectorio bajo `reconstructions/`:

```bash
python -m src.visualize_reconstruction supervised_epoch99            # todos los volúmenes del experimento
python -m src.visualize_reconstruction supervised_epoch99 <prefijo>  # un volumen concreto primero
```

Ninguno requiere preprocesado: trabajan sobre los `.nii.gz` originales (y, en el segundo caso, sobre los `*_recon.nii.gz` ya generados por `reconstruct`).

---

## 5. Diseño de la carpeta

```
pet_reconstruction/
├── README.md                  (este fichero)
├── checkpoints/               salida de entrenamiento ({pipeline}/checkpoint-epoch-NNN/)
└── src/
    ├── main.py                dispatcher CLI
    ├── config.py              dataclasses con TODOS los hiperparámetros
    ├── preprocess.py          NIfTI -> cortes .pt normalizados
    ├── splits.py              descubrimiento de pacientes y train/val/test
    ├── volume_io.py           load/save NIfTI, asinh ±resize, bbox, reensamblado
    ├── data.py                Dataset/DataLoader sobre la caché
    ├── _unet_builder.py       construcción del UNet (compartido)
    ├── _train_engine.py       loop de entrenamiento (compartido)
    ├── model_supervised.py    Pipeline A (channel-concat)
    ├── model_unconditional.py Pipeline B (incondicional)
    ├── train_supervised.py    entry-point de entrenamiento A
    ├── train_unconditional.py entry-point de entrenamiento B
    ├── reconstruct_supervised.py     DDIM A -> NIfTI
    ├── reconstruct_unconditional.py  DDIM + DPS B -> NIfTI
    ├── evaluate.py            métricas + figuras 4-panel
    ├── metrics.py             PSNR/SSIM/NRMSE, intensity preservation
    ├── visualize.py           figuras 3-panel y 4-panel
    ├── visualize_dataset.py   visor interactivo del dataset crudo (full vs 1/20)
    ├── visualize_reconstruction.py  visor de reconstrucciones + métricas por experimento
    ├── preview_reconstruction.py     comparación de un corte
    ├── slice_recon.py                exporta PNGs full/low/recon de un corte (figuras memoria)
    └── mip_recon.py                  exporta MIP full/low/recon de volumen completo por plano
```

---

## 6. Convenciones rápidas

- **Espacio de trabajo del modelo**: cortes 2D, normalizados con asinh por volumen al rango ~[-1, +1].
- **Normalización sin fuga**: la escala `M`, el bbox y el filtro de cortes se derivan **solo de la baja dosis** (`prepare_low_dose`), nunca de la dosis completa. En inferencia (`reconstruct`/`evaluate`) `M` se **recalcula en runtime** desde el volumen de baja dosis, no se lee de `metadata.json` —así un volumen no visto funciona igual—. El `metadata.json` sigue guardando estos valores (idénticos a los de runtime) para el entrenamiento y para análisis.
- **Espacio de evaluación de intensidad**: cuentas PET originales (se invierte el asinh con esa misma `M` de baja dosis y `k`).
- **Split test**: blindado contra preview/entrenamiento; `preview` se niega a procesar pacientes fuera de test.
- **Checkpoints**: cada uno es un directorio con `unet/` (pesos) y `scheduler/` (config del scheduler). Apunta a ese directorio en los flags `--checkpoint`.
- **Reproducibilidad**: `split_seed=0` en `DataConfig`, `seed=0` en `TrainConfig`. Cambia en `config.py` si necesitas otra partición.
