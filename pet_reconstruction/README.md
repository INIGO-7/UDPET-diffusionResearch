# pet_reconstruction — Manual de usuario

Reconstrucción de PET de dosis completa a partir de PET de baja dosis (1/20) mediante modelos de difusión 2D por cortes axiales. Dos pipelines:

- **`supervised`** (Pipeline A): UNet condicionado por concatenación de canales (entrada = ruido ⊕ corte de baja dosis), muestreo DDIM.
- **`unconditional`** (Pipeline B): UNet incondicional + guiado por **DPS** (Diffusion Posterior Sampling) usando el corte de baja dosis como medida.

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
python -m src.main train --pipeline supervised --resume-from latest
python -m src.main train --pipeline supervised --resume-from checkpoint-epoch-019
python -m src.main train --pipeline unconditional --smoke
```

Flags:
- `--pipeline {supervised, unconditional}` (obligatorio).
- `--smoke`: shrink rápido (resolución 128², 5 epochs, checkpoint por epoch).
- `--resume-from`: reanuda entrenamiento. `latest` selecciona el `checkpoint-epoch-*` más reciente bajo el directorio del pipeline; alternativamente, el nombre exacto del subdirectorio.

Salida: `checkpoints/{supervised|unconditional}/checkpoint-epoch-NNN/` con subcarpetas `unet/` y `scheduler/`. Cada `save_model_epochs` epochs se guarda también una rejilla de reconstrucciones de ejemplo.

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

---

## 3. Llamada directa a los scripts (más control)

El dispatcher `src.main` reenvía todos los flags al script subyacente, así que puedes llamarlos directamente si lo prefieres:

```bash
python -m src.preprocess [--limit N] [--smoke]
python -m src.train_supervised        # equivalente al dispatcher pero sin smoke/resume helpers
python -m src.train_unconditional
python -m src.reconstruct_supervised      --checkpoint ... [flags]
python -m src.reconstruct_unconditional   --checkpoint ... [flags]
python -m src.evaluate                    --pipeline ... --checkpoint ... --output-dir ...
python -m src.preview_reconstruction      --pipeline ... [flags]
```

---

## 4. Utilidad extra: visualizador NIfTI

Inspeccionar pares dose_Full / dose_20 del dataset crudo con sliders sincronizados en los tres planos ortogonales:

```bash
python -m src.visualize_nifti                                # primer par del dataset
python -m src.visualize_nifti <prefijo>                      # ej. 01122021_1_20211201_164050
python -m src.visualize_nifti <ruta_full> <ruta_20>          # rutas explícitas
```

No requiere preprocesado: trabaja sobre los `.nii.gz` originales.

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
    ├── visualize_nifti.py     visor interactivo de NIfTI crudo
    └── preview_reconstruction.py     comparación de un corte
```

---

## 6. Convenciones rápidas

- **Espacio de trabajo del modelo**: cortes 2D, normalizados con asinh por volumen al rango ~[-1, +1].
- **Normalización sin fuga**: la escala `M`, el bbox y el filtro de cortes se derivan **solo de la baja dosis** (`prepare_low_dose`), nunca de la dosis completa. En inferencia (`reconstruct`/`evaluate`) `M` se **recalcula en runtime** desde el volumen de baja dosis, no se lee de `metadata.json` —así un volumen no visto funciona igual—. El `metadata.json` sigue guardando estos valores (idénticos a los de runtime) para el entrenamiento y para análisis.
- **Espacio de evaluación de intensidad**: cuentas PET originales (se invierte el asinh con esa misma `M` de baja dosis y `k`).
- **Split test**: blindado contra preview/entrenamiento; `preview` se niega a procesar pacientes fuera de test.
- **Checkpoints**: cada uno es un directorio con `unet/` (pesos) y `scheduler/` (config del scheduler). Apunta a ese directorio en los flags `--checkpoint`.
- **Reproducibilidad**: `split_seed=0` en `DataConfig`, `seed=0` en `TrainConfig`. Cambia en `config.py` si necesitas otra partición.
