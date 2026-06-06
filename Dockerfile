# =============================================================================
# TFM — PET reconstruction (diffusion) — imagen CUDA para RunPod / local
# =============================================================================
# Base: Python 3.13.5 oficial (slim). NO trae CUDA, pero los wheels de PyTorch
# del índice cu124 ya empaquetan las libs de CUDA; solo hace falta el driver
# NVIDIA del host (lo provee RunPod / nvidia-container-toolkit en local).
FROM python:3.13.5-slim

# --- Configuración de entorno -------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface

# --- Dependencias del sistema -------------------------------------------------
# git: por si algún paquete lo necesita; libGL/glib: backends de matplotlib /
# scikit-image que tiran de libs gráficas aunque uses Agg.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        rsync \
        openssh-server \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /var/run/sshd

WORKDIR /workspace

# --- PyTorch con CUDA ---------------------------------------------------------
# Se instala ANTES que requirements.txt y desde el índice cu124, para que el
# resto de dependencias vea torch/torchvision ya satisfechos y no baje la build
# CPU desde PyPI. Sin fijar versión: pip elige la más reciente con wheels cp313
# (torch 2.4.1 NO los tiene; se necesita torch >= 2.6).
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# --- Resto de dependencias de Python -----------------------------------------
# Copiamos solo requirements.txt primero para cachear la capa de pip mientras
# el código cambia. huggingface_hub trae el CLI `hf` que usa el entrypoint para
# descargar el dataset preprocesado.
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r /workspace/requirements.txt && \
    pip install "huggingface_hub[cli,hf_transfer]"

# --- Entrypoint ---------------------------------------------------------------
# NO copiamos el código del proyecto: el entrypoint clona el repo en /workspace
# al arrancar (así RunPod siempre parte del repo actualizado) y descarga +
# reconstruye el dataset preprocesado desde Hugging Face. Solo horneamos el
# script de arranque.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

WORKDIR /workspace
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Por defecto el contenedor se queda vivo (RunPod necesita que el proceso
# principal no termine, y así el sshd sigue accesible). Para una sesión local
# interactiva sobreescribe el comando: docker compose run --rm tfm bash
CMD ["sleep", "infinity"]
