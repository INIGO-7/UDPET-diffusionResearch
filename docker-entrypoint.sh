#!/usr/bin/env bash
# =============================================================================
# Entrypoint: prepara el entorno en /workspace al arrancar el contenedor.
#   1) Clona (o actualiza) el repositorio del proyecto.
#   2) Descarga el dataset preprocesado desde Hugging Face Datasets y
#      reconstruye data/pet_cache/ a partir de los shards de tar.
#   3) Verifica el conteo de ficheros .pt y limpia los shards.
#   4) Ejecuta el comando pasado (por defecto, una shell).
#
# Es idempotente: si /workspace ya tiene el repo y el dataset reconstruido
# (almacenamiento persistente de RunPod), se salta los pasos ya hechos.
# =============================================================================
set -euo pipefail

# --- Configuración (sobreescribible por variables de entorno) -----------------
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_URL="${REPO_URL:-https://github.com/INIGO-7/UDPET-diffusionResearch.git}"
REPO_BRANCH="${REPO_BRANCH:-master}"
REPO_DIR="${REPO_DIR:-${WORKSPACE}/UDPET-diffusionResearch}"

# Repo de Hugging Face Datasets (privado) con los shards del tar + splits.json.
HF_DATASET_REPO="${HF_DATASET_REPO:-}"     # ej: INIGO-7/udpet-pet-cache  (OBLIGATORIO)
# HF_TOKEN se lee automáticamente por el CLI de huggingface si está exportado.

# Verificación de integridad del dataset reconstruido.
EXPECTED_PT_COUNT="${EXPECTED_PT_COUNT:-477690}"

# Flags opcionales.
SYNC_DEPS="${SYNC_DEPS:-0}"   # 1 = reinstala requirements.txt del repo clonado
SKIP_DATASET="${SKIP_DATASET:-0}"   # 1 = no toca el dataset (p.ej. ya en un volumen)

# Acelera descargas grandes desde HF.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

log() { echo -e "\033[1;34m[entrypoint]\033[0m $*"; }
err() { echo -e "\033[1;31m[entrypoint]\033[0m $*" >&2; }

# --- 0) SSH (para RunPod) -----------------------------------------------------
# RunPod inyecta la clave pública del usuario en la env var PUBLIC_KEY. Se
# arranca sshd ANTES de la descarga para poder conectarse mientras baja el
# dataset. En local (sin PUBLIC_KEY y sin puerto mapeado) es inofensivo.
setup_ssh() {
    mkdir -p /var/run/sshd ~/.ssh
    chmod 700 ~/.ssh
    if [ -n "${PUBLIC_KEY:-}" ]; then
        echo "${PUBLIC_KEY}" >> ~/.ssh/authorized_keys
        chmod 600 ~/.ssh/authorized_keys
        log "Clave pública de RunPod añadida a authorized_keys."
    else
        log "PUBLIC_KEY no definida; sshd arranca igualmente (sin clave inyectada)."
    fi
    ssh-keygen -A >/dev/null 2>&1 || true   # genera host keys si faltan
    /usr/sbin/sshd && log "sshd arrancado (puerto 22)." || err "no se pudo arrancar sshd."
}
setup_ssh

# --- 1) Repositorio -----------------------------------------------------------
if [ ! -d "${REPO_DIR}/.git" ]; then
    log "Clonando ${REPO_URL} (rama ${REPO_BRANCH}) en ${REPO_DIR} ..."
    git clone --branch "${REPO_BRANCH}" "${REPO_URL}" "${REPO_DIR}"
else
    log "Repo ya presente en ${REPO_DIR}; haciendo git pull ..."
    git -C "${REPO_DIR}" pull --ff-only || log "git pull falló (¿cambios locales?); continúo."
fi

if [ "${SYNC_DEPS}" = "1" ] && [ -f "${WORKSPACE}/requirements.txt" ]; then
    log "SYNC_DEPS=1 → reinstalando requirements.txt del repo ..."
    pip install -r "${WORKSPACE}/requirements.txt" || err "pip install falló; continúo."
fi

# data/ vive en la raíz del repo (config.py: _REPO_ROOT/data/pet_cache)
DATA_DIR="${REPO_DIR}/data"
CACHE_DIR="${DATA_DIR}/pet_cache"
READY_MARK="${DATA_DIR}/.dataset_ready"

# --- 2 + 3) Dataset -----------------------------------------------------------
if [ "${SKIP_DATASET}" = "1" ]; then
    log "SKIP_DATASET=1 → no se toca el dataset."
elif [ -f "${READY_MARK}" ]; then
    log "Dataset ya reconstruido (marca ${READY_MARK}); se omite descarga."
else
    # La preparación del dataset NO debe tumbar el contenedor: si algo falla,
    # se registra el error y se sigue, dejando el pod vivo para depurar por SSH.
    # (set +e dentro de un subshell aislado.)
    (
        set +e
        if [ -z "${HF_DATASET_REPO}" ]; then
            err "HF_DATASET_REPO no está definido y el dataset no existe. No puedo descargarlo."
            err "Define HF_DATASET_REPO (y HF_TOKEN si es privado) o monta data/ por volumen + SKIP_DATASET=1."
            exit 1
        fi

        mkdir -p "${DATA_DIR}"
        DL_DIR="${DATA_DIR}/_hf_download"
        log "Descargando dataset ${HF_DATASET_REPO} desde Hugging Face ..."
        hf download "${HF_DATASET_REPO}" --repo-type dataset --local-dir "${DL_DIR}" || {
            err "La descarga de HF falló. Revisa HF_TOKEN/HF_DATASET_REPO."; exit 1; }

        # splits.json va fuera del tar, junto a los shards.
        if [ -f "${DL_DIR}/splits.json" ]; then
            cp -f "${DL_DIR}/splits.json" "${DATA_DIR}/splits.json"
            log "splits.json colocado en ${DATA_DIR}/splits.json"
        else
            err "No se encontró splits.json en la descarga."
        fi

        log "Reconstruyendo pet_cache/ a partir de los shards (cat | tar) ..."
        rm -rf "${CACHE_DIR}"
        # split -d genera ...part-00, part-01, ...; el glob ordena correctamente.
        cat "${DL_DIR}"/pet_cache.tar.part-* | tar xf - -C "${DATA_DIR}" --exclude='._*' --exclude='.DS_Store' || {
            err "Fallo al reconstruir el tar."; exit 1; }

        log "Verificando integridad (conteo de .pt, excluyendo AppleDouble) ..."
        ACTUAL_PT_COUNT="$(find "${CACHE_DIR}" -name '*.pt' -not -name '._*' | wc -l | tr -d ' ')"
        if [ "${ACTUAL_PT_COUNT}" != "${EXPECTED_PT_COUNT}" ]; then
            err "Conteo de .pt = ${ACTUAL_PT_COUNT}, esperado ${EXPECTED_PT_COUNT}. NO limpio los shards."
            err "Revisa la descarga en ${DL_DIR} antes de reintentar."
            exit 1
        fi
        log "OK: ${ACTUAL_PT_COUNT} ficheros .pt. Limpiando shards descargados ..."
        rm -rf "${DL_DIR}"
        touch "${READY_MARK}"
        log "Dataset listo en ${CACHE_DIR}"
    ) || err "Preparación del dataset incompleta; el pod sigue vivo para depurar."
fi

# --- 4) Comando ---------------------------------------------------------------
# Los comandos del proyecto se ejecutan desde pet_reconstruction/
cd "${REPO_DIR}/pet_reconstruction" 2>/dev/null || cd "${REPO_DIR}"
log "Entorno listo. CWD = $(pwd)"
exec "$@"
