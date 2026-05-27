#!/usr/bin/env bash
# ─── RunPod CLAM Training Server — Environment Setup ──────────────────────────
# Port 8002 | server_id = 4
# Source this file before running the server:
#   source /workspace/RunPod-CLAM/scripts/setup_env.sh

set -a   # auto-export all variables

# ── Service identity ───────────────────────────────────────────────────────────
export PORT="${PORT:-8002}"
export LARAVEL_SERVER_ID="${LARAVEL_SERVER_ID:-4}"
export LARAVEL_BASE_URL="${LARAVEL_BASE_URL:-https://ai.histopathology.cloud}"

# ── CLAM API Key (set in /workspace/setup_env.sh on the pod) ─────────────────
# API_KEY should come from the pod-level setup_env.sh as CLAM_API_KEY
export API_KEY="${CLAM_API_KEY:-}"

# ── HuggingFace token (needed only if downloading models) ────────────────────
export HF_TOKEN="${HF_TOKEN:-}"

# ── Activate virtual environment ─────────────────────────────────────────────
VENV_PATH="/workspace/RunPod-CLAM/venv"
if [ -f "${VENV_PATH}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_PATH}/bin/activate"
    echo "[setup_env] CLAM venv activated: ${VENV_PATH}"
else
    echo "[setup_env] WARNING: venv not found at ${VENV_PATH} — run setup first"
fi

# ── CUDA visibility ───────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

set +a
echo "[setup_env] CLAM env ready — PORT=${PORT}  LARAVEL_SERVER_ID=${LARAVEL_SERVER_ID}"
