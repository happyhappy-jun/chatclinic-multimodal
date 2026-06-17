#!/bin/bash
# Serve Qwen/Qwen3-8B with SGLang (OpenAI-compatible API) on a plain GPU box.
# Drop-in alternative to scripts/serve_qwen3_local.sh (vLLM) — the backend only
# speaks the OpenAI /v1 API, so nothing else changes except LOCAL_LLM_BASE_URL.
#
#   bash scripts/serve_qwen3_sglang.sh                 # foreground, port 30000
#   PORT=30000 TP=1 bash scripts/serve_qwen3_sglang.sh
#   MEM_FRAC=0.85 MAX_LEN=12288 bash scripts/serve_qwen3_sglang.sh
#
# Exposes http://0.0.0.0:$PORT/v1 . Then set in .env:
#   LOCAL_LLM_BASE_URL=http://<this-host>:$PORT/v1
#   LOCAL_LLM_MODEL=$SERVED_NAME
#
# Install (pip, inside the conda env — see docs/medical_rag/08_SGLANG.md):
#   conda activate chatclinic
#   pip install "sglang[all]"
set -euo pipefail
cd "$(dirname "$0")/.."

# Ignore ~/.local user-site packages (they can shadow the env's pinned torch/triton).
export PYTHONNOUSERSITE=1

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
SERVED_NAME="${SERVED_NAME:-qwen3-8b}"
PORT="${PORT:-30000}"               # SGLang's conventional port
TP="${TP:-1}"                       # tensor-parallel = number of GPUs
MAX_LEN="${MAX_LEN:-12288}"         # context length; lower if OOM on 24 GB
MEM_FRAC="${MEM_FRAC:-0.85}"        # SGLang's static GPU memory fraction (~vLLM gpu-util)

# Keep model/compile caches off the home quota and OFF any noexec mount.
DATA_ROOT="${DATA_ROOT:-$PWD/.sglang-runtime}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf}"
mkdir -p "$DATA_ROOT" "$HF_HOME"

# Some clusters mount /tmp, /dev/shm, /run NOEXEC, which breaks Triton/torch.compile
# (".so: failed to map segment"). Redirect compile caches to an exec-friendly dir.
if findmnt -no OPTIONS --target /tmp 2>/dev/null | grep -q noexec; then
    echo "[serve] /tmp is noexec -> redirecting compile caches to $DATA_ROOT"
    export TMPDIR="$DATA_ROOT/tmp"
    export TRITON_CACHE_DIR="$DATA_ROOT/triton-cache"
    export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/inductor-cache"
    export XDG_CACHE_HOME="$DATA_ROOT/xdg-cache"
    mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
fi

# Pick a python that has sglang (the active conda env). Surface the REAL import
# error — a raised exception here usually means a torch/flashinfer mismatch, not
# a missing install.
PYBIN="${PYBIN:-python}"
if ! SGLANG_VER="$("$PYBIN" -c "import sglang; print(sglang.__version__)" 2>&1)"; then
    echo "[serve] ERROR: cannot import sglang in '$PYBIN' ($("$PYBIN" -c 'import sys;print(sys.executable)'))" >&2
    echo "[serve] --- real error ---" >&2
    echo "$SGLANG_VER" | tail -6 >&2
    echo "[serve] -----------------" >&2
    echo "[serve] install/repair:  conda activate chatclinic && pip install 'sglang[all]'  (see docs/medical_rag/08_SGLANG.md)" >&2
    exit 1
fi
echo "[serve] sglang $SGLANG_VER"

echo "[serve] host=$(hostname) model=$MODEL_ID served-as=$SERVED_NAME port=$PORT TP=$TP max_len=$MAX_LEN"
"$PYBIN" -c "import torch; print('[serve] CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
echo "[serve] -> set LOCAL_LLM_BASE_URL=http://$(hostname):$PORT/v1 in .env"

exec "$PYBIN" -m sglang.launch_server \
    --model-path "$MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp "$TP" \
    --context-length "$MAX_LEN" \
    --mem-fraction-static "$MEM_FRAC"
    # add: --reasoning-parser qwen3   # only if running Qwen3 in thinking mode
