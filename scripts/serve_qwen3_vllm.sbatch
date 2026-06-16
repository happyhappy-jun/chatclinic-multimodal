#!/bin/bash
#SBATCH --job-name=qwen3-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/vllm-%j.out
#SBATCH --error=logs/vllm-%j.out
#
# Serve Qwen/Qwen3-8B with vLLM's OpenAI-compatible API for the medical-rag tools.
#
#   sbatch scripts/serve_qwen3_vllm.sbatch
#   # watch logs/vllm-<jobid>.out for the node host and "Uvicorn running on http://0.0.0.0:8000"
#
# The compute node serves on <node-host>:8000. Point the backend at it via .env:
#   LOCAL_LLM_BASE_URL=http://<node-host>:8000/v1
#
# Self-contained: builds an isolated GPU venv with vLLM (Blackwell/B200 needs a
# CUDA 12.8+ build, which the pip default provides on this CUDA 13.1 node). The
# pinned environment.yml stack (vllm 0.18.1 / cu121) is the RTX 3090 submission
# target and is NOT used here.

set -euo pipefail

# --- config (env-overridable) ---------------------------------------------- #
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
SERVED_NAME="${SERVED_NAME:-qwen3-8b}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# Keep big artifacts off the home quota (data partition).
DATA_ROOT="${DATA_ROOT:-/NHNHOME/data/byungjun/vllm-runtime}"
GPU_VENV="${GPU_VENV:-$DATA_ROOT/venv-vllm}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf}"

# This cluster mounts /tmp, /dev/shm and /run as NOEXEC, so Triton/torch.compile
# cannot map their freshly compiled .so kernels there. Redirect every compile
# cache onto the data partition (the only exec-friendly mount).
export TMPDIR="$DATA_ROOT/tmp"
export TRITON_CACHE_DIR="$DATA_ROOT/triton-cache"
export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/inductor-cache"
export XDG_CACHE_HOME="$DATA_ROOT/xdg-cache"
mkdir -p "$DATA_ROOT" "$HF_HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME" logs

echo "[serve] node=$(hostname) model=$MODEL_ID served-as=$SERVED_NAME port=$PORT TP=$TP"
echo "[serve] reach this server at: http://$(hostname):$PORT/v1"
echo "[serve] -> set LOCAL_LLM_BASE_URL=http://$(hostname):$PORT/v1 in .env"

# --- build the GPU venv once ------------------------------------------------ #
if [ ! -x "$GPU_VENV/bin/python" ]; then
    echo "[serve] creating GPU venv at $GPU_VENV"
    python3 -m venv "$GPU_VENV"
    "$GPU_VENV/bin/pip" install --upgrade pip
    # Try the pinned version first (matches environment.yml); fall back to latest
    # if that wheel is not Blackwell-compatible on this node.
    "$GPU_VENV/bin/pip" install "vllm==0.18.1" \
        || "$GPU_VENV/bin/pip" install vllm
fi

# prometheus-fastapi-instrumentator (bundled by vLLM) calls route.path, which
# raises on Starlette's newer _IncludedRouter routes. Guard it idempotently.
ROUTING_PY="$("$GPU_VENV/bin/python" -c 'import prometheus_fastapi_instrumentator.routing as m; print(m.__file__)')"
sed -i 's/route_name = route\.path/route_name = getattr(route, "path", "") or ""/g' "$ROUTING_PY"

"$GPU_VENV/bin/python" -c "import vllm, torch; print('[serve] vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda)"

# --- serve ------------------------------------------------------------------ #
exec "$GPU_VENV/bin/python" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL"
    # add: --reasoning-parser qwen3   # only if running Qwen3 in thinking mode
