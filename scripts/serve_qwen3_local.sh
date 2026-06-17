#!/bin/bash
# Serve Qwen/Qwen3-8B with vLLM on a plain GPU box (NO SLURM).
# Tuned defaults for a single 24 GB GPU (e.g. RTX 3090). Override via env.
#
#   bash scripts/serve_qwen3_local.sh            # foreground
#   PORT=8000 TP=1 bash scripts/serve_qwen3_local.sh
#   MODEL_ID=Qwen/Qwen3-8B GPU_UTIL=0.92 bash scripts/serve_qwen3_local.sh
#
# Exposes http://0.0.0.0:$PORT/v1 . Point the backend at it via .env:
#   LOCAL_LLM_BASE_URL=http://<this-host>:$PORT/v1
#   LOCAL_LLM_MODEL=$SERVED_NAME
set -euo pipefail
cd "$(dirname "$0")/.."

# Ignore ~/.local user-site packages — they can shadow the conda env's pinned
# torch/triton/pydantic and cause "duplicate template name" / arch-inspect failures.
export PYTHONNOUSERSITE=1

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
SERVED_NAME="${SERVED_NAME:-qwen3-8b}"
PORT="${PORT:-8000}"
TP="${TP:-1}"                       # tensor-parallel size = number of GPUs
MAX_LEN="${MAX_LEN:-12288}"         # 24 GB fits ~12k ctx for Qwen3-8B fp16; lower if OOM
GPU_UTIL="${GPU_UTIL:-0.92}"
DTYPE="${DTYPE:-auto}"              # set 'half' to force fp16 on older cards

# Keep big compile/model caches off the home quota and OFF any noexec mount.
DATA_ROOT="${DATA_ROOT:-$PWD/.vllm-runtime}"
export HF_HOME="${HF_HOME:-$DATA_ROOT/hf}"
mkdir -p "$DATA_ROOT" "$HF_HOME"

# Some clusters mount /tmp, /dev/shm, /run as NOEXEC, which breaks Triton/torch.compile
# (".so: failed to map segment"). Detect and redirect compile caches to an exec-friendly dir.
if mount 2>/dev/null | grep -E " /tmp .*noexec" >/dev/null || \
   findmnt -no OPTIONS --target /tmp 2>/dev/null | grep -q noexec; then
    echo "[serve] /tmp is noexec -> redirecting compile caches to $DATA_ROOT"
    export TMPDIR="$DATA_ROOT/tmp"
    export TRITON_CACHE_DIR="$DATA_ROOT/triton-cache"
    export TORCHINDUCTOR_CACHE_DIR="$DATA_ROOT/inductor-cache"
    export XDG_CACHE_HOME="$DATA_ROOT/xdg-cache"
    mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"
fi

# Pick a python that has vllm: active env first, else a repo-local GPU venv.
PYBIN="${PYBIN:-python}"
if ! "$PYBIN" -c "import vllm" 2>/dev/null; then
    if [ -x "$DATA_ROOT/venv/bin/python" ] && "$DATA_ROOT/venv/bin/python" -c "import vllm" 2>/dev/null; then
        PYBIN="$DATA_ROOT/venv/bin/python"
    else
        echo "[serve] ERROR: no 'vllm' in '$PYBIN'. Activate the conda env (environment.yml) first," >&2
        echo "        or: python -m venv $DATA_ROOT/venv && $DATA_ROOT/venv/bin/pip install vllm==0.18.1" >&2
        exit 1
    fi
fi

echo "[serve] host=$(hostname) model=$MODEL_ID served-as=$SERVED_NAME port=$PORT TP=$TP max_len=$MAX_LEN"
"$PYBIN" -c "import torch; print('[serve] CUDA', torch.version.cuda, '| GPUs', torch.cuda.device_count(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
echo "[serve] -> set LOCAL_LLM_BASE_URL=http://$(hostname):$PORT/v1 in .env"

# vLLM bundles prometheus-fastapi-instrumentator, which calls route.path and raises
# on Starlette's newer _IncludedRouter — crashing every request with a 500. Guard it.
ROUTING_PY="$("$PYBIN" -c 'import prometheus_fastapi_instrumentator.routing as m; print(m.__file__)' 2>/dev/null || true)"
if [ -n "$ROUTING_PY" ] && [ -f "$ROUTING_PY" ]; then
    sed -i 's/route_name = route\.path/route_name = getattr(route, "path", "") or ""/g' "$ROUTING_PY"
fi

exec "$PYBIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --max-model-len "$MAX_LEN" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_UTIL"
    # add: --reasoning-parser qwen3   # only if running Qwen3 in thinking mode
