#!/bin/bash
# One-command bring-up for the Medical RAG stack (everything EXCEPT the vLLM model server,
# which runs separately via scripts/serve_qwen3_local.sh and may live on another host).
#
# Brings up: index (if missing) -> backend (:8001) -> MCP servers (:9000-9002) -> frontend (:3000)
#
#   bash scripts/run_all.sh            # start everything, health-check, then tail
#   bash scripts/run_all.sh stop       # stop everything this script started
#   NO_FRONTEND=1 bash scripts/run_all.sh   # skip the Next.js frontend
#
# Prereqs: conda env (environment.yml) + pip install -r requirements-rag.txt; .env configured
# with LOCAL_LLM_BASE_URL pointing at a running vLLM (local or remote).
set -uo pipefail
cd "$(dirname "$0")/.."

# Ignore ~/.local user-site packages so the active env's pinned deps win.
export PYTHONNOUSERSITE=1

PY="${PY:-python}"
BACKEND_PORT="${BACKEND_PORT:-8001}"
PIDFILE=".run_all.pids"
mkdir -p logs

stop_all() {
    echo "[run_all] stopping..."
    bash scripts/run_mcp_servers.sh stop 2>/dev/null || true
    if [ -f "$PIDFILE" ]; then
        while read -r pid; do [ -n "$pid" ] && kill "$pid" 2>/dev/null || true; done < "$PIDFILE"
        rm -f "$PIDFILE"
    fi
    pkill -f "uvicorn app.main:app" 2>/dev/null || true
    pkill -f "next dev" 2>/dev/null || true
    echo "[run_all] stopped."
}

if [ "${1:-start}" = "stop" ]; then stop_all; exit 0; fi

# fresh start
stop_all >/dev/null 2>&1
: > "$PIDFILE"

set -a; [ -f .env ] && . ./.env; set +a
LLM="${LOCAL_LLM_BASE_URL:-http://localhost:8000/v1}"
echo "[run_all] LOCAL_LLM_BASE_URL=$LLM  EMBED_MODEL=${EMBED_MODEL:-BAAI/bge-m3}"

wait_http() { # url, name, tries
    for _ in $(seq 1 "${3:-30}"); do
        curl -s -o /dev/null "$1" 2>/dev/null && { echo "[run_all] $2 ready"; return 0; }
        sleep 1
    done
    echo "[run_all] WARN: $2 not responding at $1"; return 1
}

# 0. model server reachable?
if curl -s -o /dev/null --max-time 5 "${LLM%/v1}/v1/models" 2>/dev/null; then
    echo "[run_all] model server reachable ✓"
else
    echo "[run_all] NOTE: model server not reachable at $LLM — RAG will use the extractive fallback."
    echo "          Start it with: bash scripts/serve_qwen3_local.sh   (then set LOCAL_LLM_BASE_URL)"
fi

# 1. build the retrieval index if absent
INDEX_DIR="${GUIDELINE_INDEX_DIR:-data/guideline_index}"
if [ ! -f "$INDEX_DIR/faiss.index" ]; then
    echo "[run_all] building guideline index ..."
    "$PY" -m plugins.guideline_index_tool.logic > logs/index-build.log 2>&1 \
        && echo "[run_all] index built -> $INDEX_DIR" \
        || { echo "[run_all] index build FAILED — see logs/index-build.log"; }
else
    echo "[run_all] index present ($INDEX_DIR) ✓"
fi

# 2. backend
"$PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" > logs/backend.log 2>&1 &
echo $! >> "$PIDFILE"
wait_http "http://127.0.0.1:$BACKEND_PORT/docs" "backend(:$BACKEND_PORT)" 30

# 3. MCP servers (pubmed live :9002, pubmed-lite :9001, our rag server :9000)
bash scripts/run_mcp_servers.sh start
wait_http "http://127.0.0.1:9002/mcp" "mcp-pubmed(:9002)" 15

# 4. frontend (optional)
if [ "${NO_FRONTEND:-0}" != "1" ]; then
    if command -v npm >/dev/null 2>&1; then
        [ -d node_modules ] || { echo "[run_all] npm install ..."; npm install > logs/npm-install.log 2>&1; }
        npm run dev:webapp > logs/frontend.log 2>&1 &
        echo $! >> "$PIDFILE"
        wait_http "http://127.0.0.1:3000" "frontend(:3000)" 40
    else
        echo "[run_all] npm not found — skipping frontend (backend API still usable)."
    fi
fi

echo ""
echo "[run_all] UP. Endpoints:"
echo "   backend  : http://localhost:$BACKEND_PORT/docs"
echo "   demo UI  : http://localhost:3000/guideline"
echo "   MCP      : :9000 (our rag)  :9002 (pubmed live)  :9001 (pubmed-lite)"
echo "   logs/    : backend.log, frontend.log, mcp-*.log"
echo "   stop     : bash scripts/run_all.sh stop"
