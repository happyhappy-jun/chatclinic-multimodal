#!/bin/bash
# Launch the MCP gateway servers locally (Streamable HTTP):
#   - pubmed-lite (sample EXTERNAL server)         -> http://127.0.0.1:9001/mcp
#   - chatclinic-guideline-rag (OUR server)        -> http://127.0.0.1:9000/mcp
#
#   bash scripts/run_mcp_servers.sh           # foreground-ish (backgrounds both, tails logs)
#   bash scripts/run_mcp_servers.sh stop      # stop them
set -euo pipefail
cd "$(dirname "$0")/.."

# Use $PY if set, else a repo-local .venv if present, else the active env's python (conda).
if [ -z "${PY:-}" ]; then
    if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; else PY="python"; fi
fi
mkdir -p logs

if [ "${1:-start}" = "stop" ]; then
    pkill -f "mcp_servers/pubmed_server.py" 2>/dev/null || true
    pkill -f "mcp_servers/pubmed_lite_server.py" 2>/dev/null || true
    pkill -f "mcp_servers/chatclinic_rag_server.py" 2>/dev/null || true
    echo "stopped MCP servers"
    exit 0
fi

# load backend env (EMBED_MODEL / LOCAL_LLM_BASE_URL / PUBMED_EMAIL / NCBI_API_KEY)
set -a; [ -f .env ] && . ./.env; set +a

PORT=9002 "$PY" mcp_servers/pubmed_server.py        > logs/mcp-pubmed.log 2>&1 &
echo "pubmed (live NCBI) -> http://127.0.0.1:9002/mcp  (pid $!)"

PORT=9001 "$PY" mcp_servers/pubmed_lite_server.py   > logs/mcp-pubmed-lite.log 2>&1 &
echo "pubmed-lite (offline fallback) -> http://127.0.0.1:9001/mcp  (pid $!)"

PORT=9000 "$PY" mcp_servers/chatclinic_rag_server.py > logs/mcp-rag-server.log 2>&1 &
echo "chatclinic-guideline-rag (ours) -> http://127.0.0.1:9000/mcp  (pid $!)"

echo "logs: logs/mcp-pubmed-lite.log , logs/mcp-rag-server.log"
