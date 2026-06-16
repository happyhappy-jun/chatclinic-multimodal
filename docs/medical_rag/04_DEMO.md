# 04 · Running the Demo

Full stack: **browser `/guideline` page → FastAPI backend → local Qwen3-8B (vLLM on a B200) → grounded, cited, verified answer.**

## One-time setup
```bash
# backend deps (already done in .venv)
pip install -r requirements-rag.txt          # sentence-transformers, faiss-cpu, pypdf

# frontend deps (workspace-hoisted to repo-root node_modules)
npm install                                   # from repo root

# build the retrieval index from the guideline corpus
set -a; . ./.env; set +a
.venv/bin/python -m plugins.guideline_index_tool.logic
```

## 1. Serve the local LLM (GPU via SLURM)
```bash
sbatch scripts/serve_qwen3_vllm.sbatch
# find the node + readiness:
squeue -u $USER
grep "Uvicorn running" logs/vllm-<jobid>.out      # ~50s once venv+model are cached
# point the backend at it:
#   .env -> LOCAL_LLM_BASE_URL=http://<node-host>:8000/v1   (e.g. ALIN-B200-10)
```
Stop it when done to free the GPU: `scancel <jobid>`.

## 2. Start the backend (port 8001 = the frontend's default API base)
```bash
set -a; . ./.env; set +a
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

## 3. Start the frontend
```bash
npm run dev:webapp            # http://localhost:3000
```
Open **http://localhost:3000/guideline**, click an example question or type your own.

## What you'll see
- The grounded answer with inline **[REF#]** citation chips.
- A **faithfulness %** badge + per-claim ✓/✕ from the citation verifier.
- The retrieved evidence passages with cosine scores and sources.
- Out-of-corpus questions return *"Insufficient evidence in the provided guidelines."*

## API (no UI)
```bash
curl -X POST http://127.0.0.1:8001/api/v1/guideline-rag/run \
  -H 'Content-Type: application/json' \
  -d '{"question":"First-line antibiotics for outpatient community-acquired pneumonia?","top_k":4}'
```

## Notes
- If the LLM endpoint is unreachable, the tool degrades to an **extractive** answer (top passages
  verbatim) so the demo never hard-fails.
- The `/guideline` page is a self-contained demo route; the tool is also registered as a Studio
  renderer (`guideline_rag`) for integration into the main 3-panel workspace.
