# 07 · Deploy on Another Server (e.g. RTX 3090)

The **application is portable** — every host/path/model is read from env vars; nothing is hardcoded.
Only the SLURM script (`serve_qwen3_vllm.sbatch`) was cluster-specific. For a plain GPU box use the
no-SLURM scripts below.

## What runs where

| Component | Port | Process | GPU? |
|---|---|---|---|
| vLLM model server (Qwen3-8B) | 8000 | `scripts/serve_qwen3_local.sh` | **yes (1× 24 GB)** |
| FastAPI backend | 8001 | `uvicorn app.main:app` | no |
| MCP servers (ours + pubmed + lite) | 9000–9002 | `scripts/run_mcp_servers.sh` | no |
| Next.js frontend (demo UI) | 3000 | `npm run dev:webapp` | no |

The model server can run on the **same** box or a **different** one — the backend just needs
`LOCAL_LLM_BASE_URL` pointing at it.

## 1. Environment (once)

`environment.yml` is **self-contained** — it installs the base ChatClinic stack **and** the medical-rag
tools (sentence-transformers, faiss-cpu, pypdf, mcp) in one shot. No separate `requirements-rag.txt`
step is needed on the conda path.

```bash
git clone <repo> && cd chatclinic-multimodal
conda env create -f environment.yml      # base + medical-rag deps; python 3.10, torch 2.5.1+cu121, vllm 0.18.1 (Ampere/3090)
conda activate chatclinic
cp .env.example .env
```

**venv path (no conda):** `requirements.txt` also pulls in the RAG deps via `-r`, so one install covers all:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt        # base + medical-rag in one command
```

> Tip: if a freshly created env is missing packages that exist in `~/.local`, conda/pip may have skipped
> them during creation. Run with `PYTHONNOUSERSITE=1` (the serve/run scripts set this automatically) so
> only the env's pinned deps load.

Edit `.env`:
```
LOCAL_LLM_BASE_URL=http://localhost:8000/v1     # or http://<gpu-host>:8000/v1 if remote
LOCAL_LLM_MODEL=qwen3-8b
EMBED_MODEL=BAAI/bge-m3                          # production retriever
PUBMED_EMAIL=you@example.com                     # NCBI etiquette (no key needed)
# NCBI_API_KEY=...                               # optional, 3->10 req/s
```

## 2. Serve the model (1× 3090)

The backend only speaks the **OpenAI-compatible `/v1` API**, so the serving framework is swappable —
use **vLLM or SGLang**; only `LOCAL_LLM_BASE_URL` differs.

**Option A — vLLM** (uses the pinned `environment.yml` stack):
```bash
bash scripts/serve_qwen3_local.sh
# defaults: Qwen/Qwen3-8B, port 8000, TP=1, max_len 12288, gpu_util 0.92
# .env -> LOCAL_LLM_BASE_URL=http://localhost:8000/v1
```

**Option B — SGLang** (pip-install into the conda env; full guide in [08_SGLANG.md](08_SGLANG.md)):
```bash
conda activate chatclinic && pip install "sglang[all]"
bash scripts/serve_qwen3_sglang.sh
# defaults: Qwen/Qwen3-8B, port 30000, TP=1, max_len 12288, mem_frac 0.85
# .env -> LOCAL_LLM_BASE_URL=http://localhost:30000/v1   (note port 30000)
```

Common knobs (both): `TP=2` for multi-GPU, `MAX_LEN=8192` if you OOM on 24 GB. Both scripts auto-detect
a `noexec` `/tmp` and redirect Triton/compile caches, and set `PYTHONNOUSERSITE=1`. 3090 is Ampere
(sm_86), fully supported by both — no B200-style rebuild. Qwen3 runs in non-thinking mode by default
(our client sends `enable_thinking=false`; add `--reasoning-parser qwen3` to serve thinking mode).

## 3. Bring up the rest

```bash
bash scripts/run_all.sh
#   builds the FAISS index if missing -> backend(:8001) -> MCP(:9000-9002) -> frontend(:3000)
#   stop: bash scripts/run_all.sh stop
#   headless (no UI): NO_FRONTEND=1 bash scripts/run_all.sh
```

Open **http://localhost:3000/guideline**, or call the API:
```bash
curl -X POST http://localhost:8001/api/v1/guideline-rag/run \
  -H 'Content-Type: application/json' \
  -d '{"question":"first-line antibiotics for outpatient pneumonia","external_evidence":true}'
```

## Sizing notes (24 GB GPU)

- Qwen3-8B fp16 ≈ 16 GB weights; the rest is KV cache. `max_len 12288 @ util 0.92` is a safe start.
- If you see CUDA OOM: drop `MAX_LEN` (e.g. 8192) or `GPU_UTIL` (e.g. 0.88).
- Two 3090s? `TP=2` halves per-GPU memory and roughly doubles throughput.

## Remote model server

Run `serve_qwen3_local.sh` on the GPU host, then on the app host set
`LOCAL_LLM_BASE_URL=http://<gpu-host>:8000/v1`. Ensure port 8000 is reachable (open firewall or
`ssh -L 8000:localhost:8000 <gpu-host>`).

## Access from your laptop (SSH port-forward)

The `/guideline` page calls the backend from the browser, so forward **both**:
```bash
ssh -L 3000:localhost:3000 -L 8001:localhost:8001 <server>
# then open http://localhost:3000/guideline
```

## Portability checklist

- [ ] `conda activate chatclinic` (or a venv with `vllm`) before serving.
- [ ] `.env` set: `LOCAL_LLM_BASE_URL`, `EMBED_MODEL`, `PUBMED_EMAIL`.
- [ ] Index built (`run_all.sh` does it, or `python -m plugins.guideline_index_tool.logic`).
- [ ] Ports 8000/8001/3000 reachable (or forwarded).
- [ ] First run downloads Qwen3-8B (~16 GB) + the embedder — needs internet (or pre-cache `HF_HOME`).
