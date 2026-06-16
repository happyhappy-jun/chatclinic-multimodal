# 01 · Environment Setup (all three tracks)

We are running **fully local inference** — no OpenAI API. Two local models:
a **local embedding model** (retrieval) and a **local LLM served by vLLM** (generation).

Three setup tracks:

- **A. Backend dev env** — ✅ done (base), needs RAG deps added.
- **B. Frontend** — Node/npm install + `webapp/`.
- **C. Local model / RAG stack** — embeddings + FAISS + vLLM server, sized for ≤ 4× RTX 3090.

---

## A. Backend dev environment

**Status: ✅ working.** Recap of what's set up:

- Clone at `/NHNHOME/data/byungjun/health-project/chatclinic-multimodal`, branch `medical-rag`.
- venv `.venv/` (python 3.12) with base deps: fastapi, uvicorn, python-multipart, pydantic, pysam, openpyxl, Pillow, nibabel.
- Run: `.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8011` → 40 routes, Swagger `/docs`.
- `.env` from `.env.example`.

**RAG deps to add** (CPU-only retrieval libs; safe on this box):

```bash
.venv/bin/pip install \
  "sentence-transformers>=3.0" \
  "FlagEmbedding>=1.2" \
  "faiss-cpu>=1.8" \
  "pypdf>=4.0" \
  "numpy<2.3"          # keep numpy compatible with faiss/torch wheels
```

> Note: the **submission** environment is `environment.yml` (conda, python 3.10, torch 2.5.1+cu121,
> transformers 4.57.6, vllm 0.18.1). Our `.venv` is a fast dev shim. Before the PR we must confirm the
> tools import and run under the pinned conda env on an RTX 3090 box. Add the RAG deps to a
> `requirements-rag.txt` (or extend `environment.yml`) so graders can reproduce.

---

## B. Frontend (`webapp/`)

Node/npm are **not installed**. Install without root via `nvm`:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
. "$HOME/.nvm/nvm.sh"
nvm install 20            # Next.js 14+ wants Node 18/20+
node -v && npm -v
```

Then:

```bash
cd webapp
npm install
cp ../.env .env.local      # if the frontend needs the API base URL; check webapp/.env usage
npm run dev                # dev server (usually :3000), proxies to backend :8011
npm run build              # REQUIRED PR check (CONTRIBUTING.md)
```

> The frontend is needed for the **demo video** and the `npm run build` PR gate. The RAG logic itself
> lives in the backend, so we can build/iterate the tools before the UI is wired.

---

## C. Local model / RAG stack (the important one)

### C.1 Retrieval (embeddings + FAISS) — small, runs anywhere

- **Embedding model** (local, downloaded once, ~0.5–2 GB): pick one of
  - `BAAI/bge-m3` (strong general multilingual retriever), or
  - `ncbi/MedCPT-Query-Encoder` + `MedCPT-Article-Encoder` (biomedical, asymmetric), or
  - `pritamdeka/S-PubMedBert-MS-MARCO` (biomedical sentence embeddings).
- **Index:** `faiss-cpu` is fine — guideline corpora are small (thousands of chunks). No GPU needed.
- Footprint: negligible (<2 GB RAM/VRAM). This part runs today on the B200 box or even CPU.

### C.2 Generation (local LLM via vLLM) — sized for ≤ 4× RTX 3090

**Chosen generator: `Qwen/Qwen3-8B`** (served by vLLM). Budget: 4× RTX 3090 = **96 GB VRAM**.

| Model | Precision | ~VRAM | GPUs | Notes |
|-------|-----------|-------|------|-------|
| **`Qwen/Qwen3-8B`** (chosen) | fp16/bf16 | ~16–18 GB | 1 | Strong general+reasoning 8B; fits one 3090 with room for the KV cache. Leaves 3 GPUs free. |
| `Qwen/Qwen3-8B` | fp16 + TP=2 | ~9 GB/GPU | 2 | Optional: split across 2 GPUs for longer context / bigger batch. |
| `Qwen/Qwen3-14B` (fallback-up) | AWQ 4-bit | ~12 GB | 1 | If we want more capacity and a quantized weight is available. |

**Qwen3 specifics that affect us:**
- Qwen3 has a **thinking / non-thinking** mode. For grounded RAG we default to **non-thinking**
  (`enable_thinking=false`, or append `/no_think`) for fast, deterministic, citation-clean answers; we can
  A/B thinking-mode on the eval set and report it.
- vLLM serves Qwen3 with a `--reasoning-parser qwen3` option if we *do* enable thinking; otherwise plain
  chat/completions is fine.
- Use the official chat template (vLLM applies it automatically from the HF repo).

Serve it with vLLM's OpenAI-compatible server:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --served-model-name qwen3-8b \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --port 8000
# exposes http://localhost:8000/v1/chat/completions
# (add --reasoning-parser qwen3 only if running in thinking mode)
```

Our `guideline_rag_tool` calls **`http://localhost:8000/v1/chat/completions`** (configurable via
`.env`: `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL`) — same urllib pattern the repo already uses, just
pointed at the local server instead of OpenAI. **No OpenAI key anywhere.**

### C.3 The B200 vs RTX 3090 caveat (must resolve before submission)

- This dev box has **8× B200** (sm_100). The pinned `torch 2.5.1 + cu121` / `vllm 0.18.1` are built for
  older archs and **may not run on B200** (B200 needs CUDA 12.8+ / newer torch+vllm).
- **Plan:**
  1. **Dev now:** build retrieval + grounding logic against the local vLLM endpoint. If pinned vLLM
     won't start on B200, dev-serve with a B200-compatible vLLM build (separate env) — the tool only
     depends on the HTTP endpoint, so the server build is swappable.
  2. **Submission target:** keep `environment.yml` (cu121 / vllm 0.18.1) as the canonical env and
     **validate end-to-end on an actual RTX 3090 machine** with the chosen 8B model. Record the exact
     `vllm serve` command + VRAM used in the PR / slides.
  3. Make the LLM endpoint **fully configurable** so grader hardware just changes `LOCAL_LLM_BASE_URL`.

### C.4 Offline fallback (no GPU / model down)

Keep a deterministic **extractive fallback**: if the LLM endpoint is unreachable, return the top
retrieved passages verbatim with citations and a clear "LLM unavailable — showing retrieved evidence
only" banner. This mirrors the repo's existing `_fallback_chat_answer` pattern and keeps the demo robust.

---

## Setup order (what we'll actually run)

1. Add RAG deps to `.venv` (track A). ← fast, do first
2. Build a tiny corpus + FAISS index (track C.1) and prove retrieval works — no LLM needed yet.
3. Stand up vLLM with an 8B model (track C.2); wire `guideline_rag_tool` to it.
4. Install Node + build frontend, add the Studio card (track B).
5. Validate the pinned `environment.yml` path on an RTX 3090 box.

See `03_TASKS.md` for the checklist.
