# Medical RAG — Feature Catalog

Everything this PR (`medical-rag`) adds to ChatClinic. Team: 윤병준, 이현석, 김태영.

**One line:** an evidence-grounded clinical **guideline / literature RAG** with a **citation-faithfulness
verifier** and an **MCP gateway** (server-to-server) that federates to **live PubMed** — fully local
(local embeddings + local Qwen3-8B, no OpenAI), integrated natively into the ChatClinic workspace.

---

## 1. Clinical Guideline RAG (core)

Ask a clinical question → retrieve the most relevant passages from a guideline/literature corpus →
generate a grounded answer with inline `[REF#]` citations. Deterministic retrieval first, LLM second.

- **Retriever:** local sentence embeddings + **FAISS** (exact cosine). Default embedder
  **`NeuML/pubmedbert-base-embeddings`** (biomedical, 768-dim).
- **Generator:** local **Qwen3-8B** served by vLLM (or SGLang) — OpenAI-compatible, **no OpenAI API**.
- **Grounding guardrail:** answers only from retrieved passages; returns *"Insufficient evidence in the
  provided guidelines."* when the corpus doesn't cover the question.
- **Relevance floor:** passages below **cosine 0.5** are never shown to the model (`RAG_MIN_SCORE`),
  so weak/off-topic context can't produce ungrounded answers.
- **Robust fallback:** if the LLM endpoint is unreachable, returns the top retrieved passages verbatim
  (extractive) instead of failing.

**Tools:** `guideline_index_tool`, `guideline_rag_tool` · **Endpoint:** `POST /api/v1/guideline-rag/run`

## 2. Citation Faithfulness Verifier (hallucination guard)

For each cited sentence in the answer, checks whether the cited passage actually supports it, and
reports a **faithfulness score** (`supported / total`). Unsupported claims are surfaced, not hidden.
LLM NLI check with a lexical-overlap fallback when the LLM is down.

**Tool:** `citation_verifier_tool` · **Endpoint:** `POST /api/v1/citation-check/run`

## 3. MCP Gateway — server-to-server / agent2agent

ChatClinic becomes both an **MCP server** and an **MCP client**, over **Streamable HTTP** (official `mcp` SDK).

- **Our RAG as an MCP server** (`mcp_servers/chatclinic_rag_server.py`) — any agent/MCP client can call
  `retrieve_guidelines`, `guideline_rag`, `verify_citations`.
- **MCP client / federation** (`app/services/mcp_gateway.py`) — discovers and invokes tools on external
  MCP servers; configured in `mcp_servers.json`. Unreachable servers are reported, never crash a request.

**Tool:** `mcp_federation_tool` · **Endpoints:** `POST /api/v1/mcp/tools` (discover), `POST /api/v1/mcp/call` (invoke)

## 4. Live PubMed federation

`external_evidence=true` performs real retrieval-augmented generation over **live PubMed**:

1. PubMed MCP server (`mcp_servers/pubmed_server.py`) runs NCBI **esearch → efetch** for real
   titles + abstracts (with a key-term fallback query).
2. The gateway fetches a candidate pool (~20) and **semantically re-ranks** it with our embedder
   (PubMed recall + embedder precision).
3. The top hits merge with local guideline passages (continuing `REF#`); Qwen3 grounds and cites them
   with real **PMIDs**.

Offline demo fallback: `mcp_servers/pubmed_lite_server.py`. No NCBI key required (set `PUBMED_EMAIL`).

## 5. Native ChatClinic integration

- **`@guideline <question>` in the main chat** — works with **no uploaded source** (source-less clinical
  Q&A), or grounds in an active **text note / FHIR bundle** when one is loaded (`source_types: text, fhir`,
  wired via the `direct_chat` executor pattern).
- **`@guideline +pubmed <question>`** — enables live-PubMed federation from chat.
- **Studio renderer** `guideline_rag` — a card showing the cited answer, faithfulness badge, per-claim
  ✓/✕, and retrieved passages with scores/sources.
- **`@help` / `@guideline help`** discoverability; orchestration rules added to `SKILL.md`.

## 6. Corpus management UI (`/guideline` page)

A self-contained page that doubles as a corpus manager:

- **Ask** — question box, `top_k`, **"live PubMed"** toggle, cited answer + faithfulness + passages.
- **Add to corpus** — upload `.md` / `.txt` / `.pdf`; saves to the corpus dir and **rebuilds the FAISS
  index** in one step. New docs are immediately retrievable.
- **Knowledge base panel** — lists every indexed document (title, source, chunk count) + embedder & totals.

**Endpoints:** `POST /api/v1/guideline/upload`, `GET /api/v1/guideline/docs`, `POST /api/v1/guideline/index`

## 7. Evaluation harness

`eval/evaluate.py` — reproducible benchmarks (writeup in [06_EVAL.md](06_EVAL.md)):

- **PubMedQA:** closed-book Qwen3-8B vs our RAG → **+30 pts** (21.5% → 51.5%), citation **faithfulness 91.7%**.
- **MIRAGE:** Qwen3-8B zero-shot macro **62.0%** across 5 subsets.

## 8. Deployment & ops

- **Serving** — `scripts/serve_qwen3_local.sh` (vLLM), `scripts/serve_qwen3_sglang.sh` (SGLang),
  `scripts/serve_qwen3_vllm.sbatch` (SLURM). Auto-handle a `noexec` `/tmp`, `~/.local` shadowing, and the
  bundled prometheus-instrumentator bug.
- **One-command bring-up** — `scripts/run_all.sh` (index → backend → MCP servers → frontend, with health
  checks) and `scripts/run_mcp_servers.sh`.
- **Portable** — every host/path/model/device is env-driven; runs on ≤4× RTX 3090.
- **Observability** — `[guideline]` progress logging during embedding / index build.
- Full guide: [07_DEPLOY.md](07_DEPLOY.md) · SGLang: [08_SGLANG.md](08_SGLANG.md) · MCP: [05_MCP.md](05_MCP.md).

---

## API reference (added endpoints)

| Method & path | Purpose |
|---|---|
| `POST /api/v1/guideline-rag/run` | Grounded, cited answer (`external_evidence`, `source_context`, `top_k`, `min_score`, `verify`) |
| `POST /api/v1/citation-check/run` | Faithfulness check of an answer vs passages |
| `POST /api/v1/guideline/index` | Build the FAISS index from the corpus |
| `POST /api/v1/guideline/upload` | Ingest a `.md`/`.txt`/`.pdf` into the corpus + rebuild |
| `GET  /api/v1/guideline/docs` | List indexed documents |
| `POST /api/v1/mcp/tools` | Discover tools on federated external MCP servers |
| `POST /api/v1/mcp/call` | Invoke a tool on an external MCP server |

## Configuration (env vars)

| Var | Default | Meaning |
|---|---|---|
| `LOCAL_LLM_BASE_URL` | `http://localhost:8000/v1` | Local LLM server (vLLM `:8000`, SGLang `:30000`) |
| `LOCAL_LLM_MODEL` | `qwen3-8b` | Served model name |
| `LOCAL_LLM_THINKING` | `false` | Qwen3 thinking mode |
| `EMBED_MODEL` | `NeuML/pubmedbert-base-embeddings` | Retrieval embedder |
| `EMBED_DEVICE` | `cpu` | `cpu` \| `cuda` \| `cuda:N` for the embedder |
| `RAG_MIN_SCORE` | `0.5` | Hard cosine floor for passages shown to the model |
| `GUIDELINE_CORPUS_DIR` | `examples/guidelines` | Corpus directory |
| `GUIDELINE_INDEX_DIR` | `data/guideline_index` | FAISS index directory |
| `PUBMED_EMAIL` / `NCBI_API_KEY` | — | NCBI etiquette / higher rate limit (key optional) |
| `MCP_SERVERS_CONFIG` / `MCP_SERVERS` | `mcp_servers.json` | External MCP server config |

## File map

```
plugins/{guideline_index_tool, guideline_rag_tool, citation_verifier_tool, mcp_federation_tool}/
app/services/guideline/{config, corpus, embedder, store, pipeline}.py
app/services/{local_llm, mcp_gateway}.py
app/main.py, app/models.py            # endpoints + schemas
mcp_servers/{chatclinic_rag_server, pubmed_server, pubmed_lite_server}.py
mcp_servers.json
webapp/app/guideline/page.tsx         # demo + corpus manager
webapp/app/components/GuidelineRagCard.tsx   # Studio card
webapp/app/page.tsx                   # @guideline in main chat
skills/chatgenome-orchestrator/SKILL.md      # orchestration
examples/guidelines/, examples/guideline_sample_queries.md   # sample data
eval/evaluate.py                      # benchmarks
scripts/{serve_qwen3_local, serve_qwen3_sglang, run_all, run_mcp_servers}.sh
docs/medical_rag/                     # design / setup / demo / mcp / eval / deploy / sglang
submission/                           # skill patch, rationale, background papers
```
