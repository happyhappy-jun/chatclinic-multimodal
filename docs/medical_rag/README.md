# Medical RAG — Clinical Guideline & Literature RAG (AI619 Team Project)

Team: 윤병준, 이현석, 김태영 · Branch: `medical-rag` · Repo: `bispl-create/chatclinic-multimodal`

This folder is the team's planning + design space. Read in order:

1. **[00_OVERVIEW.md](00_OVERVIEW.md)** — what we're building, why, scope, and how it maps to the grading rubric.
2. **[01_SETUP.md](01_SETUP.md)** — environment setup for all three tracks (backend ✓, frontend, model/RAG stack), incl. the B200 → RTX 3090 strategy.
3. **[02_PLUGIN_DESIGN.md](02_PLUGIN_DESIGN.md)** — the tool design: corpus, embeddings + FAISS retriever, grounded answer with citations, the verifier, and exact integration steps in this codebase.
4. **[03_TASKS.md](03_TASKS.md)** — actionable checklist + milestones toward **Jun 18 (present)** and **Jun 21 (code due)**.
5. **[04_DEMO.md](04_DEMO.md)** — how to run the full stack (vLLM → backend → `/guideline` page).
6. **[05_MCP.md](05_MCP.md)** — MCP gateway (server-to-server / agent2agent over Streamable HTTP).
7. **[06_EVAL.md](06_EVAL.md)** — PubMedQA (RAG lift + faithfulness) and MIRAGE results.
8. **[07_DEPLOY.md](07_DEPLOY.md)** — run on another GPU box (e.g. RTX 3090): no-SLURM serve + `run_all.sh`.
9. **[08_SGLANG.md](08_SGLANG.md)** — serve Qwen3-8B with SGLang instead of vLLM (pip install in the conda env).

## TL;DR

We add a small **pipeline of tools** to ChatClinic that turns a clinical question into an
**evidence-grounded answer with inline citations**, retrieved from a corpus of clinical guidelines /
literature. Three cooperating tools (index → retrieve+ground → verify) satisfy the
"Multiple Tools" rubric and follow the repo's core principle: *deterministic retrieval first,
LLM explanation second, never hide uncertainty.*
