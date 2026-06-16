# 03 · Tasks & Milestones

Deadlines: **present Jun 18** · **code (PR) due Jun 21**. Today: Jun 15.

Status key: ✅ done · 🔜 next · ⬜ todo

## Track A — Environment
- ✅ Clone repo, branch `medical-rag`, `.env`, base `.venv`, backend boots (40 routes).
- ✅ Plan docs (`docs/medical_rag/`).
- 🔜 Add RAG deps to `.venv` (`sentence-transformers`, `FlagEmbedding`, `faiss-cpu`, `pypdf`).
- ⬜ Record RAG deps in `requirements-rag.txt` and extend `environment.yml` notes for reproducibility.

## Track C — Retrieval (no GPU needed)
- ⬜ Assemble sample corpus under `examples/guidelines/` (openly-licensed; keep title/source/url).
- ⬜ `guideline_index_tool`: chunk + embed (`BAAI/bge-m3`) + FAISS build → `data/guideline_index/`.
- ⬜ Ship a small prebuilt sample index as test data.
- ⬜ `guideline_rag_tool` retrieval half: question → top-k passages (prove retrieval quality before LLM).

## Track C — Local LLM (Qwen3-8B via vLLM)
- ⬜ Stand up vLLM serving `Qwen/Qwen3-8B` (`--served-model-name qwen3-8b`, TP=1) on a GPU.
  - On the B200 dev box: if pinned `vllm 0.18.1` won't start, dev-serve with a B200-compatible vLLM build.
- ⬜ `app/services/local_llm.py`: urllib client → `/v1/chat/completions`, non-thinking mode, fallback on error.
- ⬜ `guideline_rag_tool` generation half: grounded prompt → cited `draft_answer` + `ReferenceItem`s.
- ⬜ `citation_verifier_tool`: per-claim NLI check → faithfulness %.
- ⬜ Extractive fallback when LLM endpoint is down.

## Track B — Frontend & Studio card
- ⬜ Install Node (nvm) + `npm install` in `webapp/`.
- ⬜ Studio renderer `guideline_rag` (passages + cited answer + verifier badges).
- ⬜ `npm run build` passes (PR gate).

## Backend wiring
- ⬜ Models in `app/models.py` (see `02_PLUGIN_DESIGN.md`).
- ⬜ Endpoints `POST /api/v1/guideline-rag/run`, `/api/v1/citation-check/run` in `app/main.py`.
- ⬜ Register tools; update `skills/chatgenome-orchestrator/SKILL.md` help + policy.
- ⬜ `python3 -m py_compile app/main.py app/models.py app/services/*.py plugins/*/logic.py`.

## Evaluation & deliverables
- ⬜ Eval on PubMedQA / MIRAGE subset: retrieval hit@k, RAG accuracy lift, **faithfulness w/ vs w/o verifier**.
- ⬜ Validate full pinned-env path on an **RTX 3090** box; record `vllm serve` cmd + VRAM.
- ⬜ Slides (problem → architecture → live demo → eval → limitations) → KLMS.
- ⬜ Demo video → KLMS. Optional checkpoint/index → KLMS.
- ⬜ Open PR `medical-rag → main` with code + sample data.

## Suggested split (3 people)
- **Retrieval owner:** corpus + index tool + embedding/chunking + retrieval eval.
- **Generation owner:** vLLM/Qwen3 serving + `local_llm.py` + RAG prompt + verifier.
- **Integration/UI owner:** models/endpoints wiring + Studio card + frontend build + slides/demo.

## Critical-path order
1. RAG deps → 2. corpus + index + retrieval working → 3. Qwen3 vLLM up + grounded answer →
4. verifier → 5. UI card + frontend build → 6. eval → 7. RTX 3090 validation → 8. PR + slides/video.
