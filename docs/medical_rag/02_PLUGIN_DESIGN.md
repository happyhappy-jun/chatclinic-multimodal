# 02 · Plugin Design & Integration

Concrete design for the three tools, written against how this codebase actually works.

## How a tool runs here (verified from the repo)

- A plugin = `plugins/<tool>/{tool.json, logic.py}`. `tool.json.entrypoint` =
  `plugins.<tool>.logic:execute`; backend imports and calls `execute(payload: dict) -> dict`
  (`app/services/tool_runner.py` / `plugin_runtime.py`).
- Evidence is modeled by `ReferenceItem(id, title, source, url, note)` in `app/models.py`; the existing
  `grounded_summary_tool` already consumes `ReferenceItem`s and returns `{"draft_answer": ...}`.
- The LLM is currently called in `app/services/chat.py` via `urllib` to OpenAI's `/v1/responses`
  (`_call_openai_for_source`, keyed on `OPENAI_API_KEY`, with `_fallback_chat_answer` when absent).
  **We replace that call path with a local vLLM client** (see "LLM client" below).
- Studio cards: register a renderer key in `webapp/app/components/studioRenderers.tsx` and add the card
  component in `customStudioRenderers.tsx`; `tool.json.direct_chat.studio.renderer` names the key.

## Data model additions (`app/models.py`)

```python
class RetrievedPassage(BaseModel):
    ref_id: str                 # "REF1"
    doc_title: str
    source: str                 # guideline name / journal
    url: str | None = None
    chunk_id: str
    text: str
    score: float                # similarity

class GuidelineRagRequest(BaseModel):
    question: str
    top_k: int = 6
    source_context: str | None = None   # optional: text from an uploaded source
    min_score: float = 0.30

class GuidelineRagResponse(BaseModel):
    draft_answer: str                    # contains inline [REF#] citations
    references: list[ReferenceItem]
    passages: list[RetrievedPassage]
    uncertainty: str                     # explicit "not covered / low confidence"
    verifier: "CitationCheckResponse | None" = None

class CitationClaim(BaseModel):
    claim: str
    ref_id: str
    supported: bool
    evidence_span: str | None = None
    note: str | None = None

class CitationCheckRequest(BaseModel):
    draft_answer: str
    passages: list[RetrievedPassage]

class CitationCheckResponse(BaseModel):
    claims: list[CitationClaim]
    unsupported_count: int
    faithfulness: float          # supported / total
```

## Tool 1 — `guideline_index_tool`

**Job:** turn the corpus into a FAISS index (offline-ish, idempotent).

- Input: corpus dir (`data/guidelines/` — markdown/txt/pdf), chunk size, overlap, embedding model id.
- Steps: load docs → clean → chunk (≈512 tokens, ≈64 overlap, respect headings) → embed (local model)
  → write `faiss.index` + `chunks.jsonl` (chunk_id, doc_title, source, url, text) under
  `data/guideline_index/`.
- Output: `{"n_docs":, "n_chunks":, "index_path":, "embed_model":}`.
- Implemented as a tool **and** runnable as a script (`python -m plugins.guideline_index_tool.logic build`)
  so the index can be rebuilt in CI / on the grader box. The index files are data artifacts (gitignore
  the large ones; ship a small prebuilt sample index as test data).

## Tool 2 — `guideline_rag_tool`  (the centerpiece)

**Job:** question → grounded, cited answer.

Flow (deterministic retrieval first, LLM second — per `architecture.md`):

1. **Embed** the question with the same local embedding model.
2. **Retrieve** top-k chunks from FAISS; drop below `min_score`; assign `REF1..REFk`.
3. **Build context** = numbered passages (+ optional `source_context` from an uploaded note/FHIR).
4. **Generate** via the **local LLM**: a strict prompt — *answer only from the numbered passages, cite
   every claim with [REF#], say "insufficient evidence" if not covered.*
5. **Assemble** `GuidelineRagResponse`: `draft_answer`, `references` (`ReferenceItem` per used passage),
   `passages`, `uncertainty`.
6. Optionally call Tool 3 inline and attach `verifier`.

`tool.json` sketch:

```json
{
  "name": "guideline_rag_tool",
  "entrypoint": "plugins.guideline_rag_tool.logic:execute",
  "description": "Evidence-grounded answer to a clinical question, retrieved from a guideline/literature corpus with inline citations.",
  "task": "guideline-rag",
  "modality": "clinical",
  "approval_required": false,
  "source": "plugin",
  "routing": { "trigger_keywords": ["guideline", "evidence", "rag", "literature"] },
  "direct_chat": {
    "endpoint": "guideline-rag",
    "requested_view": "guideline_rag",
    "studio": { "renderer": "guideline_rag" }
  },
  "help": {
    "summary": "Ask a clinical question; get a cited, guideline-grounded answer.",
    "options": [
      {"name": "question", "type": "str", "description": "Clinical question"},
      {"name": "top_k", "type": "int", "description": "Passages to retrieve (default 6)"}
    ],
    "examples": ["@guideline help", "@guideline first-line treatment for community-acquired pneumonia"]
  }
}
```

## Tool 3 — `citation_verifier_tool`  (faithfulness / hallucination guard)

**Job:** for each (claim, [REF#]) in the answer, check the cited passage supports it.

- Split `draft_answer` into claim sentences with their cited `ref_id`s.
- For each claim, ask the **local LLM** (NLI-style): does `passages[ref_id]` entail the claim?
  → `supported: bool` + `evidence_span`.
- Return `CitationCheckResponse` with `faithfulness = supported/total` and `unsupported_count`.
- The Studio card highlights unsupported claims in red. This is our **quality differentiator** and the
  natural Q&A talking point.

## LLM client (local, shared)

Add `app/services/local_llm.py` (new, small):

```python
# Reads LOCAL_LLM_BASE_URL (default http://localhost:8000/v1) and LOCAL_LLM_MODEL from env.
# POSTs to {base}/chat/completions via urllib (same style as the existing OpenAI call).
# On connection failure -> raise LocalLLMUnavailable so callers fall back to extractive mode.
def chat(messages: list[dict], *, temperature=0.0, max_tokens=800) -> str: ...
```

`.env` additions:

```
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL=qwen3-8b            # --served-model-name from the vLLM server (Qwen/Qwen3-8B)
LOCAL_LLM_THINKING=false            # non-thinking mode for clean, fast grounded answers
EMBED_MODEL=BAAI/bge-m3
GUIDELINE_INDEX_DIR=data/guideline_index
```

> Keep the OpenAI code path untouched (other source types use it); our tools simply use
> `local_llm.chat()` instead. This isolates our change and avoids regressing existing tools.

## Integration checklist (from `docs/TOOL_PLUGIN_GUIDE.md`)

For each tool:

1. `plugins/<tool>/tool.json` + `logic.py` (`execute(payload)`).
2. Request/response models in **`app/models.py`** (above).
3. Endpoint in **`app/main.py`** only if a new behavior type is needed (RAG is — add
   `POST /api/v1/guideline-rag/run` and `/api/v1/citation-check/run`). Index tool can be script-only.
4. Studio renderer: register key `guideline_rag` in `studioRenderers.tsx`; card component in
   `customStudioRenderers.tsx` (passages list + cited answer + verifier badges).
5. Orchestrator policy in **`skills/chatgenome-orchestrator/SKILL.md`** (add the `@guideline` tool to the
   help message + when to recommend it).
6. Tests: `python3 -m py_compile app/main.py app/models.py app/services/*.py plugins/*/logic.py`;
   `cd webapp && npm run build`; manual `@guideline help`, `@guideline <question>`, card render.

## Corpus & sample test data (rubric requires sample data)

- Ship a **small curated corpus** of openly-licensed guideline/literature docs under
  `examples/guidelines/` (e.g., a handful of disease-management guideline excerpts or open-access review
  abstracts), plus a **prebuilt small FAISS index** so the demo runs without a rebuild.
- Keep provenance (title, source, url) per doc for honest citations.
- Avoid restricted data (no MIMIC/PHI in the repo).

## Evaluation (for slides + Project Quality)

- Run on a public set: **PubMedQA** (yes/no/maybe) or a **MIRAGE**-style guideline-QA subset.
- Report: retrieval hit-rate@k, answer accuracy with vs without retrieval (RAG lift), and
  **faithfulness %** from the verifier (with vs without it). The faithfulness delta is the headline number.
