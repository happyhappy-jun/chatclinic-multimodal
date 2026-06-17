# Demo Scenario — Medical RAG

A ~10-minute live demo that contrasts the **original ChatClinic** with **our PR**, then walks the main
clinical scenario end-to-end. Each act lists what to type, what to expect, and the **before → after**.

## Setup (before you present)

```bash
# Terminal 1 — model on a free GPU
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_qwen3_local.sh
# Terminal 2 — app + MCP + frontend
bash scripts/run_all.sh
# laptop — tunnel both ports
ssh -L 3000:localhost:3000 -L 8001:localhost:8001 junyoon@alin14
```
Open **http://localhost:3000/** (main workspace) and **/guideline** (RAG page). Confirm PubMed is up:
`curl -s -X POST localhost:8001/api/v1/mcp/tools` → `servers: ['pubmed','pubmed-lite']`.

---

## The gap we close

| | Original ChatClinic (`main`) | Our PR (`medical-rag`) |
|---|---|---|
| Ask a clinical question | ❌ "Please upload a source file first" | ✅ Answered with citations, no upload |
| Knowledge base / RAG | ❌ none | ✅ guideline corpus + FAISS retrieval |
| Citations | ❌ | ✅ inline `[REF#]` to real passages/PMIDs |
| Trust / hallucination check | ❌ | ✅ per-claim faithfulness verifier |
| Out-of-scope handling | ❌ guesses | ✅ refuses ("insufficient evidence", 0.5 floor) |
| Live literature | ❌ | ✅ live PubMed via MCP federation |
| Agent interoperability | ❌ | ✅ our RAG exposed as an MCP server (agent2agent) |
| Knowledge base management | ❌ | ✅ upload/list/index docs from the UI |
| LLM | OpenAI API | ✅ local Qwen3-8B (no data leaves the box) |

---

## Act 1 — The "before" (the limitation)

**In the main UI, with nothing uploaded, type a plain clinical question:**
```
What is the first-line vasopressor in septic shock?
```
➡️ **Original behavior (still the default for non-tool chat):** *"Please upload a source file first…"* —
ChatClinic is upload-centric; it can't answer a free clinical question.

> **Say:** "ChatClinic is built around uploaded files. A clinician with a question and no file is stuck.
> That's the gap we close."

## Act 2 — Core RAG: grounded, cited answer (NEW)

**Now use our tool — still no upload:**
```
@guideline first-line vasopressor and MAP target in septic shock
```
➡️ A grounded answer appears in chat with inline `[REF1]` citations, and the **Guideline RAG card** opens
in Studio (right panel) with the answer + retrieved passages.

> **Say:** "Same blank workspace — `@guideline` answers from our guideline corpus, every claim cited.
> Deterministic retrieval first, the local model grounds on it second."

## Act 3 — Trust: citation faithfulness (NEW)

**Point at the Studio card's faithfulness badge and the per-claim ✓/✕ list.**
➡️ e.g. *faithfulness 100% (3/3)* — each cited claim is verified against its source passage.

> **Say:** "We don't just cite — we *verify*. A second model checks each claim against the passage it
> cites. This is the hallucination guard, the #1 risk in clinical LLMs."

## Act 4 — Safety: refuse out-of-scope (NEW)

```
@guideline what chemotherapy regimen treats stage III colon cancer
```
➡️ *"Insufficient evidence in the provided guidelines."* — no passage clears the **0.5 cosine floor**, so
the model refuses instead of guessing.

> **Say:** "Off-corpus questions are refused, not hallucinated. A 0.5 relevance floor means weak matches
> never reach the model."

## Act 5 — Live evidence: PubMed federation (NEW)

```
@guideline +pubmed latest evidence on vasopressor choice in septic shock
```
➡️ The card now shows **`external MCP: pubmed`** passages with real **PMIDs** (e.g. the VASST trial),
merged with the local guideline, all cited.

> **Say:** "`+pubmed` fans out to a live PubMed server over MCP, pulls real abstracts, re-ranks them with
> our biomedical embedder, and grounds on guideline + current literature together."

## Act 6 — Manage the knowledge base (NEW)

Go to **/guideline**. Show the **Knowledge base** panel (lists indexed docs). Click **Add to corpus**,
upload a guideline PDF/markdown. Watch it re-index, then ask a question only that doc can answer.
➡️ The new document appears in the list and is **immediately retrievable**.

> **Say:** "The knowledge base is editable from the browser — drop in a guideline, it's indexed and
> searchable in one step."

## Act 7 — Interoperability: our RAG as an MCP server (NEW)

```
@mcp
```
➡️ Lists tools exposed by federated MCP servers (`pubmed`, `pubmed-lite`). Then call one from chat:
```
@mcp call pubmed search_literature query=septic shock norepinephrine
```
➡️ Real PubMed results (PMIDs) returned through the gateway. (Same as `POST /api/v1/mcp/call`.)

> **Say:** "ChatClinic is now both an MCP client *and* server — our RAG tools are callable by any external
> agent. This is the agentic/MCP piece of the course."

## Act 8 (optional) — Source-grounded answer

Upload a short clinical note (`.txt`) in the main UI, then:
```
@guideline what blood pressure target applies to this patient
```
➡️ The answer grounds in **guidelines + the patient's note** (chat note says *grounded in note.txt*).

> **Say:** "When a patient note or FHIR bundle is loaded, the same tool grounds in that record too — the
> native ChatClinic source-attached pattern."

---

## Closing line

> "From an upload-only genomics workspace, we added an evidence-grounded clinical copilot: cited answers,
> verified faithfulness, safe refusals, live PubMed, agent interoperability, and an editable knowledge
> base — all local, all on ≤4× RTX 3090."

## One-screen prompt cheat-sheet
```
What is the first-line vasopressor in septic shock?              # Act 1: old limitation
@guideline first-line vasopressor and MAP target in septic shock # Act 2: cited answer + card
@guideline what chemotherapy regimen treats stage III colon cancer  # Act 4: refusal
@guideline +pubmed latest evidence on vasopressor choice in septic shock  # Act 5: live PubMed
@mcp                                                             # Act 7: federation
@guideline help                                                  # discoverability
```
