# Skill Rationale

## Why this tool should exist
Clinical LLM answers are prone to confident but unsupported claims. ChatClinic already follows an
"evidence-grounded" principle (deterministic facts → references → LLM explains only from those). This
suite makes that real for free-text clinical questions: retrieval from a guideline corpus first, a
local Qwen3-8B answer grounded **only** on retrieved passages second, and an explicit faithfulness
check third. It fills the gap between the platform's stubbed reference layer and a usable evidence
copilot.

## Why the orchestrator should call it
- It is the right tool when the user asks a guideline/treatment question and wants a sourced answer,
  as opposed to general chat. Retrieval makes the answer auditable (every claim carries a `[REF#]`).
- It composes: `guideline_index_tool` → `guideline_rag_tool` → `citation_verifier_tool` form a clean
  pipeline the orchestrator can sequence, mirroring the platform's "facts → evidence → grounded answer"
  flow.
- The MCP gateway lets the orchestrator extend evidence beyond the local corpus on demand without new
  bespoke integrations — any conforming external MCP server is reachable via config.

## Why approval is or is not required
Approval is **not** required: the tools are read-only (retrieval + generation over an internal model),
produce no file writes, persist no PHI, and surface uncertainty explicitly. The citation verifier
further reduces risk by flagging unsupported claims rather than presenting them as fact.

## What educational value it adds
- Demonstrates a full **medical RAG** stack: chunking, embeddings, FAISS retrieval, grounded
  generation, and **faithfulness evaluation** — directly tied to the course's Medical RAG module.
- Demonstrates **agentic / MCP** patterns (server-to-server, agent2agent) on top of RAG.
- Shows the safety discipline expected in clinical AI: grounding, explicit "insufficient evidence",
  and citation verification.
