# 00 · Overview

## Goal

Build a **Clinical Guideline / Literature RAG** capability inside ChatClinic: the user asks a clinical
question (optionally alongside an uploaded source — a note, a FHIR bundle, an image report), and the
system retrieves the most relevant passages from a curated corpus of **clinical practice guidelines and
medical literature**, then produces an **evidence-grounded answer with inline citations** and an explicit
statement of uncertainty / what is *not* covered.

This deliberately reuses the repo's existing architecture (`architecture.md`):

> deterministic services create facts → evidence services fetch references → the LLM explains **only**
> from those structured inputs.

Our contribution is the **evidence-retrieval + grounding** layer the repo currently only stubs for VCF.

## Why this shape

- It fits the repo's "evidence-grounded clinical copilot" identity — minimal friction to integrate.
- The repo already has a `ReferenceItem` model and a `references.py` service; we make retrieval *real*
  (corpus-backed) instead of hardcoded.
- It demos cleanly: ask a question → see retrieved guideline passages → see a cited answer → see the
  verifier flag any unsupported claim.

## Scope (what we will and won't build)

**In scope**
- A small, openly-licensed **guideline/literature corpus** shipped as `sample test data` (required by the rubric).
- An offline **index builder** (chunk → embed → FAISS) + a runtime **retriever**.
- A **grounded-answer** tool (retrieve top-k → LLM answer with `[REF#]` citations → `ReferenceItem` list).
- A **citation verifier** tool (checks each cited claim is supported by its passage — hallucination guard).
- A Studio card to display retrieved passages + the cited answer.
- A short **evaluation** on a public medical-QA set (PubMedQA / MIRAGE-style) for the slides.

**Out of scope (state as limitations)**
- No persistent DB, no auth/PHI guardrails (inherited repo limitation).
- Not a production clinical decision system — research/education use only.
- Corpus is a curated sample, not an exhaustive guideline library.

## The three cooperating tools ("Multiple Tools" = 10 pts)

| Tool | Folder | Role |
|------|--------|------|
| **Guideline Index** | `plugins/guideline_index_tool/` | Build/refresh the FAISS index from the corpus (chunk + embed). Mostly an offline build, exposed as a tool for transparency. |
| **Guideline RAG** | `plugins/guideline_rag_tool/` | Retrieve top-k passages for a question, generate a cited, grounded answer, emit `ReferenceItem`s + Studio card. |
| **Citation Verifier** | `plugins/citation_verifier_tool/` | For each claim+citation in the answer, check the cited passage actually supports it; flag unsupported claims. Mirrors the Parkinson "Delta-Verifier" idea. |

> The verifier is the "Project Quality" differentiator — it turns a plain RAG demo into a
> faithfulness-checked one, directly addressing the #1 risk reviewers probe in Q&A (hallucinated citations).

## Mapping to the grading rubric (100 + 5 bonus)

| Criterion | Pts | How we earn it |
|-----------|-----|----------------|
| Multiple Tools | 10 | Three cooperating tools (index / RAG / verifier). |
| Project Quality | 40 | Real corpus-backed retrieval, citation faithfulness check, evaluation numbers, clean integration, sample data + reproducible env. |
| Presentation | 35 | Clear story: problem (hallucinated medical answers) → architecture → live demo → eval → limitations. |
| Q&A | 10 | Be ready on: embedding model choice, chunking, faithfulness/verifier, ≤4×RTX3090 footprint, failure modes. |
| Midterm bonus | 5 | N/A for us (final-period team). |

## Key constraints (from the TA notice + project PDF)

- Submit as a **PR to `bispl-create/chatclinic-multimodal`** from branch `medical-rag`, including **code + sample test data**.
- **Inference must run on ≤ 4× RTX 3090** (24 GB each → 96 GB total). See `01_SETUP.md` for the model budget.
- Base environment on the repo `environment.yml`.
- Slides + demo video → KLMS. Checkpoint (optional) → KLMS.
- Deadlines: **present Jun 18**, **code Jun 21**.
