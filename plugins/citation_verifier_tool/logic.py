"""citation_verifier_tool — per-claim faithfulness check over a grounded answer.

For each cited sentence in ``draft_answer``, checks whether the cited passage
actually supports it (local LLM NLI; lexical fallback if the LLM is down).
"""
from __future__ import annotations

from app.services.guideline.pipeline import verify_citations


def execute(payload: dict) -> dict:
    answer = (payload.get("draft_answer") or "").strip()
    passages = payload.get("passages") or []
    if not answer:
        raise ValueError("citation_verifier_tool requires a non-empty 'draft_answer'.")
    return verify_citations(answer, passages)
