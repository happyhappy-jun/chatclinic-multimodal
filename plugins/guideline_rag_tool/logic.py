"""guideline_rag_tool — retrieve guideline passages and produce a cited, grounded answer."""
from __future__ import annotations

from app.services.guideline.pipeline import run_guideline_rag


def execute(payload: dict) -> dict:
    question = (payload.get("question") or "").strip()
    if not question:
        raise ValueError("guideline_rag_tool requires a non-empty 'question'.")
    return run_guideline_rag(
        question,
        top_k=int(payload.get("top_k", 6)),
        min_score=float(payload.get("min_score", 0.2)),
        source_context=payload.get("source_context"),
        index_dir=payload.get("index_dir"),
        verify=bool(payload.get("verify", True)),
        external_evidence=bool(payload.get("external_evidence", False)),
        external_max=int(payload.get("external_max", 3)),
    )
