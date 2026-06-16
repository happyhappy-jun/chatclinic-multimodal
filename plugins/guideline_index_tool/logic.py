"""guideline_index_tool — build the FAISS retrieval index from the corpus.

Runnable as a backend tool (``execute``) or as a CLI:
    python -m plugins.guideline_index_tool.logic
"""
from __future__ import annotations

from app.services.guideline.pipeline import build_guideline_index


def execute(payload: dict) -> dict:
    return build_guideline_index(
        corpus_dir=payload.get("corpus_dir"),
        index_dir=payload.get("index_dir"),
        embed_model=payload.get("embed_model"),
        chunk_size=int(payload.get("chunk_size", 220)),
        overlap=int(payload.get("chunk_overlap", 40)),
    )


if __name__ == "__main__":
    import json

    print(json.dumps(execute({}), indent=2, ensure_ascii=False))
