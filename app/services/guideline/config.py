"""Paths and model defaults for the guideline RAG pipeline (env-overridable)."""
from __future__ import annotations

import os
from pathlib import Path

# .../app/services/guideline/config.py -> repo root is parents[3]
ROOT_DIR = Path(__file__).resolve().parents[3]


def corpus_dir() -> Path:
    return Path(os.getenv("GUIDELINE_CORPUS_DIR", str(ROOT_DIR / "examples" / "guidelines")))


def index_dir() -> Path:
    return Path(os.getenv("GUIDELINE_INDEX_DIR", str(ROOT_DIR / "data" / "guideline_index")))


def min_score_floor() -> float:
    """Hard minimum cosine similarity for a passage to be shown to the model.
    Passages below this are never used for grounding. Override via RAG_MIN_SCORE."""
    try:
        return float(os.getenv("RAG_MIN_SCORE", "0.5"))
    except ValueError:
        return 0.5


def embed_model_name() -> str:
    # Default is a biomedical sentence embedder (PubMedBERT, 768-dim) — domain-matched
    # for clinical text and light enough to embed quickly. Override via EMBED_MODEL
    # (e.g. all-MiniLM-L6-v2 for fastest local tests, or BAAI/bge-m3 for multilingual).
    return os.getenv("EMBED_MODEL", "NeuML/pubmedbert-base-embeddings")
