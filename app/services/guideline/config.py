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


def embed_model_name() -> str:
    # Production default is a biomedical-capable multilingual retriever; can be
    # overridden to a small model (e.g. all-MiniLM-L6-v2) for fast local tests.
    return os.getenv("EMBED_MODEL", "BAAI/bge-m3")
