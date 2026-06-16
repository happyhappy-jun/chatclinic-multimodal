"""FAISS index persistence: build / load / search over guideline chunks."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from app.services.guideline.corpus import Chunk

INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.jsonl"
META_FILE = "meta.json"


def build_index(index_dir: Path, chunks: list[Chunk], embeddings: np.ndarray, embed_model: str) -> dict[str, Any]:
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)
    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine
    index.add(embeddings)
    faiss.write_index(index, str(index_dir / INDEX_FILE))

    with (index_dir / CHUNKS_FILE).open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    meta = {"embed_model": embed_model, "dim": dim, "n_chunks": len(chunks)}
    (index_dir / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_chunks(index_dir: Path) -> list[dict[str, Any]]:
    path = index_dir / CHUNKS_FILE
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_meta(index_dir: Path) -> dict[str, Any]:
    path = index_dir / META_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_index(index_dir: Path):
    import faiss

    path = index_dir / INDEX_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {path}. Build it first with the guideline_index_tool."
        )
    return faiss.read_index(str(path))


def search(index_dir: Path, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    index = load_index(index_dir)
    scores, ids = index.search(query_vec, top_k)
    return [(int(i), float(s)) for i, s in zip(ids[0].tolist(), scores[0].tolist()) if i >= 0]
