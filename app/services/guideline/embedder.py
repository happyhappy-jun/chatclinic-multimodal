"""Local sentence-embedding wrapper (SentenceTransformer), normalized for cosine."""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

_ENCODE_LOCK = threading.Lock()


@lru_cache(maxsize=2)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    # CPU is fine for retrieval; avoids GPU/driver coupling for the embedder.
    return SentenceTransformer(name, device="cpu")


def embed_texts(texts: list[str], model_name: str, *, batch_size: int = 32) -> np.ndarray:
    """Return L2-normalized float32 embeddings (cosine == inner product)."""
    model = _load_model(model_name)
    with _ENCODE_LOCK:
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    return np.asarray(vectors, dtype="float32")
