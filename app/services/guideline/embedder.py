"""Local sentence-embedding wrapper (SentenceTransformer), normalized for cosine."""
from __future__ import annotations

import os
import threading
from functools import lru_cache

import numpy as np

_ENCODE_LOCK = threading.Lock()


def _resolve_device() -> str:
    """EMBED_DEVICE env: 'cpu' (default), 'cuda', or 'cuda:N'. Falls back to CPU."""
    device = os.getenv("EMBED_DEVICE", "cpu").strip() or "cpu"
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
    return device


@lru_cache(maxsize=2)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer

    # Default CPU keeps the embedder off the LLM's GPU; set EMBED_DEVICE=cuda:N
    # to put a heavy retriever (e.g. bge-m3) on a free GPU for a big speedup.
    return SentenceTransformer(name, device=_resolve_device())


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
