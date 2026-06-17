"""Local sentence-embedding wrapper (SentenceTransformer), normalized for cosine."""
from __future__ import annotations

import os
import sys
import threading
import time
from functools import lru_cache

import numpy as np

_ENCODE_LOCK = threading.Lock()


def _log(msg: str) -> None:
    print(f"[guideline] {msg}", file=sys.stderr, flush=True)


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
    device = _resolve_device()
    _log(f"loading embedder '{name}' on {device} (first use downloads the model) ...")
    started = time.time()
    model = SentenceTransformer(name, device=device)
    _log(f"embedder ready on {device} in {time.time() - started:.1f}s")
    return model


def embed_texts(
    texts: list[str], model_name: str, *, batch_size: int = 32, progress: bool = False
) -> np.ndarray:
    """Return L2-normalized float32 embeddings (cosine == inner product).

    Set ``progress=True`` (used by the index builder) to log timing and show a
    live progress bar — handy for large corpora where embedding is the slow step.
    """
    model = _load_model(model_name)
    items = list(texts)
    with _ENCODE_LOCK:
        if progress:
            _log(f"embedding {len(items)} chunks (batch_size={batch_size}) ...")
            started = time.time()
        vectors = model.encode(
            items,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        if progress:
            took = time.time() - started
            rate = len(items) / took if took else 0
            _log(f"embedded {len(items)} chunks in {took:.1f}s ({rate:.0f} chunks/s)")
    return np.asarray(vectors, dtype="float32")
