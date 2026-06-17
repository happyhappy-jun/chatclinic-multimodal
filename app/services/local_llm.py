"""Local LLM client for the medical-rag tools.

Talks to a local OpenAI-compatible server (vLLM or SGLang serving ``Qwen/Qwen3-8B``)
at ``LOCAL_LLM_BASE_URL``. No OpenAI API is used anywhere in the RAG pipeline.

Env:
- ``LOCAL_LLM_BASE_URL``      default ``http://localhost:8000/v1`` (SGLang: ``:30000/v1``)
- ``LOCAL_LLM_MODEL``         default ``qwen3-8b`` (the server's ``--served-model-name``)
- ``LOCAL_LLM_THINKING``      default ``false`` (Qwen3 thinking mode toggle)
- ``LOCAL_LLM_TIMEOUT_SECONDS`` default ``120``
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any


class LocalLLMUnavailable(RuntimeError):
    """Raised when the local LLM endpoint cannot be reached or returns garbage.

    Callers should catch this and fall back to a deterministic extractive answer
    so the demo stays robust when no GPU/server is running.
    """


def _base_url() -> str:
    return os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")


def _model() -> str:
    return os.getenv("LOCAL_LLM_MODEL", "qwen3-8b")


def _timeout() -> float:
    return float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120"))


def _thinking_enabled() -> bool:
    return os.getenv("LOCAL_LLM_THINKING", "false").strip().lower() in {"1", "true", "yes", "on"}


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text: str) -> str:
    """Remove any Qwen3 ``<think>...</think>`` block from the returned content."""
    return _THINK_BLOCK.sub("", text or "").strip()


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 800,
    model: str | None = None,
) -> str:
    """Call the local chat-completions endpoint and return assistant text.

    Raises ``LocalLLMUnavailable`` on any connection/HTTP/parse failure.
    """
    body: dict[str, Any] = {
        "model": model or _model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if not _thinking_enabled():
        # vLLM and SGLang both forward chat_template_kwargs into the Qwen3 chat template.
        body["chat_template_kwargs"] = {"enable_thinking": False}

    request = urllib.request.Request(
        f"{_base_url()}/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(body).encode("utf-8"),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_timeout()) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as exc:
        raise LocalLLMUnavailable(f"local LLM call failed: {exc}") from exc

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LocalLLMUnavailable(f"unexpected local LLM response: {result!r}") from exc
    return strip_thinking(content or "")


def is_available() -> bool:
    """Cheap reachability probe used by tools to decide on the extractive fallback."""
    try:
        chat([{"role": "user", "content": "ping"}], max_tokens=1)
        return True
    except LocalLLMUnavailable:
        return False
