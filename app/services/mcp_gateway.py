"""MCP federation gateway (client role).

Connects OUT to external MCP servers over Streamable HTTP, discovers their tools,
calls them, and normalizes a literature-search tool into RAG passages. This is the
"server-to-server / agent2agent" half of the gateway.

Config (first found wins):
  1. env MCP_SERVERS_CONFIG -> path to a JSON file
  2. <repo>/mcp_servers.json
  3. env MCP_SERVERS = "name1=url1,name2=url2"

JSON shape:
  {"servers": [{"name": "pubmed-lite", "url": "http://127.0.0.1:9001/mcp",
                "evidence_tool": "search_literature"}]}
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

_ROOT = Path(__file__).resolve().parents[2]
_CONNECT_TIMEOUT = float(os.getenv("MCP_CONNECT_TIMEOUT", "15"))


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_servers() -> list[dict[str, Any]]:
    cfg_path = os.getenv("MCP_SERVERS_CONFIG")
    candidates = [Path(cfg_path)] if cfg_path else []
    candidates.append(_ROOT / "mcp_servers.json")
    for path in candidates:
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            servers = data.get("servers", []) if isinstance(data, dict) else data
            return [s for s in servers if s.get("url")]
    raw = os.getenv("MCP_SERVERS", "").strip()
    if raw:
        out = []
        for item in raw.split(","):
            if "=" in item:
                name, url = item.split("=", 1)
                out.append({"name": name.strip(), "url": url.strip()})
        return out
    return []


def _server_by_name(name: str | None) -> dict[str, Any] | None:
    for srv in load_servers():
        if srv.get("name") == name:
            return srv
    return None


# --------------------------------------------------------------------------- #
# Async primitives
# --------------------------------------------------------------------------- #

def _content_text(result: Any) -> str:
    texts = []
    for chunk in getattr(result, "content", []) or []:
        if getattr(chunk, "type", None) == "text":
            texts.append(chunk.text)
    return "\n".join(texts)


async def _alist_tools(url: str) -> list[dict[str, Any]]:
    async with streamablehttp_client(url, timeout=_CONNECT_TIMEOUT) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.list_tools()
            return [
                {"name": t.name, "description": t.description or "", "input_schema": t.inputSchema}
                for t in res.tools
            ]


async def _acall_tool(url: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async with streamablehttp_client(url, timeout=_CONNECT_TIMEOUT) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(name, arguments or {})
            text = _content_text(res)
            structured = getattr(res, "structuredContent", None)
            return {"is_error": bool(getattr(res, "isError", False)), "text": text, "structured": structured}


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Sync API used by tools / endpoints
# --------------------------------------------------------------------------- #

def list_external_tools() -> dict[str, Any]:
    """Discover tools across all configured external MCP servers."""
    servers = load_servers()
    out: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for srv in servers:
        try:
            tools = _run(_alist_tools(srv["url"]))
            for tool in tools:
                out.append({"server": srv.get("name"), "url": srv["url"], **tool})
        except Exception as exc:  # noqa: BLE001 - report unreachable servers, don't crash
            errors.append({"server": srv.get("name", "?"), "url": srv.get("url", "?"), "error": str(exc)})
    return {"servers": [s.get("name") for s in servers], "tools": out, "errors": errors}


def call_external_tool(server: str, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    srv = _server_by_name(server)
    if srv is None:
        raise ValueError(f"Unknown MCP server: {server!r}. Configured: {[s.get('name') for s in load_servers()]}")
    result = _run(_acall_tool(srv["url"], tool, arguments or {}))
    parsed = None
    if result.get("text"):
        try:
            parsed = json.loads(result["text"])
        except (json.JSONDecodeError, TypeError):
            parsed = None
    return {"server": server, "tool": tool, **result, "json": parsed}


def search_external_evidence(
    question: str,
    max_results: int = 3,
    *,
    candidate_pool: int = 20,
    rerank: bool = True,
) -> list[dict[str, Any]]:
    """Federate evidence from external MCP servers, then semantically re-rank.

    Each server contributes up to ``candidate_pool`` lexical hits (e.g. PubMed esearch);
    we then embed the question + candidate abstracts and keep the top ``max_results`` by
    cosine — recall from the source, precision from our embedder.
    """
    passages: list[dict[str, Any]] = []
    for srv in load_servers():
        evidence_tool = srv.get("evidence_tool")
        if not evidence_tool:
            continue
        try:
            res = _run(_acall_tool(srv["url"], evidence_tool, {"query": question, "max_results": candidate_pool}))
        except Exception:  # noqa: BLE001 - external server optional
            continue
        articles = []
        if res.get("text"):
            try:
                payload = json.loads(res["text"])
                articles = payload.get("articles", payload if isinstance(payload, list) else [])
            except (json.JSONDecodeError, TypeError):
                articles = []
        for art in articles:
            title = art.get("title", "external result")
            abstract = art.get("abstract") or art.get("text") or ""
            passages.append(
                {
                    "doc_title": title,
                    "source": f"external MCP: {srv.get('name')}",
                    "url": art.get("url"),
                    "chunk_id": f"mcp::{srv.get('name')}::{art.get('pmid', len(passages))}",
                    "text": f"{title}. {abstract}".strip() if abstract else title,
                    "score": float(art.get("score", 0.0)),
                    "origin": "external_mcp",
                }
            )

    if rerank and len(passages) > max_results:
        try:
            import numpy as np

            from app.services.guideline import config as gconfig
            from app.services.guideline.embedder import embed_texts

            model = gconfig.embed_model_name()
            mat = embed_texts([p["text"] for p in passages], model)
            qvec = embed_texts([question], model)[0]
            sims = mat @ qvec
            order = np.argsort(-sims)[: max_results]
            ranked = []
            for idx in order:
                passage = passages[int(idx)]
                passage["score"] = round(float(sims[int(idx)]), 4)
                ranked.append(passage)
            return ranked
        except Exception:  # noqa: BLE001 - fall back to lexical order if embedder unavailable
            pass

    return passages[: max_results]
