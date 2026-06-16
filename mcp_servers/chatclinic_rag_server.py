"""OUR MCP server: exposes the ChatClinic Guideline-RAG tools over MCP.

Any MCP client / agent (agent2agent) can now call our retrieval, grounded
answering, and citation-verification tools via Streamable HTTP.

Run:
    python mcp_servers/chatclinic_rag_server.py          # -> http://127.0.0.1:9000/mcp
    PORT=9000 python mcp_servers/chatclinic_rag_server.py

Needs the backend env (EMBED_MODEL / LOCAL_LLM_BASE_URL / GUIDELINE_INDEX_DIR)
and a built index — same as the FastAPI backend.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Allow `import app...` when launched as a plain script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from app.services.guideline.pipeline import (  # noqa: E402
    retrieve_passages,
    run_guideline_rag,
    verify_citations,
)

mcp = FastMCP(
    "chatclinic-guideline-rag",
    instructions=(
        "Clinical guideline RAG over a curated corpus with a local Qwen3-8B model. "
        "Use retrieve_guidelines for passages, guideline_rag for a grounded cited answer, "
        "and verify_citations to fact-check an answer against passages."
    ),
    host=os.getenv("HOST", "127.0.0.1"),
    port=int(os.getenv("PORT", "9000")),
)


@mcp.tool()
def retrieve_guidelines(question: str, top_k: int = 6, min_score: float = 0.2) -> str:
    """Retrieve the top-k guideline passages for a clinical question (JSON)."""
    passages = retrieve_passages(question, top_k=top_k, min_score=min_score)
    return json.dumps({"question": question, "passages": passages}, ensure_ascii=False)


@mcp.tool()
def guideline_rag(question: str, top_k: int = 6, verify: bool = True, external_evidence: bool = False) -> str:
    """Answer a clinical question, grounded + cited in the guideline corpus (JSON).

    Set external_evidence=true to also pull evidence from federated external MCP servers.
    """
    result = run_guideline_rag(question, top_k=top_k, verify=verify, external_evidence=external_evidence)
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
def verify_citations_tool(draft_answer: str, passages: list) -> str:
    """Check that each cited claim in an answer is supported by its passage (JSON)."""
    return json.dumps(verify_citations(draft_answer, passages or []), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
