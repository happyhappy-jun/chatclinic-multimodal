"""mcp_federation_tool — discover and call tools on external MCP servers.

Actions:
  - list (default): discover tools across all configured external MCP servers
  - call: invoke one external tool

This is the client/federation half of the MCP gateway (server-to-server).
"""
from __future__ import annotations

from app.services.mcp_gateway import call_external_tool, list_external_tools


def execute(payload: dict) -> dict:
    action = (payload.get("action") or "list").strip().lower()
    if action == "list":
        return list_external_tools()
    if action == "call":
        server = payload.get("server")
        tool = payload.get("tool")
        if not server or not tool:
            raise ValueError("action='call' requires 'server' and 'tool'.")
        return call_external_tool(server, tool, payload.get("arguments") or {})
    raise ValueError(f"Unknown action: {action!r} (expected 'list' or 'call').")
