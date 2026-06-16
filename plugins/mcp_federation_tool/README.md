# mcp_federation_tool

Client side of the **MCP gateway** (server-to-server / agent2agent). Discovers and invokes tools on
external MCP servers over **Streamable HTTP**, so ChatClinic can reach third-party MCP tools (e.g. a
PubMed/FHIR/drug-info server) and fold their results into the RAG pipeline.

## Contract
- Entrypoint: `plugins.mcp_federation_tool.logic:execute`
- Input: `{action="list"|"call", server?, tool?, arguments?}`
- Output (list): `{servers[], tools[], errors[]}` · Output (call): `{server, tool, is_error, text, structured, json}`
- HTTP: `POST /api/v1/mcp/tools` (list), `POST /api/v1/mcp/call` (call)

## Config
External servers are read from `mcp_servers.json` (or `MCP_SERVERS_CONFIG` / `MCP_SERVERS` env):
```json
{"servers": [{"name": "pubmed-lite", "url": "http://127.0.0.1:9001/mcp", "evidence_tool": "search_literature"}]}
```
Unreachable servers are reported in `errors`, never crash the request.

## Related
- Our RAG is also exposed *as* an MCP server: `mcp_servers/chatclinic_rag_server.py`.
- Sample external server for the demo: `mcp_servers/pubmed_lite_server.py`.
- See `docs/medical_rag/05_MCP.md`.
