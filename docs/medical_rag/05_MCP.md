# 05 · MCP Gateway (server-to-server)

Our Medical RAG also acts as an **MCP gateway** — it is both an MCP **server** (exposing our RAG
tools to other agents) and an MCP **client** (federating out to other MCP servers). Transport:
**Streamable HTTP** (official `mcp` Python SDK).

> Spec context: per `bispl-create/chatclinic-class` `docs/COURSE_TOOLS.md`, a separate MCP server is
> *optional* for a tool team — it's the **Agentic-model** pattern ("MCP-connected external tools /
> agent2agent"). We add it as a Project-Quality enhancement on top of the required plugin deliverable.

## Components

| Piece | File | Role |
|------|------|------|
| Our MCP **server** | `mcp_servers/chatclinic_rag_server.py` | Exposes `retrieve_guidelines`, `guideline_rag`, `verify_citations_tool` over MCP → any agent can call our RAG. |
| Sample **external** MCP server | `mcp_servers/pubmed_lite_server.py` | Stand-in third-party literature server (`search_literature`) so we can demo real server-to-server traffic. |
| MCP **client / federation** | `app/services/mcp_gateway.py` | Discovers + calls tools on configured external MCP servers; normalizes literature hits into RAG passages. |
| Federation **tool** | `plugins/mcp_federation_tool/` | ChatClinic tool: `action=list` / `action=call` over the gateway. |
| Config | `mcp_servers.json` | List of external MCP servers (name, url, optional `evidence_tool`). |

## Data flow

```
 other agent / MCP client ──► [ our MCP server :9000 ] ──► guideline_rag pipeline ──► Qwen3-8B
                                                                  ▲
 ChatClinic backend ──(MCP client)──► [ external MCP :9001 pubmed-lite ] ──► search_literature
        │                                                         │
        └────────── merges external evidence into RAG passages ◄──┘   (external_evidence=true)
```

## Run

```bash
bash scripts/run_mcp_servers.sh        # starts pubmed-lite (:9001) + our rag server (:9000)
# stop: bash scripts/run_mcp_servers.sh stop
```

## API (via the FastAPI backend)

```bash
# discover tools across federated external MCP servers
curl -X POST http://127.0.0.1:8001/api/v1/mcp/tools

# call one external tool
curl -X POST http://127.0.0.1:8001/api/v1/mcp/call -H 'Content-Type: application/json' \
  -d '{"server":"pubmed-lite","tool":"search_literature","arguments":{"query":"sepsis","max_results":2}}'

# RAG that fuses local guideline corpus + external MCP evidence
curl -X POST http://127.0.0.1:8001/api/v1/guideline-rag/run -H 'Content-Type: application/json' \
  -d '{"question":"antibiotic duration for outpatient pneumonia","external_evidence":true}'
```

## Verified

- Gateway lists + calls the external server's tool over Streamable HTTP. ✓
- `guideline_rag(external_evidence=true)` merges `external MCP: pubmed-lite` passages alongside local
  guideline passages (continuing REF numbering); answer grounded by Qwen3, faithfulness 1.0. ✓
- Our `guideline_rag` tool is callable by an MCP client (agent2agent). ✓
- Endpoints `/api/v1/mcp/tools`, `/api/v1/mcp/call`, and `external_evidence` on `/guideline-rag/run`. ✓

## Notes / extensibility

- Add a real external MCP server (PubMed, FHIR, drug DB) by appending to `mcp_servers.json`; set
  `evidence_tool` if it should feed RAG. Unreachable servers are reported in `errors`, never crash.
- The RAG server needs the backend env (index + `LOCAL_LLM_BASE_URL`); `run_mcp_servers.sh` loads `.env`.
