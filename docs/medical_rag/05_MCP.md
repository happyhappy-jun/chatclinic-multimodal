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
| **Live PubMed** MCP server | `mcp_servers/pubmed_server.py` | Real NCBI E-utilities (`esearch` → `efetch`) → titles + abstracts. The active evidence source. |
| Offline **fallback** MCP server | `mcp_servers/pubmed_lite_server.py` | Canned literature (`search_literature`) for offline demos; discoverable but not used for evidence. |
| MCP **client / federation** | `app/services/mcp_gateway.py` | Discovers + calls external MCP tools; fetches a candidate pool and **semantically re-ranks** it with our embedder before merging into RAG passages. |
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

## Live PubMed retrieval (Mode A)

`external_evidence=true` now performs **live retrieval-augmented generation over PubMed**:

1. `mcp_servers/pubmed_server.py` runs `esearch` (with a key-term fallback if the full question returns
   nothing) → PMIDs, then `efetch` → real titles + abstracts (up to a `candidate_pool` of ~20).
2. The gateway **embeds the question + candidate abstracts and keeps the top-k by cosine** — lexical
   recall from PubMed, semantic precision from our embedder.
3. Those passages merge with the local guideline passages (continuing REF numbering); Qwen3 grounds the
   answer and cites them with real PMIDs.

Env: `PUBMED_EMAIL` (NCBI etiquette), optional `NCBI_API_KEY` (raises the rate limit 3→10 req/s).
Verified: a "vasopressor in septic shock" query re-ranked 20 live PubMed hits to the on-topic
*"Vasopressors in septic shock: which, when, and how much?"* (cos 0.60) and merged it with the local
sepsis guideline.

## Notes / extensibility

- Add another real external MCP server (FHIR, drug DB) by appending to `mcp_servers.json`; set
  `evidence_tool` if it should feed RAG. Unreachable servers are reported in `errors`, never crash —
  the request falls back to local-corpus evidence.
- The RAG server needs the backend env (index + `LOCAL_LLM_BASE_URL`); `run_mcp_servers.sh` loads `.env`.
