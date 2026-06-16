# Submission Package — Medical RAG (AI619)

Team: 윤병준, 이현석, 김태영 · Topic: **Medical RAG** · Branch: `medical-rag`

Maps to the `chatclinic-class` final-submission package structure:

| Required item | Location |
|---|---|
| **Plugin package** | `plugins/guideline_index_tool/`, `plugins/guideline_rag_tool/`, `plugins/citation_verifier_tool/`, `plugins/mcp_federation_tool/` (each has `tool.json`, `logic.py`, `README.md`, `requirements.txt`) |
| **Skill patch proposal** | `submission/skill_update/skill_patch.md`, `skill_rationale.md` (applied to `skills/chatgenome-orchestrator/SKILL.md`) |
| **Background papers** | `submission/references/background_papers.md` |
| **Sample test data** | `examples/guidelines/` (guideline corpus) + `examples/guideline_sample_queries.md` |
| **Slides** | → KLMS (not in repo) |
| **Demo video** | → KLMS (not in repo) |

## Supporting docs
- Design + setup + demo + MCP: `docs/medical_rag/`
- Run guide: `docs/medical_rag/04_DEMO.md` · MCP: `docs/medical_rag/05_MCP.md`

## What it does
Clinical guideline/literature RAG: retrieve guideline passages → grounded, `[REF#]`-cited answer from a
local **Qwen3-8B** (vLLM) → **citation-faithfulness** check. Plus an **MCP gateway** (server-to-server):
ChatClinic is exposed as an MCP server and federates to external MCP servers for extra evidence.
