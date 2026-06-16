# citation_verifier_tool

**Faithfulness / hallucination guard** for grounded answers. For each cited sentence (`[REF#]`) in an
answer, it checks whether the cited passage actually supports the claim, and reports a faithfulness
score. This is the quality differentiator that turns a plain RAG demo into a fact-checked one.

## Contract
- Entrypoint: `plugins.citation_verifier_tool.logic:execute`
- Input: `{draft_answer, passages[]}` (passages carry `ref_id` + `text`)
- Output: `{claims[], total_claims, supported_count, unsupported_count, faithfulness}`
- HTTP: `POST /api/v1/citation-check/run`

## Behavior
- Splits the answer into sentences, keeps only cited ones, and asks the local LLM (NLI-style) whether
  the cited passage entails each claim.
- Falls back to a lexical-overlap heuristic if the LLM endpoint is unavailable.
- `faithfulness = supported / total`; unsupported claims are surfaced, never hidden.

## Dependencies
No extra packages beyond the backend; uses the shared local-LLM client over HTTP.
