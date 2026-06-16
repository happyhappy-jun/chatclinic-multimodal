# Sample Test Queries — Guideline RAG

Sample inputs for testing `guideline_rag_tool` against the `examples/guidelines/` corpus.
Run after building the index (`python -m plugins.guideline_index_tool.logic`).

## In-corpus (should return a grounded, cited answer)
1. First-line antibiotics for a healthy adult outpatient with community-acquired pneumonia, and treatment duration?
2. Which first-line drug classes are recommended for stage 2 hypertension?
3. When should an SGLT2 inhibitor be preferred over a GLP-1 receptor agonist in type 2 diabetes?
4. What is the first-line vasopressor and the mean arterial pressure target in septic shock?
5. What blood-pressure threshold defines stage 1 hypertension?

## Out-of-corpus (should refuse: "Insufficient evidence in the provided guidelines.")
6. What is the recommended chemotherapy regimen for stage III colon cancer?
7. What is the dosing schedule for the measles-mumps-rubella vaccine?

## Federated (external_evidence=true → should also cite `external MCP: pubmed-lite`)
8. How long should antibiotics last for outpatient community-acquired pneumonia? (with external_evidence=true)

## Example call
```bash
curl -X POST http://127.0.0.1:8001/api/v1/guideline-rag/run \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which first-line drug classes are recommended for stage 2 hypertension?","top_k":4}'
```
