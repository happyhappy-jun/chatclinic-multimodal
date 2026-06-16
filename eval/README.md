# Evaluation

Harness for the Medical RAG project. Results summary: `../docs/medical_rag/06_EVAL.md`.

## Data (downloaded, gitignored)
```bash
mkdir -p eval/data
curl -fsSL -o eval/data/pubmedqa_ori_pqal.json \
  https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json
curl -fsSL -o eval/data/mirage_benchmark.json \
  https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json
```

## Run
```bash
# vLLM serving Qwen3-8B; .env LOCAL_LLM_BASE_URL + EMBED_MODEL set
python eval/evaluate.py --pubmedqa-n 200 --mirage-n 60 --top-k 5 --workers 16
# -> eval/results/results.json
```

## What it measures
- **PubMedQA**: closed-book Qwen3 vs our RAG (retrieve over the question's contexts) → yes/no/maybe
  accuracy, RAG lift, and citation faithfulness (via `citation_verifier_tool`).
- **MIRAGE**: Qwen3-8B zero-shot multiple-choice accuracy across its 5 subsets (no-retrieval baseline).
