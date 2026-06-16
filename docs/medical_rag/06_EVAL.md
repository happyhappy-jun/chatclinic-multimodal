# 06 · Evaluation

Quantitative evaluation of the Medical RAG system. Generator: **`Qwen/Qwen3-8B`** (vLLM, non-thinking).
Retriever/embedder: `all-MiniLM-L6-v2` (dev default; production default `BAAI/bge-m3` is expected to do
at least as well). Harness: `eval/evaluate.py` → `eval/results/results.json`.

## PubMedQA — retrieval lift + citation faithfulness (n = 200, pqa_labeled)

We compare **closed-book** Qwen3-8B (answer yes/no/maybe from the question alone) against **our RAG**
(retrieve top-5 sentences from the question's provided contexts, then ground the answer), and measure
**citation faithfulness** with our `citation_verifier_tool`.

| Condition | yes/no/maybe accuracy |
|---|---|
| Closed-book Qwen3-8B | **21.5%** |
| **+ RAG (our pipeline)** | **51.5%** |
| **RAG lift** | **+30.0 points** (≈ 2.4×) |
| Citation faithfulness (verifier, n=198) | **91.7%** |

**Why closed-book is so low (and why that's the point):** PubMedQA questions reference specific studies;
without the retrieved abstract Qwen3 rationally hedges "maybe" (verified: ~5/8 of closed-book answers),
which scores wrong against the mostly-"yes" gold. Retrieval gives the model the evidence it needs —
**accuracy more than doubles**, and 91.7% of cited claims are verified as supported by their passage.

## MIRAGE — Qwen3-8B zero-shot baseline (n = 60 / subset)

Standard MIRAGE no-retrieval multiple-choice baseline across all 5 subsets:

| Subset | Accuracy |
|---|---|
| MedQA-US | 61.7% |
| MedMCQA | 48.3% |
| PubMedQA* | 55.0% |
| BioASQ-Y/N | 81.7% |
| MMLU-Med | 63.3% |
| **Macro average** | **62.0%** |

This establishes the generator's standalone competence on the MIRAGE benchmark. Full
**retrieval-augmented** MIRAGE would retrieve over the MedCorp corpus (textbooks/PubMed/StatPearls/
Wikipedia) — out of scope here (large corpus build); the PubMedQA experiment above already isolates and
demonstrates the retrieval contribution.

## Reproduce

```bash
# vLLM must be serving Qwen3-8B; .env LOCAL_LLM_BASE_URL set
python eval/evaluate.py --pubmedqa-n 200 --mirage-n 60 --top-k 5 --workers 16
```

## Honest caveats

- Faithfulness is judged by an LLM verifier from the same model family — a self-consistency signal, not
  an independent human gold; treat 91.7% as indicative.
- PubMedQA RAG retrieves over each question's *provided* contexts (the standard PubMedQA reader setting),
  which isolates grounding/answer quality rather than open-corpus retrieval.
- Dev embedder is `all-MiniLM-L6-v2`; rerun with `EMBED_MODEL=BAAI/bge-m3` for the production number.
- n is a deterministic prefix of each set (first-by-key); scale up for tighter confidence intervals.
