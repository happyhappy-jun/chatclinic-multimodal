# Background Papers

Representative references behind the Medical RAG tool (Clinical Guideline / Literature RAG) and the
MCP gateway. These are starting points; the implementation is grounded in but not limited to them.

## Retrieval-Augmented Generation (core)
- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Gao et al. *Retrieval-Augmented Generation for Large Language Models: A Survey.* arXiv 2023.

## Dense retrieval & embeddings
- Karpukhin et al. *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP 2020.
- Chen et al. *BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings.* arXiv 2024. (our default embedder, `BAAI/bge-m3`)
- Johnson, Douze, Jégou. *Billion-scale similarity search with GPUs* (FAISS). arXiv 2017. (vector index)

## Medical / biomedical RAG & retrieval
- Jin et al. *MedCPT: Contrastive Pre-trained Transformers for zero-shot biomedical information retrieval.* Bioinformatics 2023. (biomedical retriever option)
- Xiong et al. *Benchmarking Retrieval-Augmented Generation for Medicine (MIRAGE / MedRAG).* arXiv 2024. (eval design we mirror)
- Zakka et al. *Almanac: Retrieval-Augmented Language Models for Clinical Medicine.* NEJM AI 2024.
- Jin et al. *PubMedQA: A Dataset for Biomedical Research Question Answering.* EMNLP 2019. (eval set)

## Faithfulness / citation evaluation
- Es et al. *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv 2023.
- Bohnet et al. *Attributed Question Answering: Evaluation and Modeling for Attributed LLMs.* arXiv 2022. (motivation for the citation verifier)

## Generation model
- Qwen Team. *Qwen3 Technical Report.* 2025. (local generator, `Qwen/Qwen3-8B`, served via vLLM)
- Kwon et al. *Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).* SOSP 2023. (serving stack)

## Agentic / MCP
- Anthropic. *Model Context Protocol (MCP) — specification and documentation.* 2024–2025.
  <https://modelcontextprotocol.io> (server-to-server / agent2agent gateway).
