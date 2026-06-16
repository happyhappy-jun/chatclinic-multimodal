# guideline_index_tool

Builds the **FAISS retrieval index** for the guideline RAG pipeline: loads the corpus
(`.md`/`.txt`/`.pdf`, with optional `---` frontmatter for title/source/url), chunks it (word window +
overlap), embeds chunks with a local SentenceTransformer, and writes `faiss.index` + `chunks.jsonl` +
`meta.json`.

## Contract
- Entrypoint: `plugins.guideline_index_tool.logic:execute`
- Input: `{corpus_dir?, index_dir?, embed_model?, chunk_size=220, chunk_overlap=40}`
- Output: `{index_dir, embed_model, n_docs, n_chunks, dim, documents[]}`
- HTTP: `POST /api/v1/guideline/index`
- CLI: `python -m plugins.guideline_index_tool.logic`

## Defaults
- Corpus: `examples/guidelines/` · Index: `data/guideline_index/`
- Embedding model: `EMBED_MODEL` env (default `BAAI/bge-m3`; a small model like
  `sentence-transformers/all-MiniLM-L6-v2` works for fast local tests).

Run this once before any `guideline_rag_tool` query, and again whenever the corpus changes.
