"""Orchestration for the guideline RAG pipeline.

Functions here are the thin tools' real implementation:
- ``build_guideline_index`` : corpus -> chunks -> embeddings -> FAISS
- ``retrieve_passages``     : question -> top-k cited passages
- ``run_guideline_rag``     : retrieve -> grounded answer -> (optional) verify -> studio card
- ``verify_citations``      : per-claim faithfulness check (hallucination guard)
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from app.services.guideline import config, corpus as corpus_mod, embedder, store
from app.services.local_llm import LocalLLMUnavailable, chat

# --------------------------------------------------------------------------- #
# Index build
# --------------------------------------------------------------------------- #

ALLOWED_UPLOAD_SUFFIXES = {".md", ".markdown", ".txt", ".text", ".pdf"}


def list_corpus_documents(index_dir: str | None = None) -> dict[str, Any]:
    """List the documents currently in the FAISS index, grouped by source doc."""
    idir = Path(index_dir) if index_dir else config.index_dir()
    chunks = store.load_chunks(idir)
    meta = store.load_meta(idir)
    by_doc: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        doc_id = chunk.get("doc_id") or str(chunk.get("chunk_id", "")).split("::")[0]
        entry = by_doc.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "title": chunk.get("title", "") or doc_id,
                "source": chunk.get("source", "") or "",
                "url": chunk.get("url"),
                "n_chunks": 0,
            },
        )
        entry["n_chunks"] += 1
    documents = sorted(by_doc.values(), key=lambda d: str(d["title"]).lower())
    return {
        "embed_model": meta.get("embed_model", "") or config.embed_model_name(),
        "n_docs": len(documents),
        "n_chunks": len(chunks),
        "documents": documents,
    }


def ingest_guideline_document(filename: str, data: bytes, *, rebuild: bool = True) -> dict[str, Any]:
    """Save an uploaded document into the corpus dir and (re)build the index."""
    from pathlib import Path as _Path

    safe_name = os.path.basename(filename or "").strip() or "document.md"
    suffix = _Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}"
        )
    cdir = config.corpus_dir()
    cdir.mkdir(parents=True, exist_ok=True)
    dest = cdir / safe_name
    dest.write_bytes(data)

    if not rebuild:
        return {"uploaded": safe_name, "corpus_dir": str(cdir)}
    result = build_guideline_index()
    result["uploaded"] = safe_name
    return result


def build_guideline_index(
    *,
    corpus_dir: str | None = None,
    index_dir: str | None = None,
    embed_model: str | None = None,
    chunk_size: int = 220,
    overlap: int = 40,
) -> dict[str, Any]:
    cdir = Path(corpus_dir) if corpus_dir else config.corpus_dir()
    idir = Path(index_dir) if index_dir else config.index_dir()
    model_name = embed_model or config.embed_model_name()

    docs = corpus_mod.load_documents(cdir)
    if not docs:
        raise FileNotFoundError(f"No guideline documents found under {cdir}")
    chunks = corpus_mod.chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    vectors = embedder.embed_texts([c.text for c in chunks], model_name)
    meta = store.build_index(idir, chunks, vectors, model_name)

    return {
        "index_dir": str(idir),
        "embed_model": model_name,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
        "dim": int(meta["dim"]),
        "documents": [d.title for d in docs],
        "draft_answer": f"Indexed {len(docs)} document(s) into {len(chunks)} chunks at {idir}.",
    }


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #

def retrieve_passages(
    question: str,
    *,
    top_k: int = 6,
    min_score: float = 0.2,
    index_dir: str | None = None,
) -> list[dict[str, Any]]:
    idir = Path(index_dir) if index_dir else config.index_dir()
    chunks = store.load_chunks(idir)
    if not chunks:
        raise FileNotFoundError(f"No index chunks at {idir}; build the index first.")
    meta = store.load_meta(idir)
    model_name = meta.get("embed_model") or config.embed_model_name()

    query_vec = embedder.embed_texts([question], model_name)
    hits = store.search(idir, query_vec, top_k)

    passages: list[dict[str, Any]] = []
    rank = 0
    for idx, score in hits:
        if score < min_score or idx >= len(chunks):
            continue
        rank += 1
        chunk = chunks[idx]
        passages.append(
            {
                "ref_id": f"REF{rank}",
                "doc_title": chunk.get("title", ""),
                "source": chunk.get("source", ""),
                "url": chunk.get("url"),
                "chunk_id": chunk.get("chunk_id", str(idx)),
                "text": chunk.get("text", ""),
                "score": round(float(score), 4),
            }
        )
    return passages


# --------------------------------------------------------------------------- #
# Grounded generation
# --------------------------------------------------------------------------- #

_ANSWER_SYSTEM = (
    "You are a careful clinical evidence assistant. Answer ONLY using the numbered guideline "
    "passages provided. Cite every clinical claim with its passage tag, e.g. [REF1]. If the passages "
    "do not contain the answer, reply exactly: 'Insufficient evidence in the provided guidelines.' "
    "Do not use outside knowledge. Be concise and clinical."
)


def _format_passages(passages: list[dict[str, Any]]) -> str:
    blocks = []
    for passage in passages:
        header = f"[{passage['ref_id']}] {passage['doc_title']} ({passage['source']})"
        blocks.append(f"{header}\n{passage['text']}")
    return "\n\n".join(blocks)


def _extractive_fallback(passages: list[dict[str, Any]]) -> str:
    lines = ["LLM endpoint unavailable — showing the top retrieved guideline passages as evidence:"]
    for passage in passages:
        snippet = passage["text"][:400].rsplit(" ", 1)[0]
        lines.append(f"\n[{passage['ref_id']}] {passage['doc_title']} ({passage['source']}): {snippet}…")
    return "\n".join(lines)


def generate_grounded_answer(
    question: str,
    passages: list[dict[str, Any]],
    *,
    source_context: str | None = None,
) -> dict[str, Any]:
    if not passages:
        return {
            "draft_answer": "Insufficient evidence in the provided guidelines.",
            "used_fallback": False,
            "model": os.getenv("LOCAL_LLM_MODEL", "qwen3-8b"),
            "uncertainty": "No passages retrieved above the similarity threshold.",
        }

    user = f"Clinical question:\n{question}\n\nNumbered guideline passages:\n{_format_passages(passages)}"
    if source_context:
        user += f"\n\nAdditional patient/source context:\n{source_context}"
    user += "\n\nWrite a grounded answer that cites each claim with [REF#]."

    try:
        answer = chat(
            [{"role": "system", "content": _ANSWER_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=700,
        )
        used_fallback = False
        model = os.getenv("LOCAL_LLM_MODEL", "qwen3-8b")
    except LocalLLMUnavailable:
        answer = _extractive_fallback(passages)
        used_fallback = True
        model = "extractive-fallback"

    uncertainty = "" if "[REF" in answer or "REF" in answer else (
        "Answer did not cite specific passages; manual verification recommended."
    )
    return {
        "draft_answer": answer.strip(),
        "used_fallback": used_fallback,
        "model": model,
        "uncertainty": uncertainty,
    }


# --------------------------------------------------------------------------- #
# Citation verification (faithfulness / hallucination guard)
# --------------------------------------------------------------------------- #

_VERIFY_SYSTEM = (
    "You are a strict fact-checking verifier. Given one clinical CLAIM and one guideline PASSAGE, "
    "decide whether the passage directly supports the claim. Respond ONLY with a compact JSON object: "
    '{"supported": true|false, "evidence_span": "<short supporting quote or empty>"}. '
    "Default to false if unsure."
)


def _split_claims(answer: str) -> list[tuple[str, list[str]]]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    claims: list[tuple[str, list[str]]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            claims.append((sentence, re.findall(r"REF\d+", sentence)))
    return claims


def _parse_json(raw: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _lexical_support(claim: str, passage_text: str) -> dict[str, Any]:
    claim_terms = {w for w in re.findall(r"[a-z0-9]+", claim.lower()) if len(w) > 3}
    passage_terms = set(re.findall(r"[a-z0-9]+", passage_text.lower()))
    if not claim_terms:
        return {"supported": False, "evidence_span": None, "note": "no content terms"}
    overlap = len(claim_terms & passage_terms) / len(claim_terms)
    return {
        "supported": overlap >= 0.5,
        "evidence_span": None,
        "note": f"lexical overlap {overlap:.2f} (LLM unavailable)",
    }


def _verify_one(claim: str, passage_text: str) -> dict[str, Any]:
    user = f"CLAIM:\n{claim}\n\nPASSAGE:\n{passage_text}\n\nReturn the JSON verdict."
    try:
        raw = chat(
            [{"role": "system", "content": _VERIFY_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=200,
        )
    except LocalLLMUnavailable:
        return _lexical_support(claim, passage_text)
    parsed = _parse_json(raw)
    if parsed is None:
        return _lexical_support(claim, passage_text)
    return {
        "supported": bool(parsed.get("supported")),
        "evidence_span": parsed.get("evidence_span") or None,
        "note": None,
    }


def verify_citations(answer: str, passages: list[dict[str, Any]]) -> dict[str, Any]:
    by_ref = {p["ref_id"]: p for p in passages}
    claims_out: list[dict[str, Any]] = []
    supported = 0
    total = 0
    for sentence, refs in _split_claims(answer):
        if not refs:
            continue  # only verify claims that carry a citation
        total += 1
        ref = refs[0]
        passage = by_ref.get(ref)
        if passage is None:
            claims_out.append(
                {
                    "claim": sentence,
                    "ref_id": ref,
                    "supported": False,
                    "evidence_span": None,
                    "note": f"{ref} not present in retrieved passages",
                }
            )
            continue
        verdict = _verify_one(sentence, passage["text"])
        supported += 1 if verdict["supported"] else 0
        claims_out.append(
            {
                "claim": sentence,
                "ref_id": ref,
                "supported": verdict["supported"],
                "evidence_span": verdict.get("evidence_span"),
                "note": verdict.get("note"),
            }
        )
    faithfulness = round(supported / total, 3) if total else 1.0
    return {
        "claims": claims_out,
        "total_claims": total,
        "supported_count": supported,
        "unsupported_count": total - supported,
        "faithfulness": faithfulness,
    }


# --------------------------------------------------------------------------- #
# Top-level RAG entrypoint used by guideline_rag_tool
# --------------------------------------------------------------------------- #

def _build_studio_card(question, answer, passages, verifier) -> dict[str, Any]:
    return {
        "renderer": "guideline_rag",
        "title": "Guideline RAG",
        "data": {
            "question": question,
            "answer": answer,
            "passages": passages,
            "verifier": verifier,
        },
    }


def run_guideline_rag(
    question: str,
    *,
    top_k: int = 6,
    min_score: float = 0.2,
    source_context: str | None = None,
    index_dir: str | None = None,
    verify: bool = True,
    external_evidence: bool = False,
    external_max: int = 3,
) -> dict[str, Any]:
    passages = retrieve_passages(question, top_k=top_k, min_score=min_score, index_dir=index_dir)

    # Federation: optionally pull evidence from external MCP servers (server-to-server)
    # and merge it in, continuing the REF numbering after the local passages.
    if external_evidence:
        try:
            from app.services.mcp_gateway import search_external_evidence

            external = search_external_evidence(question, max_results=external_max)
        except Exception:  # noqa: BLE001 - external evidence is best-effort
            external = []
        for offset, passage in enumerate(external):
            passage["ref_id"] = f"REF{len(passages) + offset + 1}"
        passages = passages + external

    generated = generate_grounded_answer(question, passages, source_context=source_context)
    answer = generated["draft_answer"]

    references = [
        {
            "id": p["ref_id"],
            "title": p["doc_title"],
            "source": p["source"],
            "url": p["url"] or "",
            "note": (p["text"][:160] + "…") if len(p["text"]) > 160 else p["text"],
        }
        for p in passages
    ]

    verifier = None
    if verify and not generated["used_fallback"] and passages:
        verifier = verify_citations(answer, passages)

    return {
        "question": question,
        "draft_answer": answer,
        "references": references,
        "passages": passages,
        "uncertainty": generated["uncertainty"],
        "used_fallback": generated["used_fallback"],
        "model": generated["model"],
        "verifier": verifier,
        "studio": _build_studio_card(question, answer, passages, verifier),
    }
