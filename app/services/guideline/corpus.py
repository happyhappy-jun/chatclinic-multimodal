"""Load and chunk the guideline/literature corpus.

Supports ``.md``/``.txt`` (with optional ``---`` frontmatter for title/source/url)
and ``.pdf`` (via pypdf). Chunking is a word-window with overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    url: str | None
    text: str
    path: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    url: str | None
    text: str


_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip().strip('"').strip("'")
    return meta, raw[match.end():]


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def load_documents(corpus_dir: Path) -> list[Document]:
    docs: list[Document] = []
    if not corpus_dir.exists():
        return docs
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in TEXT_SUFFIXES:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            meta, body = _parse_frontmatter(raw)
        elif suffix in PDF_SUFFIXES:
            body, meta = _read_pdf(path), {}
        else:
            continue
        body = body.strip()
        if not body:
            continue
        docs.append(
            Document(
                doc_id=path.stem,
                title=meta.get("title") or path.stem.replace("_", " ").title(),
                source=meta.get("source") or "local guideline corpus",
                url=meta.get("url") or None,
                text=body,
                path=str(path),
            )
        )
    return docs


def chunk_documents(docs: list[Document], *, chunk_size: int = 220, overlap: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    for doc in docs:
        words = doc.text.split()
        if not words:
            continue
        idx, start = 0, 0
        while start < len(words):
            window = words[start:start + chunk_size]
            if not window:
                break
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{idx}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    source=doc.source,
                    url=doc.url,
                    text=" ".join(window),
                )
            )
            idx += 1
            start += step
    return chunks
