"""Real PubMed MCP server (live NCBI E-utilities) over Streamable HTTP.

Exposes `search_literature` which runs esearch -> efetch against PubMed and returns
real titles + abstracts. Kept "pure" (no embedder/LLM): it provides lexical recall;
the gateway (app/services/mcp_gateway.py) does semantic re-ranking for precision.

Run:
    python mcp_servers/pubmed_server.py            # -> http://127.0.0.1:9002/mcp
    PORT=9002 python mcp_servers/pubmed_server.py

Env (NCBI etiquette / limits):
    PUBMED_EMAIL   contact email (recommended)
    NCBI_API_KEY   raises rate limit 3->10 req/s (optional)
    PUBMED_TIMEOUT request timeout seconds (default 15)
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from mcp.server.fastmcp import FastMCP

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "chatclinic-rag"
EMAIL = os.getenv("PUBMED_EMAIL", "")
API_KEY = os.getenv("NCBI_API_KEY", "")
TIMEOUT = float(os.getenv("PUBMED_TIMEOUT", "15"))

_STOP = {
    "the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are", "with",
    "what", "which", "does", "do", "should", "when", "how", "be", "on", "at", "by",
}

mcp = FastMCP(
    "pubmed",
    instructions="Live PubMed literature search via NCBI E-utilities (titles + abstracts).",
    host=os.getenv("HOST", "127.0.0.1"),
    port=int(os.getenv("PORT", "9002")),
)


def _params(extra: dict) -> str:
    p = {"tool": TOOL}
    if EMAIL:
        p["email"] = EMAIL
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(extra)
    return urllib.parse.urlencode(p)


def _get(path: str, params: dict) -> bytes:
    url = f"{EUTILS}/{path}?{_params(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": TOOL})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return response.read()


def _esearch(query: str, retmax: int) -> list[str]:
    raw = _get("esearch.fcgi", {"db": "pubmed", "term": query, "retmax": retmax,
                                "sort": "relevance", "retmode": "json"})
    data = json.loads(raw.decode("utf-8"))
    return data.get("esearchresult", {}).get("idlist", [])


def _simplify(query: str) -> str:
    words = [w for w in re.findall(r"[A-Za-z0-9-]+", query) if w.lower() not in _STOP and len(w) > 2]
    return " ".join(words)


def _efetch(pmids: list[str]) -> list[dict]:
    if not pmids:
        return []
    raw = _get("efetch.fcgi", {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"})
    root = ET.fromstring(raw)
    articles: list[dict] = []
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//MedlineCitation/PMID")
        pmid = pmid_el.text if pmid_el is not None else ""
        title_el = art.find(".//Article/ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        parts = []
        for ab in art.findall(".//Article/Abstract/AbstractText"):
            label = ab.get("Label")
            txt = "".join(ab.itertext()).strip()
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)
        abstract = " ".join(parts)
        journal_el = art.find(".//Article/Journal/Title")
        journal = journal_el.text if journal_el is not None else ""
        year_el = art.find(".//Article/Journal/JournalIssue/PubDate/Year")
        year = year_el.text if year_el is not None else ""
        if title or abstract:
            articles.append({
                "pmid": pmid, "title": title, "abstract": abstract,
                "journal": journal, "year": year,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })
    return articles


@mcp.tool()
def search_literature(query: str, max_results: int = 20) -> str:
    """Search PubMed for a clinical query; return real titles + abstracts (JSON).

    Returns {"query","count","articles":[{pmid,title,abstract,journal,year,url}]}.
    The gateway re-ranks these candidates semantically before grounding.
    """
    pmids = _esearch(query, max_results)
    if not pmids:
        pmids = _esearch(_simplify(query), max_results)  # fallback to key terms
    time.sleep(0.34)  # be polite to NCBI between calls
    articles = _efetch(pmids[:max_results])
    return json.dumps({"query": query, "count": len(articles), "articles": articles}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
