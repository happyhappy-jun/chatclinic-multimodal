"""Sample EXTERNAL MCP server: 'pubmed-lite'.

Stands in for a third-party biomedical-literature MCP server so we can demo real
MCP server-to-server traffic locally. Exposes a `search_literature` tool over
Streamable HTTP. The "articles" are a small canned set representing *recent
literature* distinct from our guideline corpus.

Run:
    python mcp_servers/pubmed_lite_server.py            # -> http://127.0.0.1:9001/mcp
    PORT=9001 python mcp_servers/pubmed_lite_server.py
"""
from __future__ import annotations

import json
import os
import re

from mcp.server.fastmcp import FastMCP

# A tiny "external" literature store (NOT in the guideline corpus on purpose).
_ARTICLES = [
    {
        "pmid": "40012345",
        "title": "Five-day vs longer beta-lactam courses in outpatient community-acquired pneumonia: a randomized non-inferiority trial",
        "abstract": "In low-severity CAP, a fixed 5-day amoxicillin course was non-inferior to longer courses for clinical cure, supporting short-course therapy when patients are afebrile and stable.",
        "journal": "J Resp Med",
        "year": 2025,
        "keywords": ["pneumonia", "cap", "antibiotic", "amoxicillin", "outpatient", "duration"],
    },
    {
        "pmid": "40023456",
        "title": "SGLT2 inhibitors and cardiorenal outcomes in type 2 diabetes: updated meta-analysis",
        "abstract": "SGLT2 inhibitors reduced heart-failure hospitalization and chronic kidney disease progression in type 2 diabetes irrespective of baseline HbA1c, reinforcing organ-protective first-line use.",
        "journal": "Diabetes Care Rev",
        "year": 2025,
        "keywords": ["diabetes", "sglt2", "heart failure", "kidney", "cardiorenal", "glp-1"],
    },
    {
        "pmid": "40034567",
        "title": "Early norepinephrine initiation in septic shock and time-to-target MAP",
        "abstract": "Earlier norepinephrine initiation was associated with faster achievement of a mean arterial pressure of 65 mmHg and lower cumulative fluid balance in septic shock.",
        "journal": "Crit Care Lett",
        "year": 2024,
        "keywords": ["sepsis", "septic shock", "norepinephrine", "vasopressor", "map"],
    },
    {
        "pmid": "40045678",
        "title": "Intensive vs standard blood-pressure targets in older adults: a pragmatic trial",
        "abstract": "A target below 130/80 mmHg reduced major cardiovascular events versus a standard target in adults over 60 without increasing serious adverse events, with careful monitoring.",
        "journal": "Hypertension Today",
        "year": 2025,
        "keywords": ["hypertension", "blood pressure", "target", "older adults", "cardiovascular"],
    },
]

mcp = FastMCP(
    "pubmed-lite",
    instructions="Sample external biomedical literature MCP server for the AI619 medical-rag demo.",
    host=os.getenv("HOST", "127.0.0.1"),
    port=int(os.getenv("PORT", "9001")),
)


@mcp.tool()
def search_literature(query: str, max_results: int = 3) -> str:
    """Search recent biomedical literature for a clinical query.

    Returns a JSON string: {"articles": [{pmid,title,abstract,journal,year,url,score}]}.
    """
    terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    scored = []
    for art in _ARTICLES:
        hay = set(art["keywords"]) | set(re.findall(r"[a-z0-9]+", (art["title"] + " " + art["abstract"]).lower()))
        overlap = len(terms & hay)
        if overlap:
            scored.append((overlap, art))
    scored.sort(key=lambda x: x[0], reverse=True)
    articles = [
        {
            "pmid": a["pmid"],
            "title": a["title"],
            "abstract": a["abstract"],
            "journal": a["journal"],
            "year": a["year"],
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{a['pmid']}/",
            "score": score,
        }
        for score, a in scored[: max(1, max_results)]
    ]
    return json.dumps({"query": query, "articles": articles}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
