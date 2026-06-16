"""Evaluation harness for the Medical RAG project.

- PubMedQA (pqa_labeled): closed-book Qwen3 vs our RAG (retrieve over the question's
  provided contexts) -> yes/no/maybe accuracy, RAG lift, and verifier faithfulness.
- MIRAGE: Qwen3-8B zero-shot multiple-choice accuracy across its 5 subsets (the
  standard no-retrieval baseline; full retrieval-augmented MIRAGE needs the MedCorp corpus).

Usage:
    python eval/evaluate.py --pubmedqa-n 150 --mirage-n 40 --top-k 5 --workers 8
Needs a running vLLM (LOCAL_LLM_BASE_URL) and EMBED_MODEL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.guideline.embedder import embed_texts  # noqa: E402
from app.services.guideline.pipeline import verify_citations  # noqa: E402
from app.services.local_llm import LocalLLMUnavailable, chat  # noqa: E402

DATA = ROOT / "eval" / "data"
RESULTS = ROOT / "eval" / "results"
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def sent_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def cosine_topk(q: np.ndarray, mat: np.ndarray, k: int) -> list[int]:
    sims = mat @ q  # normalized vectors -> cosine
    return np.argsort(-sims)[:k].tolist()


def parse_decision(text: str) -> str | None:
    m = re.search(r"\b(yes|no|maybe)\b", (text or "").lower())
    return m.group(1) if m else None


def parse_letter(text: str, options: dict) -> str | None:
    t = (text or "").strip()
    m = re.search(r"\b([A-E])\b", t.upper())
    if m and m.group(1) in options:
        return m.group(1)
    return None


# --------------------------------------------------------------------------- #
# PubMedQA
# --------------------------------------------------------------------------- #

_PQA_CB = (
    "You are a biomedical QA system. Answer the research question with exactly one word: "
    "yes, no, or maybe. Output only that word."
)
_PQA_RAG = (
    "You are a biomedical QA system. Using ONLY the numbered passages, answer the research "
    "question. Line 1: exactly one word (yes/no/maybe). Line 2: one justification sentence that "
    "cites passages like [REF1]. Do not use outside knowledge."
)


def pubmedqa_closed_book(question: str) -> str | None:
    try:
        out = chat([{"role": "system", "content": _PQA_CB}, {"role": "user", "content": question}],
                   temperature=0.0, max_tokens=8)
    except LocalLLMUnavailable:
        return None
    return parse_decision(out)


def pubmedqa_rag(question: str, contexts: list[str], top_k: int) -> dict:
    passages_text = [s for ctx in contexts for s in sent_split(ctx)] or [c for c in contexts if c]
    mat = embed_texts(passages_text, EMBED_MODEL)
    q = embed_texts([question], EMBED_MODEL)[0]
    idx = cosine_topk(q, mat, min(top_k, len(passages_text)))
    passages = [{"ref_id": f"REF{i+1}", "text": passages_text[j]} for i, j in enumerate(idx)]
    block = "\n".join(f"[{p['ref_id']}] {p['text']}" for p in passages)
    user = f"Question: {question}\n\nPassages:\n{block}"
    try:
        out = chat([{"role": "system", "content": _PQA_RAG}, {"role": "user", "content": user}],
                   temperature=0.0, max_tokens=160)
    except LocalLLMUnavailable:
        return {"decision": None, "faithfulness": None}
    decision = parse_decision(out.splitlines()[0] if out else "")
    faith = None
    if "REF" in out:
        v = verify_citations(out, passages)
        faith = v["faithfulness"] if v["total_claims"] else None
    return {"decision": decision, "faithfulness": faith}


def run_pubmedqa(n: int, top_k: int, workers: int) -> dict:
    data = json.load(open(DATA / "pubmedqa_ori_pqal.json"))
    items = [(k, v) for k, v in sorted(data.items())][:n]

    def work(kv):
        _, v = kv
        gold = v["final_decision"]
        cb = pubmedqa_closed_book(v["QUESTION"])
        rag = pubmedqa_rag(v["QUESTION"], v["CONTEXTS"], top_k)
        return {"gold": gold, "cb": cb, "rag": rag["decision"], "faith": rag["faithfulness"]}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(work, items))

    cb_acc = np.mean([r["cb"] == r["gold"] for r in rows])
    rag_acc = np.mean([r["rag"] == r["gold"] for r in rows])
    faiths = [r["faith"] for r in rows if r["faith"] is not None]
    return {
        "n": len(rows),
        "closed_book_accuracy": round(float(cb_acc), 4),
        "rag_accuracy": round(float(rag_acc), 4),
        "rag_lift": round(float(rag_acc - cb_acc), 4),
        "mean_faithfulness": round(float(np.mean(faiths)), 4) if faiths else None,
        "faithfulness_n": len(faiths),
    }


# --------------------------------------------------------------------------- #
# MIRAGE (zero-shot multiple choice)
# --------------------------------------------------------------------------- #

_MIRAGE_SYS = (
    "You are a medical exam solver. Choose the single best option. "
    "Respond with only the option letter (A, B, C, D, or E)."
)


def mirage_one(question: str, options: dict) -> str | None:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = f"Question: {question}\n\nOptions:\n{opts}\n\nAnswer with one letter."
    try:
        out = chat([{"role": "system", "content": _MIRAGE_SYS}, {"role": "user", "content": user}],
                   temperature=0.0, max_tokens=8)
    except LocalLLMUnavailable:
        return None
    return parse_letter(out, options)


def run_mirage(per_n: int, workers: int) -> dict:
    bench = json.load(open(DATA / "mirage_benchmark.json"))
    out = {}
    for ds, sub in bench.items():
        items = [v for _, v in sorted(sub.items())][:per_n]

        def work(it):
            pred = mirage_one(it["question"], it["options"])
            return pred == it["answer"]

        with ThreadPoolExecutor(max_workers=workers) as ex:
            correct = list(ex.map(work, items))
        out[ds] = {"n": len(items), "accuracy": round(float(np.mean(correct)), 4)}
    overall = np.mean([v["accuracy"] for v in out.values()])
    out["_macro_avg"] = round(float(overall), 4)
    return out


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmedqa-n", type=int, default=150)
    ap.add_argument("--mirage-n", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"[eval] embed_model={EMBED_MODEL} llm={os.getenv('LOCAL_LLM_BASE_URL')}")
    print(f"[eval] PubMedQA (n={args.pubmedqa_n}, top_k={args.top_k}) ...")
    pubmedqa = run_pubmedqa(args.pubmedqa_n, args.top_k, args.workers)
    print("  ", pubmedqa)
    print(f"[eval] MIRAGE (n={args.mirage_n}/subset) ...")
    mirage = run_mirage(args.mirage_n, args.workers)
    print("  ", mirage)

    report = {"embed_model": EMBED_MODEL, "model": os.getenv("LOCAL_LLM_MODEL", "qwen3-8b"),
              "pubmedqa": pubmedqa, "mirage": mirage}
    (RESULTS / "results.json").write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote {RESULTS / 'results.json'}")


if __name__ == "__main__":
    main()
