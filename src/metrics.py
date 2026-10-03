"""Eval metrics.

Two layers, because they answer different questions:

1. Retrieval metrics (no LLM needed, fully deterministic):
   recall@k, MRR — did we fetch the right documents?
2. Answer metrics (needs an LLM):
   faithfulness — is every cited claim supported by its chunk?
   refusal correctness — did it say "I don't know" exactly when it should?

The faithfulness check here is a heuristic (token-overlap support),
not an LLM judge. It's cheap, deterministic, and honest about what it
is. If you want judge-based scoring, plug it in behind the same
function signature.
"""
import re
from typing import List, Set

import numpy as np

from src.answer import Answer

_WORD = re.compile(r"[a-zàâäéèêëîïôöùûüç0-9]+")


def _tokens(text: str) -> Set[str]:
    return set(_WORD.findall(text.lower()))


def recall_at_k(retrieved: List[str], expected: List[str], k: int) -> float:
    if not expected:
        return 1.0 if not retrieved else 0.0
    got = set(retrieved[:k])
    return len(got & set(expected)) / len(expected)


def mrr(retrieved: List[str], expected: List[str]) -> float:
    expected = set(expected)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def sentence_support(sentence: str, chunk_text: str,
                     threshold: float = 0.4) -> bool:
    """Heuristic: a sentence is 'supported' if enough of its content
    tokens appear in the cited chunk. Crude, but catches blatant
    hallucinations and runs in microseconds."""
    s_tokens = {t for t in _tokens(sentence) if len(t) > 3}
    if not s_tokens:
        return True
    c_tokens = _tokens(chunk_text)
    return len(s_tokens & c_tokens) / len(s_tokens) >= threshold


def faithfulness(answer: Answer) -> float:
    """Fraction of cited claims supported by their chunk.

    Uncited sentences are ignored here (that's a coverage problem, not
    a faithfulness problem). An answer with no citations scores 0 unless
    it refused to answer, which scores 1.
    """
    if answer.refused:
        return 1.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.text)
                 if s.strip()]
    if not sentences:
        return 0.0
    supported, checked = 0, 0
    for sent in sentences:
        cites = [int(m) for m in re.findall(r"\[(\d+)\]", sent)]
        cites = [c for c in cites if 1 <= c <= len(answer.hits)]
        if not cites:
            continue
        checked += 1
        if any(sentence_support(sent, answer.hits[c - 1].chunk.text)
               for c in cites):
            supported += 1
    if checked == 0:
        return 0.0
    return supported / checked


def aggregate(scores: List[float]) -> dict:
    arr = np.asarray(scores, dtype=float)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "n": len(scores),
    }
