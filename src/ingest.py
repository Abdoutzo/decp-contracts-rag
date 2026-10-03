"""Document loading and chunking.

Three chunking strategies, because "just split every 500 chars" is the
first thing that breaks in a real RAG system and I wanted to measure it
instead of arguing about it.

A Document is the unit we index. A Chunk is what retrieval returns.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Callable, List

import config


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    position: int  # index of the chunk inside its document


def load_sample_corpus(path: str = config.EVAL_CORPUS) -> List[Document]:
    """Load the sample corpus (JSON list of contract records)."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    docs = []
    for rec in records:
        text = "\n".join(
            f"{k}: {v}" for k, v in rec.items() if k != "id" and v
        )
        docs.append(
            Document(
                doc_id=rec["id"],
                title=rec.get("objet", rec["id"]),
                text=text,
                metadata={k: v for k, v in rec.items() if k != "id"},
            )
        )
    return docs


def chunk_fixed(text: str, size: int = config.FIXED_CHUNK_SIZE,
                overlap: int = config.FIXED_CHUNK_OVERLAP) -> List[str]:
    """Naive sliding window over characters. Baseline, nothing more."""
    if not text:
        return []
    chunks, start = [], 0
    step = max(size - overlap, 1)
    while start < len(text):
        chunks.append(text[start:start + size])
        start += step
    return chunks


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def chunk_recursive(text: str, size: int = config.RECURSIVE_CHUNK_SIZE) -> List[str]:
    """Sentence-aware packing: fill chunks up to `size` chars without
    cutting sentences in half. Paragraph breaks are preferred boundaries."""
    if not text:
        return []
    chunks, current = [], ""
    # paragraphs first, then sentences inside oversized paragraphs
    for para in [p for p in text.split("\n") if p.strip()]:
        for sent in _sentences(para):
            if len(current) + len(sent) + 1 <= size:
                current = f"{current} {sent}".strip()
            else:
                if current:
                    chunks.append(current)
                # a single sentence longer than size gets hard-split
                while len(sent) > size:
                    chunks.append(sent[:size])
                    sent = sent[size:]
                current = sent
    if current:
        chunks.append(current)
    return chunks


def chunk_semantic(text: str,
                   embed_fn: Callable[[List[str]], "object"],
                   threshold: float = config.SEMANTIC_BREAK_THRESHOLD,
                   window: int = 2) -> List[str]:
    """Split where meaning shifts: embed sentences, cut at points where
    cosine similarity between neighboring windows drops below `threshold`.

    Needs an embedding function (imported lazily by callers to avoid a
    hard dependency on sentence-transformers at import time).
    """
    import numpy as np

    sents = _sentences(text.replace("\n", " "))
    if len(sents) < 2:
        return sents
    embs = embed_fn(sents)
    embs = np.asarray(embs, dtype=float)
    norms = (embs ** 2).sum(axis=1, keepdims=True) ** 0.5
    embs = embs / (norms + 1e-9)

    chunks, current = [], [sents[0]]
    for i in range(1, len(sents)):
        lo = max(0, i - window)
        left = embs[lo:i].mean(axis=0)
        right = embs[i:i + window].mean(axis=0)
        sim = float(left @ right)
        if sim < threshold and current:
            chunks.append(" ".join(current))
            current = [sents[i]]
        else:
            current.append(sents[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(doc: Document, strategy: str = config.CHUNK_STRATEGY,
                   embed_fn=None) -> List[Chunk]:
    """Chunk one document with the given strategy."""
    if strategy == "fixed":
        texts = chunk_fixed(doc.text)
    elif strategy == "recursive":
        texts = chunk_recursive(doc.text)
    elif strategy == "semantic":
        if embed_fn is None:
            raise ValueError("semantic chunking needs an embed_fn")
        texts = chunk_semantic(doc.text, embed_fn)
    else:
        raise ValueError(f"unknown chunking strategy: {strategy}")
    return [
        Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id,
              text=t, position=i)
        for i, t in enumerate(texts)
    ]


def chunk_corpus(docs: List[Document], strategy: str = config.CHUNK_STRATEGY,
                 embed_fn=None) -> List[Chunk]:
    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, strategy, embed_fn))
    return chunks
