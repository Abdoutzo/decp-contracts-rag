"""Build a FAISS index from the corpus.

Usage:
    python scripts/build_index.py --strategy recursive --out index/recursive
    python scripts/build_index.py --strategy semantic --corpus data/sample/contracts.json

The semantic strategy needs embeddings at chunk time, so the embedder is
created first and passed to the chunker.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config
from src.embeddings import Embedder
from src.ingest import chunk_corpus, load_sample_corpus
from src.store import VectorStore


def build(corpus_path: str, strategy: str, out_dir: str) -> VectorStore:
    docs = load_sample_corpus(corpus_path)
    print(f"loaded {len(docs)} documents from {corpus_path}")

    embedder = Embedder()
    embed_fn = embedder.embed if strategy == "semantic" else None

    chunks = chunk_corpus(docs, strategy=strategy, embed_fn=embed_fn)
    print(f"strategy={strategy}: {len(chunks)} chunks")

    embeddings = embedder.embed([c.text for c in chunks])
    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embeddings)

    os.makedirs(out_dir, exist_ok=True)
    import faiss
    faiss.write_index(store.index, os.path.join(out_dir, "index.faiss"))
    with open(os.path.join(out_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"chunk_id": c.chunk_id, "doc_id": c.doc_id,
              "text": c.text, "position": c.position} for c in chunks],
            f, ensure_ascii=False, indent=1,
        )
    meta = {
        "strategy": strategy,
        "embedding_model": embedder.model_name,
        "n_docs": len(docs),
        "n_chunks": len(chunks),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"index written to {out_dir}")
    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=config.CHUNK_STRATEGY,
                        choices=["fixed", "recursive", "semantic"])
    parser.add_argument("--corpus", default=config.EVAL_CORPUS)
    parser.add_argument("--out", default="index/recursive")
    args = parser.parse_args()
    build(args.corpus, args.strategy, args.out)
