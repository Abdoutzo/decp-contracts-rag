"""Eval harness: compare chunking strategies on the question set.

For each strategy it builds an in-memory index, runs every question,
and measures:
  - retrieval: recall@5, MRR
  - answers (only if an LLM key is configured): faithfulness, refusal
    correctness on the unanswerable question, latency, estimated cost

Writes a markdown report to evals/report.md.

Usage:
    python evals/run_eval.py                          # retrieval only
    python evals/run_eval.py --with-answers            # needs LLM key in .env
    python evals/run_eval.py --strategies fixed recursive
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.answer import LLMClient, answer_question
from src.embeddings import Embedder
from src.ingest import chunk_corpus, load_sample_corpus
from src.metrics import aggregate, faithfulness, mrr, recall_at_k
from src.store import VectorStore


def load_questions(path: str = config.EVAL_QUESTIONS):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_strategy(strategy: str, questions, embedder: Embedder,
                 llm, with_answers: bool, k: int = 5) -> dict:
    docs = load_sample_corpus()
    embed_fn = embedder.embed if strategy == "semantic" else None
    chunks = chunk_corpus(docs, strategy=strategy, embed_fn=embed_fn)
    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embedder.embed([c.text for c in chunks]))

    recalls, mrrs, faithfuls, latencies, costs = [], [], [], [], []
    refusals_ok = 0
    unanswerable = 0
    for q in questions:
        ans = answer_question(q["question"], store, embedder, llm, k=k)
        doc_ids = store.doc_ids(ans.hits)
        recalls.append(recall_at_k(doc_ids, q["expected_doc_ids"], k))
        mrrs.append(mrr(doc_ids, q["expected_doc_ids"]))
        latencies.append(ans.latency_s)
        if with_answers and llm.enabled:
            faithfuls.append(faithfulness(ans))
            costs.append(ans.cost_usd)
            if not q["expected_doc_ids"]:
                unanswerable += 1
                refusals_ok += int(ans.refused)

    result = {
        "strategy": strategy,
        "n_chunks": len(chunks),
        "recall@5": aggregate(recalls),
        "mrr": aggregate(mrrs),
        "latency_s": aggregate(latencies),
    }
    if faithfuls:
        result["faithfulness"] = aggregate(faithfuls)
        result["mean_cost_usd"] = sum(costs) / len(costs)
        result["refusal_accuracy"] = refusals_ok / unanswerable if unanswerable else None
    return result


def write_report(results: list, path: str, with_answers: bool):
    lines = [
        "# Eval report",
        "",
        f"_Generated from `{config.EVAL_QUESTIONS}` "
        f"({len(load_questions())} questions, {len(load_sample_corpus())} documents)._",
        "",
        "| strategy | chunks | recall@5 | MRR | latency p50 |"
        + (" faithfulness | cost/query |" if with_answers else ""),
        "|---|---|---|---|---|" + ("---|---|" if with_answers else ""),
    ]
    for r in results:
        row = (f"| {r['strategy']} | {r['n_chunks']} "
               f"| {r['recall@5']['mean']:.3f} | {r['mrr']['mean']:.3f} "
               f"| {r['latency_s']['p50']:.2f}s |")
        if with_answers and "faithfulness" in r:
            row += (f" {r['faithfulness']['mean']:.3f} "
                    f"| ${r['mean_cost_usd']:.5f} |")
        lines.append(row)
    lines += [
        "",
        "## Notes",
        "",
        "- Retrieval metrics are deterministic (no LLM involved) and run in CI.",
        "- Faithfulness uses a token-overlap heuristic; treat it as a smoke",
        "  test for hallucinations, not a ground truth.",
        "- The unanswerable question checks refusal behavior: the model must",
        "  say it doesn't know rather than invent an answer.",
        "- Chunk counts differ per strategy; fewer, longer chunks are not",
        "  automatically better — that's what recall@5 is for.",
    ]
    if any(r.get("refusal_accuracy") is not None for r in results):
        lines += ["",
                  "## Refusal accuracy (unanswerable questions)",
                  ""]
        for r in results:
            if r.get("refusal_accuracy") is not None:
                lines.append(f"- {r['strategy']}: {r['refusal_accuracy']:.0%}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+",
                        default=["fixed", "recursive", "semantic"])
    parser.add_argument("--with-answers", action="store_true",
                        help="generate answers with the LLM (needs API key)")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions()
    print(f"{len(questions)} questions loaded")
    embedder = Embedder()
    llm = LLMClient() if args.with_answers else None
    if args.with_answers and not llm.enabled:
        print("warning: --with-answers set but no LLM key found; "
              "running retrieval-only.")
        llm = None

    t0 = time.time()
    results = []
    for strategy in args.strategies:
        print(f"\n=== strategy: {strategy} ===")
        r = run_strategy(strategy, questions, embedder, llm,
                         with_answers=args.with_answers, k=args.k)
        print(f"recall@5={r['recall@5']['mean']:.3f} "
              f"mrr={r['mrr']['mean']:.3f} "
              f"chunks={r['n_chunks']}")
        results.append(r)
    print(f"\ntotal: {time.time() - t0:.1f}s")
    write_report(results, config.EVAL_REPORT,
                 with_answers=args.with_answers and llm is not None and llm.enabled)


if __name__ == "__main__":
    main()
