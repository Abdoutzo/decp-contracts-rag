# Eval report

_Generated from `evals/questions.jsonl` (20 questions, 14 documents)._

| strategy | chunks | recall@5 | MRR | latency p50 |
|---|---|---|---|---|
| fixed | 26 | 0.821 | 0.651 | 0.22s |
| recursive | 14 | 0.771 | 0.572 | 0.18s |
| semantic | 23 | 0.842 | 0.653 | 0.18s |

## Notes

- Retrieval metrics are deterministic (no LLM involved) and run in CI.
- Faithfulness uses a token-overlap heuristic; treat it as a smoke
  test for hallucinations, not a ground truth.
- The unanswerable question checks refusal behavior: the model must
  say it doesn't know rather than invent an answer.
- Chunk counts differ per strategy; fewer, longer chunks are not
  automatically better — that's what recall@5 is for.
