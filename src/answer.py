"""Answer generation with citations.

Retrieves top-k chunks, stuffs them into a prompt, and asks the LLM to
answer *only* from those chunks, citing sources as [1], [2], ...

Every answer comes back with latency and an estimated cost, because
"it works" is not the same as "it works at a price I can afford".
"""
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

import requests

import config
from src.embeddings import Embedder
from src.store import Hit, VectorStore

SYSTEM_PROMPT = """Tu es un assistant qui répond à des questions sur des marchés publics français.
Règles strictes :
- Réponds UNIQUEMENT à partir des extraits fournis ci-dessous.
- Si la réponse ne figure pas dans les extraits, dis exactement : "Je ne sais pas, ce n'est pas dans les documents."
- Cite chaque affirmation avec le numéro de l'extrait entre crochets, ex : [1], [3].
- Réponds en français, de façon concise."""


def _build_prompt(question: str, hits: List[Hit]) -> str:
    sources = "\n\n".join(
        f"[{i + 1}] (doc {h.chunk.doc_id})\n{h.chunk.text}"
        for i, h in enumerate(hits)
    )
    return f"{SYSTEM_PROMPT}\n\nExtraits :\n{sources}\n\nQuestion : {question}\nRéponse :"


@dataclass
class Answer:
    text: str
    citations: List[int] = field(default_factory=list)
    hits: List[Hit] = field(default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    refused: bool = False  # model said "I don't know"


_CITATION_RE = re.compile(r"\[(\d+)\]")


class LLMClient:
    """Minimal REST client for Mistral / OpenAI chat completions.

    Deliberately raw `requests` instead of SDKs: fewer dependencies,
    and the call shape is identical for both providers.
    """

    def __init__(self, provider: str = config.LLM_PROVIDER):
        self.provider = provider
        if provider == "mistral":
            self.url = "https://api.mistral.ai/v1/chat/completions"
            self.key = config.MISTRAL_API_KEY
            self.model = config.MISTRAL_MODEL
        elif provider == "openai":
            self.url = "https://api.openai.com/v1/chat/completions"
            self.key = config.OPENAI_API_KEY
            self.model = config.OPENAI_MODEL
        elif provider == "none":
            self.url = self.key = self.model = ""
        else:
            raise ValueError(f"unknown LLM provider: {provider}")

    @property
    def enabled(self) -> bool:
        return self.provider != "none" and bool(self.key)

    def complete(self, prompt: str) -> tuple[str, int, int]:
        """Returns (text, prompt_tokens, completion_tokens)."""
        resp = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return (
            data["choices"][0]["message"]["content"],
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prices = config.TOKEN_PRICES.get(self.model, {"input": 0, "output": 0})
        return (prompt_tokens / 1000 * prices["input"]
                + completion_tokens / 1000 * prices["output"])


def answer_question(question: str, store: VectorStore, embedder: Embedder,
                    llm: Optional[LLMClient] = None,
                    k: int = config.TOP_K) -> Answer:
    """Retrieve + generate. If no LLM is configured, returns the retrieved
    hits with an empty answer (retrieval-only mode, still useful for evals)."""
    t0 = time.time()
    q_emb = embedder.embed([question])[0]
    retrieve_s = time.time() - t0
    hits = store.search(q_emb, k)

    if llm is None or not llm.enabled:
        return Answer(text="", hits=hits, latency_s=retrieve_s)

    prompt = _build_prompt(question, hits)
    t1 = time.time()
    text, pt, ct = llm.complete(prompt)
    gen_s = time.time() - t1

    refused = "je ne sais pas" in text.lower()
    return Answer(
        text=text,
        citations=sorted({int(m) for m in _CITATION_RE.findall(text)}),
        hits=hits,
        latency_s=retrieve_s + gen_s,
        prompt_tokens=pt,
        completion_tokens=ct,
        cost_usd=llm.estimate_cost(pt, ct),
        refused=refused,
    )
