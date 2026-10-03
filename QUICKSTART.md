# Quickstart

Get from zero to a working eval in about 5 minutes.

```bash
git clone https://github.com/Abdoutzo/decp-contracts-rag
cd decp-contracts-rag

python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows

pip install -r requirements.txt
cp .env.example .env
```

## 1. Run the eval suite (no API key needed)

```bash
python evals/run_eval.py
```

This builds three indexes (fixed / recursive / semantic chunking), runs the
20-question eval set against each, and writes `evals/report.md` with
recall@5, MRR and latency. First run downloads the embedding model
(~400 MB), subsequent runs are fast.

## 2. Unit tests

```bash
pytest tests/ -q
```

## 3. Try the demo

```bash
streamlit run app.py
```

Without an API key it runs in retrieval-only mode: you see the top-k
chunks for your question, no generated answer.

## 4. Enable answer generation (optional)

Add your key to `.env`:

```env
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_key_here
```

then:

```bash
python evals/run_eval.py --with-answers
```

This adds faithfulness scoring, refusal checks on the unanswerable
question, and per-query cost estimates to the report.
