"""Streamlit demo: ask questions about French public contracts.

Run locally:  streamlit run app.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import config
from src.answer import LLMClient, answer_question
from src.embeddings import Embedder
from src.ingest import chunk_corpus, load_sample_corpus
from src.store import VectorStore

st.set_page_config(page_title="Contrats publics — RAG demo", layout="wide")
st.title("Recherche dans les marchés publics (démo RAG)")

with st.sidebar:
    st.header("Réglages")
    strategy = st.selectbox("Chunking", ["recursive", "fixed", "semantic"],
                            index=["recursive", "fixed", "semantic"].index(config.CHUNK_STRATEGY))
    k = st.slider("Top-k", 1, 10, config.TOP_K)
    st.caption("Corpus : 14 marchés publics fictifs (data/sample). "
               "Configurez une clé API dans .env pour la génération de réponses.")


@st.cache_resource
def load_index(strategy: str):
    docs = load_sample_corpus()
    embedder = Embedder()
    embed_fn = embedder.embed if strategy == "semantic" else None
    chunks = chunk_corpus(docs, strategy=strategy, embed_fn=embed_fn)
    store = VectorStore(dim=embedder.dim)
    store.add(chunks, embedder.embed([c.text for c in chunks]))
    return store, embedder


with st.spinner("Chargement de l'index…"):
    store, embedder = load_index(strategy)

llm = LLMClient()
if not llm.enabled:
    st.info("Pas de clé API : mode récupération seule (les extraits pertinents "
            "s'affichent sans réponse générée).")

question = st.text_input("Votre question",
                         "Quels marchés ont été attribués à BatiRhône SAS ?")
if st.button("Chercher") and question:
    with st.spinner("Recherche…"):
        ans = answer_question(question, store, embedder, llm, k=k)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Réponse")
        if ans.text:
            st.write(ans.text)
        else:
            st.write("_Génération désactivée — voir les extraits._")
        st.caption(f"Latence : {ans.latency_s:.2f}s"
                   + (f" · coût estimé : ${ans.cost_usd:.5f}" if ans.cost_usd else ""))
    with col2:
        st.subheader("Sources")
        for i, hit in enumerate(ans.hits, start=1):
            with st.expander(f"[{i}] {hit.chunk.doc_id} (score {hit.score:.3f})"):
                st.write(hit.chunk.text)
