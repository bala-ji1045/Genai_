"""Knowledge source 2 of 2: a RAG retriever — the Module 3 lab pipeline, packaged.

    documents (data/docs/*.md, *.txt)
        -> chunk (RecursiveCharacterTextSplitter), each chunk prefixed with its document's title
        -> dense (FAISS + MiniLM)  +  sparse (BM25)
        -> hybrid fusion (Reciprocal Rank Fusion)
        -> cross-encoder re-rank
        -> top-k chunks, each tagged with its source

The index is rebuilt automatically when files in the docs folder change, so the Streamlit app and the
MCP server (two separate processes) both pick up an uploaded document without a restart.
"""
import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from . import config

DOC_SUFFIXES = {".md", ".txt"}


@dataclass
class Hit:
    text: str
    source: str        # "file.md#chunk-number"
    score: float
    stage: str         # which retriever produced the score


# ---------------------------------------------------------------- loading and chunking
def load_documents(docs_dir: Path = config.DOCS_DIR) -> list:
    docs = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() in DOC_SUFFIXES:
            docs.append(Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name}))
    for d in docs:
        d.metadata["title"] = doc_title(d)
    return docs


def chunk_documents(docs: list) -> list:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    counters = {}
    for i, c in enumerate(chunks):                       # a stable id per chunk, used by fusion and for citations
        n = counters[c.metadata["source"]] = counters.get(c.metadata["source"], 0) + 1
        c.metadata.update(id=i, chunk=n)
        # Contextual chunk header: a chunk like "## Power  It has a 3,200 mAh battery" never names the product,
        # so neither BM25 nor the embedding can match it to "Zylarian Communicator battery". Prefix the title.
        title = c.metadata["title"]
        if not c.page_content.lstrip("# ").startswith(title):
            c.page_content = f"{title}\n{c.page_content}"
    return chunks


def doc_title(doc) -> str:
    first = doc.page_content.strip().splitlines()[0]
    return first.lstrip("# ").strip() if first.startswith("#") else doc.metadata["source"].rsplit(".", 1)[0].replace("_", " ")


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def cite(doc) -> str:
    return f"{doc.metadata['source']}#{doc.metadata['chunk']}"


# ---------------------------------------------------------------- models (loaded once per process)
@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(config.RERANK_MODEL)


# ---------------------------------------------------------------- the index
class RAGIndex:
    def __init__(self, docs_dir: Path = config.DOCS_DIR):
        from langchain_community.vectorstores import FAISS
        from rank_bm25 import BM25Okapi
        self.docs = load_documents(docs_dir)
        if not self.docs:
            raise RuntimeError(f"No .md or .txt documents found in {docs_dir}")
        self.chunks = chunk_documents(self.docs)
        self.vectorstore = FAISS.from_documents(self.chunks, get_embeddings())
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])

    def dense(self, query: str, k: int = 5) -> list:
        pairs = self.vectorstore.similarity_search_with_score(query, k=k)       # score = L2 distance, lower is closer
        return [Hit(d.page_content, cite(d), float(s), "dense (L2 distance)") for d, s in pairs]

    def sparse(self, query: str, k: int = 5) -> list:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [Hit(self.chunks[i].page_content, cite(self.chunks[i]), float(scores[i]), "BM25") for i in top]

    def hybrid(self, query: str, k: int = 6, top_k: int = 5, k_const: int = 60) -> list:
        """Reciprocal Rank Fusion: combine the dense and BM25 rankings by position, not raw score."""
        scores, lookup = {}, {}
        for ranked in (self.dense(query, k), self.sparse(query, k)):
            for rank, hit in enumerate(ranked):
                scores[hit.source] = scores.get(hit.source, 0) + 1 / (k_const + rank + 1)
                lookup[hit.source] = hit
        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [Hit(lookup[s].text, s, scores[s], "hybrid (RRF)") for s in best]

    def rerank(self, query: str, candidates: list, top_k: int = config.TOP_K) -> list:
        scores = get_reranker().predict([[query, h.text] for h in candidates])
        order = np.argsort(scores)[::-1][:top_k]
        return [Hit(candidates[i].text, candidates[i].source, float(scores[i]), "cross-encoder") for i in order]

    def search(self, query: str, top_k: int = config.TOP_K, use_reranker: bool = True) -> list:
        candidates = self.hybrid(query, k=6, top_k=6 if use_reranker else top_k)
        return self.rerank(query, candidates, top_k) if use_reranker else candidates


_lock = threading.Lock()
_cache = {"fingerprint": None, "index": None}


def docs_fingerprint(docs_dir: Path = config.DOCS_DIR) -> tuple:
    return tuple((p.name, p.stat().st_mtime_ns, p.stat().st_size)
                 for p in sorted(docs_dir.glob("*")) if p.suffix.lower() in DOC_SUFFIXES)


def get_index() -> RAGIndex:
    """The current index; rebuilt only when the docs folder has changed."""
    with _lock:
        fp = docs_fingerprint()
        if _cache["fingerprint"] != fp:
            _cache["index"], _cache["fingerprint"] = RAGIndex(), fp
        return _cache["index"]


def format_hits(hits: list) -> str:
    """What the research agent sees: each chunk tagged with its source, so the answer can cite it."""
    if not hits:
        return "No relevant passages found in the knowledge base."
    return "\n".join(f"[source: {h.source}] {' '.join(h.text.split())}" for h in hits)


def search_text(query: str, top_k: int = config.TOP_K) -> str:
    return format_hits(get_index().search(query, top_k=top_k))
