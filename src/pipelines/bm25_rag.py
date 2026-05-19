"""
Pipeline 3: BM25 Lexical RAG
──────────────────────────────
Classic keyword-based retrieval using TF-IDF/BM25 scoring.
No embeddings, no vectors — pure term frequency matching.

Retrieval priority:
  1. bm25s library (new.md B.4/D.4 — "benchmarks within striking distance
     of Elasticsearch on a single node while being pure-Python")
  2. Custom InMemoryBM25 (Okapi BM25) — fallback when bm25s not installed

Falls back to in-memory BM25 if Elasticsearch is unavailable.
"""

from __future__ import annotations


import json
import math
import time
import re
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from src.pipelines.base import (
    RAGPipeline, RAGResponse, IngestionReport, UpdateReport,
)
from src.utils.llm_client import LLMClient


ANSWER_PROMPT = """Answer the question using ONLY the retrieved passages below.
Cite the source document and section for each claim.
If the passages don't contain sufficient information, say "Insufficient context."

QUESTION: {question}

RETRIEVED PASSAGES:
{passages}

ANSWER:"""


class InMemoryBM25:
    """
    Simple in-memory BM25 implementation for when Elasticsearch is unavailable.
    Implements Okapi BM25 scoring with standard parameters.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: list[dict] = []       # [{id, text, metadata, tokens}]
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_freqs: Counter = Counter()   # term -> num docs containing it

    def add_document(self, doc_id: str, text: str, metadata: dict = None):
        tokens = self._tokenize(text)
        self.documents.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata or {},
            "tokens": tokens,
            "length": len(tokens),
        })
        # Update doc frequencies
        unique_terms = set(tokens)
        for term in unique_terms:
            self.doc_freqs[term] += 1
        self.doc_count = len(self.documents)
        total_len = sum(d["length"] for d in self.documents)
        self.avg_doc_len = total_len / self.doc_count if self.doc_count else 0

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = self._tokenize(query)
        scores = []

        for doc in self.documents:
            score = self._score_document(doc, query_tokens)
            scores.append({"doc": doc, "score": score})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def _score_document(self, doc: dict, query_tokens: list[str]) -> float:
        score = 0.0
        doc_tokens = doc["tokens"]
        doc_len = doc["length"]
        tf_counter = Counter(doc_tokens)

        for term in query_tokens:
            if term not in tf_counter:
                continue
            tf = tf_counter[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            tf_norm = (tf * (self.k1 + 1)) / (
                tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avg_doc_len))
            )
            score += idf * tf_norm

        return score

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 1]


def _try_import_bm25s():
    """Try importing bm25s; return (module, Stemmer) or (None, None)."""
    try:
        import bm25s
        try:
            import Stemmer
            stemmer = Stemmer.Stemmer("english")
        except ImportError:
            stemmer = None
        return bm25s, stemmer
    except ImportError:
        return None, None


class BM25RAG(RAGPipeline):
    """
    BM25 Lexical RAG pipeline.

    Ingestion: Chunk documents → index with BM25 (bm25s or in-memory fallback)
    Retrieval: BM25 keyword search → top-K chunks
    Generation: LLM answers from retrieved chunks

    Uses bm25s library if installed (new.md B.4/D.4 — significantly faster
    than Elasticsearch single-node while being pure-Python).
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})
        ing_cfg = pipeline_cfg.get("ingestion", {})
        ret_cfg = pipeline_cfg.get("retrieval", {})
        gen_cfg = pipeline_cfg.get("generation", {})

        self.chunk_size = ing_cfg.get("chunk_size", 512)
        self.chunk_overlap = ing_cfg.get("chunk_overlap", 64)
        self.top_k = ret_cfg.get("top_k", 5)
        self.gen_model = gen_cfg.get("model", "qwen3:8b")
        self.system_prompt = gen_cfg.get("system_prompt", "")

        self._gen_client = LLMClient(model=self.gen_model, temperature=0.0)
        self._bm25 = InMemoryBM25()
        self._index_dir: str | None = None
        self._chunk_count = 0

        # bm25s fast-path (new.md B.4/D.4)
        self._bm25s_mod, self._stemmer = _try_import_bm25s()
        self._bm25s_retriever = None   # bm25s.BM25 instance
        self._bm25s_corpus: list[str] = []
        self._bm25s_chunk_meta: list[dict] = []
        if self._bm25s_mod:
            logger.info("BM25RAG: using bm25s library (fast path)")
        else:
            logger.info("BM25RAG: bm25s not installed, using InMemoryBM25 fallback")

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Chunk documents and build BM25 index (bm25s fast-path or InMemoryBM25)."""
        start = time.perf_counter()
        corpus_dir = Path(corpus_path)
        self._bm25 = InMemoryBM25()  # Reset fallback
        self._bm25s_corpus = []
        self._bm25s_chunk_meta = []
        self._chunk_count = 0

        manifest_path = corpus_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            doc_files = [corpus_dir / f"{d['doc_id']}.json" for d in manifest["documents"]]
        else:
            doc_files = sorted(corpus_dir.glob("*.json"))

        doc_count = 0
        for doc_file in doc_files:
            if doc_file.name == "manifest.json":
                continue
            with open(doc_file) as f:
                doc_data = json.load(f)

            doc_id = doc_data.get("doc_id", doc_file.stem)
            full_text = doc_data.get("full_text", "")
            title = doc_data.get("title", doc_id)

            chunks = self._chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                meta = {
                    "source_doc": doc_id,
                    "title": title,
                    "chunk_index": i,
                    "domain": doc_data.get("domain", ""),
                }
                if self._bm25s_mod:
                    # bm25s path — collect texts + metadata for batch indexing
                    self._bm25s_corpus.append(chunk)
                    self._bm25s_chunk_meta.append({"id": chunk_id, "text": chunk, **meta})
                else:
                    # fallback path
                    self._bm25.add_document(doc_id=chunk_id, text=chunk, metadata=meta)

            self._chunk_count += len(chunks)
            doc_count += 1
            logger.debug(f"Indexed {doc_id}: {len(chunks)} chunks")

        # Build bm25s index (new.md B.4: batch indexing is faster)
        if self._bm25s_mod and self._bm25s_corpus:
            bm25s = self._bm25s_mod
            tokens = bm25s.tokenize(
                self._bm25s_corpus, stopwords="en", stemmer=self._stemmer
            )
            self._bm25s_retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
            self._bm25s_retriever.index(tokens)
            logger.debug(f"bm25s index built: {len(self._bm25s_corpus)} chunks")

        elapsed = time.perf_counter() - start

        # Save index metadata
        self._index_dir = str(corpus_dir / ".bm25_index")
        Path(self._index_dir).mkdir(exist_ok=True)
        meta_path = Path(self._index_dir) / "index_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "doc_count": doc_count,
                "chunk_count": self._chunk_count,
                "chunk_size": self.chunk_size,
                "vocab_size": len(self._bm25.doc_freqs),
            }, f, indent=2)
        index_size = meta_path.stat().st_size

        self._is_ingested = True
        logger.info(
            f"BM25 ingestion: {doc_count} docs, {self._chunk_count} chunks "
            f"in {elapsed:.1f}s"
        )

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=doc_count,
            ingestion_time_seconds=elapsed,
            index_size_bytes=index_size,
            index_artifacts={"index_dir": self._index_dir},
            metadata={"chunk_count": self._chunk_count},
        )

    def query(self, question: str) -> RAGResponse:
        """BM25 search → top-K chunks → LLM generation."""
        start = time.perf_counter()

        contexts = []
        references = []

        if self._bm25s_mod and self._bm25s_retriever is not None:
            # Fast path: bm25s library (new.md B.4/D.4)
            bm25s = self._bm25s_mod
            q_tokens = bm25s.tokenize([question], stopwords="en", stemmer=self._stemmer)
            results, scores = self._bm25s_retriever.retrieve(
                q_tokens, k=min(self.top_k, len(self._bm25s_corpus))
            )
            for idx, score in zip(results[0], scores[0]):
                meta = self._bm25s_chunk_meta[idx]
                contexts.append(meta["text"])
                references.append({
                    "chunk_id": meta["id"],
                    "source_doc": meta.get("source_doc", ""),
                    "title": meta.get("title", ""),
                    "bm25_score": round(float(score), 4),
                })
        else:
            # Fallback: custom InMemoryBM25
            results = self._bm25.search(question, top_k=self.top_k)
            for r in results:
                doc = r["doc"]
                contexts.append(doc["text"])
                references.append({
                    "chunk_id": doc["id"],
                    "source_doc": doc["metadata"].get("source_doc", ""),
                    "title": doc["metadata"].get("title", ""),
                    "bm25_score": round(r["score"], 4),
                })

        # Generate answer
        passages_text = "\n\n---\n\n".join(
            f"[Source: {ref['title']}] (BM25 score: {ref['bm25_score']})\n{ctx}"
            for ctx, ref in zip(contexts, references)
        )

        gen_prompt = ANSWER_PROMPT.format(
            question=question, passages=passages_text,
        )
        gen_resp = self._gen_client.generate(gen_prompt, system_prompt=self.system_prompt)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            answer=gen_resp.content,
            retrieved_contexts=contexts,
            source_references=references,
            latency_ms=elapsed_ms,
            tokens_used={
                "input": gen_resp.input_tokens,
                "output": gen_resp.output_tokens,
                "total": gen_resp.total_tokens,
            },
            cost_usd=gen_resp.cost_usd,
        )

    def add_documents(self, document_paths: list[str]) -> UpdateReport:
        """Add new documents — just chunk and add to index."""
        start = time.perf_counter()

        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            full_text = doc_data.get("full_text", "")
            chunks = self._chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                self._bm25.add_document(
                    f"{doc_id}_chunk_{i}", chunk,
                    metadata={"source_doc": doc_id, "title": doc_data.get("title", "")},
                )

        elapsed = time.perf_counter() - start
        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=False,
            new_index_size_bytes=0,
        )

    def _chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks by word count."""
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)

        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))

        return chunks if chunks else [text]
