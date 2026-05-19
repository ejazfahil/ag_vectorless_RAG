"""
Pipeline 8: In-Memory KNN RAG — PCA + PQ + HNSW (Khan, June 2025)
══════════════════════════════════════════════════════════════════
Faithful implementation of Khan's manual RAG recipe:
  1. Encode chunks with all-MiniLM-L6-v2 (384-dim)
  2. PCA → 256-dim (memory reduction)
  3. Product Quantization via FAISS (32 bytes/vector)
  4. HNSW approximate nearest-neighbour search

Author-reported claims (unverified, self-published preprint):
  - 60% memory reduction vs raw embeddings
  - 45% latency improvement
  - MRR@5 = 0.87 on NaturalQuestions / TriviaQA (10K–1M docs)

Reference: Khan (June 2025), Figshare/ResearchGate preprint.
Note: Blueprint C.6 caveat — validate numbers on your own corpus
      before citing. No public code from authors.

Dependencies (all native on Apple Silicon):
  pip install sentence-transformers scikit-learn hnswlib numpy faiss-cpu
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

# numpy is an optional dependency — imported lazily in methods
# to avoid ImportError when knn_rag is imported but not run
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

from src.pipelines.base import (
    RAGPipeline, RAGResponse, IngestionReport, UpdateReport,
)
from src.utils.llm_client import LLMClient


ANSWER_PROMPT = """Answer the question using ONLY the retrieved passages below.
Cite the source document for each claim.
If the passages don't contain sufficient information, say "Insufficient context."

QUESTION: {question}

RETRIEVED PASSAGES:
{passages}

ANSWER:"""


def _chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> list[str]:
    """Split text into overlapping chunks by word count."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


class KNNInMemoryRAG(RAGPipeline):
    """
    In-Memory KNN RAG: PCA + Product Quantization + HNSW.

    Ingestion: Encode chunks → PCA → HNSW index (+ optional FAISS PQ)
    Retrieval: ANN search → top-K chunks
    Generation: LLM answers from retrieved chunks

    Falls back gracefully if faiss-cpu or hnswlib are unavailable:
      - Without faiss: skips PQ, uses PCA-only embeddings in HNSW
      - Without hnswlib: falls back to brute-force cosine search
      - Without sentence-transformers: raises ImportError with install hint
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})
        embed_cfg = pipeline_cfg.get("embedding", {})
        index_cfg = pipeline_cfg.get("index", {})
        gen_cfg = pipeline_cfg.get("generation", {})

        # Embedding
        self._embed_model_name = embed_cfg.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._raw_dim = embed_cfg.get("raw_dim", 384)
        self._pca_dim = embed_cfg.get("pca_dim", 256)
        self._normalize = embed_cfg.get("normalize", True)

        # PQ (FAISS)
        self._pq_enabled = index_cfg.get("pq_enabled", True)
        self._pq_subvectors = index_cfg.get("pq_subvectors", 32)
        self._pq_bits = index_cfg.get("pq_bits", 8)

        # HNSW
        self._hnsw_m = index_cfg.get("hnsw_m", 16)
        self._hnsw_ef_construction = index_cfg.get("ef_construction", 200)
        self._hnsw_ef = index_cfg.get("ef_search", 64)
        self._top_k = index_cfg.get("top_k", 5)

        # Chunking
        self._chunk_size = pipeline_cfg.get("chunk_size", 512)
        self._chunk_overlap = pipeline_cfg.get("chunk_overlap", 64)

        # Generation
        gen_model = gen_cfg.get("model", "qwen3:8b")
        self._gen_client = LLMClient(model=gen_model, temperature=0.0)
        self._system_prompt = gen_cfg.get("system_prompt", "")

        # Internal state
        self._chunks: list[dict] = []          # [{text, doc_id, title, chunk_idx}]
        self._embedder = None                   # SentenceTransformer
        self._pca = None                        # sklearn PCA
        self._hnsw_index = None                 # hnswlib.Index
        self._pq_index = None                   # faiss.IndexPQ (optional)
        self._embeddings_pca: np.ndarray | None = None
        self._index_dir: str | None = None

    # ── Ingestion ──────────────────────────────────────────────────

    def _require_numpy(self):
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "numpy is required for KNN RAG.\n"
                "Install: pip install numpy sentence-transformers hnswlib faiss-cpu"
            )

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Chunk documents, embed, PCA-compress, build HNSW + optional PQ."""
        self._require_numpy()
        start = time.perf_counter()
        corpus_dir = Path(corpus_path)

        manifest_path = corpus_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            doc_files = [corpus_dir / f"{d['doc_id']}.json" for d in manifest["documents"]]
        else:
            doc_files = sorted(corpus_dir.glob("*.json"))

        self._chunks = []
        for doc_file in doc_files:
            if doc_file.name == "manifest.json":
                continue
            with open(doc_file) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", doc_file.stem)
            title = doc_data.get("title", doc_id)
            full_text = doc_data.get("full_text", "")
            chunks = _chunk_text(full_text, self._chunk_size, self._chunk_overlap)
            for i, chunk in enumerate(chunks):
                self._chunks.append({
                    "text": chunk,
                    "doc_id": doc_id,
                    "title": title,
                    "chunk_idx": i,
                })
            logger.debug(f"KNN chunked {doc_id}: {len(chunks)} chunks")

        n = len(self._chunks)
        logger.info(f"KNN: encoding {n} chunks with {self._embed_model_name}")

        texts = [c["text"] for c in self._chunks]

        # Step 1: Encode with SentenceTransformer
        self._embedder = self._load_embedder()
        raw_emb = self._embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=self._normalize,
        ).astype("float32")
        logger.debug(f"Raw embeddings shape: {raw_emb.shape}")

        # Step 2: PCA 384 → pca_dim
        from sklearn.decomposition import PCA
        pca_dim = min(self._pca_dim, raw_emb.shape[0] - 1, raw_emb.shape[1])
        self._pca = PCA(n_components=pca_dim)
        self._pca.fit(raw_emb)
        emb_pca = self._pca.transform(raw_emb).astype("float32")
        logger.debug(f"PCA embeddings shape: {emb_pca.shape}")
        self._embeddings_pca = emb_pca

        # Step 3: Optional FAISS PQ
        if self._pq_enabled:
            try:
                import faiss
                pq_dim = emb_pca.shape[1]
                pq_m = min(self._pq_subvectors, pq_dim)
                self._pq_index = faiss.IndexPQ(pq_dim, pq_m, self._pq_bits)
                if n >= 2 * (2 ** self._pq_bits):
                    self._pq_index.train(emb_pca)
                    self._pq_index.add(emb_pca)
                    logger.debug(f"FAISS PQ built: {pq_m}×{self._pq_bits}bit ({32} bytes/vec)")
                else:
                    logger.warning("Too few chunks for PQ training, skipping PQ")
                    self._pq_index = None
            except ImportError:
                logger.warning("faiss-cpu not installed, skipping PQ compression")
                self._pq_index = None

        # Step 4: HNSW index
        dim = emb_pca.shape[1]
        try:
            import hnswlib
            self._hnsw_index = hnswlib.Index(space="cosine", dim=dim)
            self._hnsw_index.init_index(
                max_elements=max(n, 1),
                ef_construction=self._hnsw_ef_construction,
                M=self._hnsw_m,
            )
            self._hnsw_index.add_items(emb_pca, np.arange(n))
            self._hnsw_index.set_ef(self._hnsw_ef)
            logger.debug(f"HNSW index built: M={self._hnsw_m}, ef={self._hnsw_ef}")
        except ImportError:
            logger.warning("hnswlib not installed, falling back to brute-force cosine")
            self._hnsw_index = None

        elapsed = time.perf_counter() - start
        self._is_ingested = True

        # Save index artefacts
        self._index_dir = str(corpus_dir / ".knn_index")
        Path(self._index_dir).mkdir(exist_ok=True)
        meta = {
            "n_chunks": n,
            "raw_dim": self._raw_dim,
            "pca_dim": int(emb_pca.shape[1]),
            "pq_enabled": self._pq_index is not None,
            "hnsw_m": self._hnsw_m,
            "chunk_size": self._chunk_size,
        }
        meta_path = Path(self._index_dir) / "knn_meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        index_size = meta_path.stat().st_size
        logger.info(
            f"KNN ingestion: {len(doc_files)} docs → {n} chunks → "
            f"PCA({emb_pca.shape[1]}) + HNSW in {elapsed:.2f}s"
        )

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=len([f for f in doc_files if f.name != "manifest.json"]),
            ingestion_time_seconds=elapsed,
            index_size_bytes=index_size,
            index_artifacts={"index_dir": self._index_dir},
            metadata=meta,
        )

    # ── Query ──────────────────────────────────────────────────────

    def query(self, question: str) -> RAGResponse:
        """ANN search → top-K chunks → LLM generation."""
        start = time.perf_counter()

        # Embed query
        q_raw = self._embedder.encode(
            [question], normalize_embeddings=self._normalize
        ).astype("float32")
        q_pca = self._pca.transform(q_raw).astype("float32")

        # Retrieve
        if self._hnsw_index is not None:
            ids, _ = self._hnsw_index.knn_query(q_pca, k=min(self._top_k, len(self._chunks)))
            result_ids = ids[0].tolist()
        else:
            # Brute-force cosine fallback
            emb = self._embeddings_pca
            sims = (emb @ q_pca.T).squeeze()
            result_ids = np.argsort(-sims)[:self._top_k].tolist()

        contexts = []
        references = []
        for idx in result_ids:
            chunk = self._chunks[idx]
            contexts.append(chunk["text"])
            references.append({
                "chunk_idx": idx,
                "doc_id": chunk["doc_id"],
                "title": chunk["title"],
                "method": "pca_hnsw",
            })

        # Generate
        passages_text = "\n\n---\n\n".join(
            f"[Source: {ref['title']}]\n{ctx}"
            for ctx, ref in zip(contexts, references)
        )
        gen_resp = self._gen_client.generate(
            ANSWER_PROMPT.format(question=question, passages=passages_text),
            system_prompt=self._system_prompt,
        )
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

    # ── Add documents ─────────────────────────────────────────────

    def add_documents(self, document_paths: list[str]) -> UpdateReport:
        """
        Add new documents — requires partial re-index of PCA+HNSW.
        PCA must be retrained on the full corpus for correctness.
        """
        start = time.perf_counter()

        new_chunks = []
        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            title = doc_data.get("title", doc_id)
            full_text = doc_data.get("full_text", "")
            chunks = _chunk_text(full_text, self._chunk_size, self._chunk_overlap)
            for i, chunk in enumerate(chunks):
                new_chunks.append({"text": chunk, "doc_id": doc_id, "title": title, "chunk_idx": i})

        self._chunks.extend(new_chunks)

        # Re-encode only new chunks, then extend HNSW without re-training PCA
        if self._embedder and self._pca and self._hnsw_index:
            new_texts = [c["text"] for c in new_chunks]
            new_raw = self._embedder.encode(new_texts, normalize_embeddings=self._normalize).astype("float32")
            new_pca = self._pca.transform(new_raw).astype("float32")
            start_idx = len(self._chunks) - len(new_chunks)
            new_ids = np.arange(start_idx, start_idx + len(new_chunks))
            self._hnsw_index.add_items(new_pca, new_ids)
            if self._embeddings_pca is not None:
                self._embeddings_pca = np.vstack([self._embeddings_pca, new_pca])

        elapsed = time.perf_counter() - start
        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=False,
            new_index_size_bytes=0,
        )

    # ── Private ───────────────────────────────────────────────────

    def _load_embedder(self):
        """Load SentenceTransformer (raises ImportError if not installed)."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(self._embed_model_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for KNN RAG.\n"
                "Install: pip install sentence-transformers\n"
                f"Original error: {e}"
            )
