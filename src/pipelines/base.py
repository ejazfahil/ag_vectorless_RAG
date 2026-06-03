"""
Abstract base class for all RAG pipelines.

Every pipeline (PageIndex, Roaming, BM25, Agentic, Hybrid SoTA) must
implement this interface so the evaluation harness can treat them uniformly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RAGResponse:
    """Standardized response from any RAG pipeline."""

    answer: str
    retrieved_contexts: list[str]       # Raw text chunks used — needed by RAGAS
    source_references: list[dict]       # [{doc, section, page, ...}]
    latency_ms: float                   # End-to-end query time
    tokens_used: dict = field(          # {"input": N, "output": M, "total": N+M}
        default_factory=lambda: {"input": 0, "output": 0, "total": 0}
    )
    cost_usd: float = 0.0              # Computed by CostTracker
    metadata: dict = field(default_factory=dict)  # Pipeline-specific extras


@dataclass
class IngestionReport:
    """Report returned after corpus ingestion."""

    pipeline_name: str
    num_documents: int
    ingestion_time_seconds: float
    index_size_bytes: int
    index_artifacts: dict = field(default_factory=dict)  # Paths to created indices
    metadata: dict = field(default_factory=dict)


@dataclass
class UpdateReport:
    """Report returned after incremental document addition."""

    pipeline_name: str
    num_new_documents: int
    update_time_seconds: float
    required_full_reindex: bool
    new_index_size_bytes: int
    metadata: dict = field(default_factory=dict)


class RAGPipeline(ABC):
    """
    Abstract base class that all RAG pipeline implementations must extend.

    The evaluation harness calls these methods in sequence:
        1. ingest(corpus_path)    → Index the full corpus
        2. query(question)        → Answer questions (repeated per Q&A pair)
        3. add_documents(paths)   → Test incremental updates
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the pipeline with a configuration dictionary.

        Args:
            config: Parsed YAML config (pipeline-specific section).
        """
        self.config = config
        self.name = config.get("pipeline", {}).get("name", "unknown")
        self.display_name = config.get("pipeline", {}).get("display_name", self.name)
        self._is_ingested = False

    @abstractmethod
    def ingest(self, corpus_path: str) -> IngestionReport:
        """
        Index/prepare the entire corpus for retrieval.

        This method is timed by the MaintenanceEvaluator to measure
        cold-start ingestion cost.

        Args:
            corpus_path: Path to the processed corpus directory.

        Returns:
            IngestionReport with timing and size metrics.
        """
        ...

    @abstractmethod
    def query(self, question: str) -> RAGResponse:
        """
        Answer a question using the ingested corpus.

        Must return a fully populated RAGResponse so the evaluation
        harness can compute all metrics (RAGAS, cost, latency, etc.).

        Args:
            question: The user's natural language question.

        Returns:
            RAGResponse with answer, contexts, timing, and token counts.
        """
        ...

    @abstractmethod
    def add_documents(self, document_paths: list[str]) -> UpdateReport:
        """
        Add new documents to the existing index.

        This tests the maintenance dimension: how easily can the
        knowledge base be expanded without full re-indexing?

        Args:
            document_paths: Paths to new documents to add.

        Returns:
            UpdateReport with timing and reindex flag.
        """
        ...

    def _timed_call(self, func, *args, **kwargs):
        """Utility to time any function call and return (result, elapsed_ms)."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        return result, elapsed_ms

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
