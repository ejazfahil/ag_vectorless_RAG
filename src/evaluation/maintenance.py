"""
Maintenance Evaluator — measures operational complexity of RAG pipelines.
"""

from __future__ import annotations


import time
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
from src.pipelines.base import RAGPipeline


@dataclass
class MaintenanceReport:
    pipeline_name: str
    domain: str
    ingestion_time_seconds: float = 0.0
    update_time_seconds: float = 0.0
    index_size_bytes: int = 0
    index_size_mb: float = 0.0
    requires_full_reindex: bool = False
    num_documents_ingested: int = 0
    num_documents_updated: int = 0
    ingestion_seconds_per_doc: float = 0.0
    update_seconds_per_doc: float = 0.0
    metadata: dict = field(default_factory=dict)

    def compute_derived(self):
        self.index_size_mb = self.index_size_bytes / (1024 * 1024)
        if self.num_documents_ingested > 0:
            self.ingestion_seconds_per_doc = self.ingestion_time_seconds / self.num_documents_ingested
        if self.num_documents_updated > 0:
            self.update_seconds_per_doc = self.update_time_seconds / self.num_documents_updated


class MaintenanceEvaluator:
    def __init__(self, config=None):
        self.config = config or {}

    def evaluate(self, pipeline, corpus_path, new_doc_paths=None, domain=""):
        logger.info(f"Maintenance eval for {pipeline.name}")
        start = time.perf_counter()
        ingest_report = pipeline.ingest(corpus_path)
        ingestion_time = time.perf_counter() - start
        index_size = ingest_report.index_size_bytes
        update_time, requires_reindex, num_updated = 0.0, False, 0
        if new_doc_paths:
            start = time.perf_counter()
            update_report = pipeline.add_documents(new_doc_paths)
            update_time = time.perf_counter() - start
            requires_reindex = update_report.required_full_reindex
            num_updated = update_report.num_new_documents
        report = MaintenanceReport(
            pipeline_name=pipeline.name, domain=domain,
            ingestion_time_seconds=ingestion_time, update_time_seconds=update_time,
            index_size_bytes=index_size, requires_full_reindex=requires_reindex,
            num_documents_ingested=ingest_report.num_documents,
            num_documents_updated=num_updated,
        )
        report.compute_derived()
        return report
