"""
Telemetry — memory & latency tracking per blueprint Section C.3/C.4.

Tracks:
- Peak RSS memory (psutil)
- p50/p95 latency (retrieval + generation split)
- Per-query telemetry records
- Error categorization (C.6)
"""

from __future__ import annotations

import json
import time
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import psutil
from loguru import logger


@dataclass
class QueryTelemetry:
    """Per-query telemetry record (blueprint C.4)."""
    question: str
    pipeline_name: str
    domain: str
    run_id: str | None = None

    # Answer & context
    answer: str = ""
    retrieved_contexts: list[str] = field(default_factory=list)
    reference_answer: str = ""
    question_type: str = ""

    # Latency — split retrieval vs generation (blueprint C.3)
    total_latency_s: float = 0.0
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0

    # Memory (blueprint C.3: psutil.Process().memory_info().rss)
    mem_before_bytes: int = 0
    mem_after_bytes: int = 0
    mem_delta_mb: float = 0.0
    mem_peak_mb: float = 0.0

    # Tokens & cost
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    # Error category (blueprint C.6)
    error_type: str = ""  # retrieval_failure, reasoning_failure, hallucination, format_failure, none
    success: bool = True

    # Evaluated scores
    faithfulness_score: float | None = None
    f1_score: float | None = None
    em_score: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["retrieved_contexts"] = len(self.retrieved_contexts)
        return d


class TelemetryTracker:
    """
    Tracks per-query telemetry with psutil memory monitoring.

    Usage:
        tracker = TelemetryTracker("bm25", "finance")
        with tracker.track_query("What is revenue?", "ref answer") as t:
            result = pipeline.query("What is revenue?")
            t.set_result(result)
        tracker.save("results/telemetry_bm25_finance.jsonl")
    """

    def __init__(self, pipeline_name: str, domain: str, run_id: str | None = None):
        self.pipeline_name = pipeline_name
        self.domain = domain
        self.run_id = run_id
        self.records: list[QueryTelemetry] = []
        self._process = psutil.Process()

    def track_query(self, question: str, reference_answer: str = "",
                    question_type: str = "") -> QueryContext:
        """Create a context manager for tracking a query."""
        return QueryContext(
            tracker=self,
            question=question,
            reference_answer=reference_answer,
            question_type=question_type,
        )

    def add_record(self, record: QueryTelemetry):
        """Add a completed telemetry record."""
        self.records.append(record)

    def get_summary(self) -> dict:
        """Compute aggregate statistics (blueprint C.5)."""
        if not self.records:
            return {"pipeline": self.pipeline_name, "domain": self.domain, "n": 0}

        latencies = [r.total_latency_s for r in self.records]
        mem_deltas = [r.mem_delta_mb for r in self.records]
        tokens = [r.total_tokens for r in self.records]
        costs = [r.cost_usd for r in self.records]

        # Error type breakdown (blueprint C.6)
        error_counts: dict[str, int] = {}
        for r in self.records:
            et = r.error_type or "none"
            error_counts[et] = error_counts.get(et, 0) + 1

        # Query type breakdown
        type_latencies: dict[str, list[float]] = {}
        for r in self.records:
            qt = r.question_type or "unknown"
            type_latencies.setdefault(qt, []).append(r.total_latency_s)

        return {
            "pipeline": self.pipeline_name,
            "domain": self.domain,
            "n": len(self.records),
            "success_rate": sum(1 for r in self.records if r.success) / len(self.records),
            "latency": {
                "mean_s": statistics.mean(latencies),
                "median_s": statistics.median(latencies),
                "p50_s": _percentile(latencies, 50),
                "p95_s": _percentile(latencies, 95),
                "min_s": min(latencies),
                "max_s": max(latencies),
            },
            "memory": {
                "mean_delta_mb": statistics.mean(mem_deltas),
                "max_delta_mb": max(mem_deltas),
                "peak_rss_mb": max(r.mem_peak_mb for r in self.records),
            },
            "tokens": {
                "total": sum(tokens),
                "mean_per_query": statistics.mean(tokens),
            },
            "cost": {
                "total_usd": sum(costs),
                "mean_per_query_usd": statistics.mean(costs),
            },
            "errors": error_counts,
            "latency_by_type": {
                qt: {"mean_s": statistics.mean(lats), "n": len(lats)}
                for qt, lats in type_latencies.items()
            },
        }

    def save(self, output_path: str):
        """Save all records to JSONL."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

        # Also save summary
        summary_path = path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)

        logger.info(
            f"Telemetry saved: {len(self.records)} records → {path.name}"
        )


class QueryContext:
    """Context manager for tracking a single query execution."""

    def __init__(self, tracker: TelemetryTracker, question: str,
                 reference_answer: str, question_type: str):
        self._tracker = tracker
        self._record = QueryTelemetry(
            question=question,
            pipeline_name=tracker.pipeline_name,
            domain=tracker.domain,
            run_id=tracker.run_id,
            reference_answer=reference_answer,
            question_type=question_type,
        )
        self._process = tracker._process

    def __enter__(self) -> "QueryContext":
        self._record.mem_before_bytes = self._process.memory_info().rss
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self._start
        self._record.total_latency_s = elapsed
        mem_after = self._process.memory_info().rss
        self._record.mem_after_bytes = mem_after
        self._record.mem_delta_mb = (mem_after - self._record.mem_before_bytes) / 1e6
        self._record.mem_peak_mb = mem_after / 1e6

        if exc_type:
            self._record.success = False
            self._record.error_type = "format_failure" if "JSON" in str(exc_val) else "reasoning_failure"

        # Compute string metrics
        if self._record.success and self._record.answer and self._record.reference_answer:
            try:
                from src.evaluation.string_metrics import exact_match, token_f1
                self._record.em_score = exact_match(self._record.answer, self._record.reference_answer)
                f1_res = token_f1(self._record.answer, self._record.reference_answer)
                self._record.f1_score = f1_res.get("f1", 0.0)
            except Exception as e:
                logger.error(f"Failed to calculate F1/EM metrics: {e}")

        self._tracker.add_record(self._record)

        # Log query to SQLite
        try:
            from src.utils.database import db_manager
            query_data = {
                "run_id": self._record.run_id,
                "question": self._record.question,
                "pipeline_name": self._record.pipeline_name,
                "domain": self._record.domain,
                "answer": self._record.answer,
                "retrieved_contexts": self._record.retrieved_contexts,
                "reference_answer": self._record.reference_answer,
                "question_type": self._record.question_type,
                "latency": self._record.total_latency_s,
                "mem_delta": self._record.mem_delta_mb,
                "mem_peak": self._record.mem_peak_mb,
                "input_tokens": self._record.input_tokens,
                "output_tokens": self._record.output_tokens,
                "total_tokens": self._record.total_tokens,
                "cost": self._record.cost_usd,
                "error_type": self._record.error_type,
                "success": self._record.success,
                "faithfulness_score": self._record.faithfulness_score,
                "f1_score": self._record.f1_score,
                "em_score": self._record.em_score,
            }
            db_manager.insert_query(query_data)
        except Exception as e:
            logger.error(f"Error logging query to SQLite: {e}")

        return False  # Don't suppress exceptions

    def set_result(self, rag_response):
        """Set result from a RAGResponse object."""
        self._record.answer = rag_response.answer
        self._record.retrieved_contexts = rag_response.retrieved_contexts
        self._record.input_tokens = rag_response.tokens_used.get("input", 0)
        self._record.output_tokens = rag_response.tokens_used.get("output", 0)
        self._record.total_tokens = rag_response.tokens_used.get("total", 0)
        self._record.cost_usd = rag_response.cost_usd

        # Split latency
        self._record.generation_latency_s = rag_response.latency_ms / 1000
        retrieval_approx = self._record.total_latency_s - self._record.generation_latency_s
        self._record.retrieval_latency_s = max(0, retrieval_approx)

    def set_error(self, error_type: str):
        """Mark error type per blueprint C.6 taxonomy."""
        self._record.error_type = error_type
        self._record.success = False


def _percentile(data: list[float], p: int) -> float:
    """Compute percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * p / 100
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac
