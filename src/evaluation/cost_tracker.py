"""
Cost Tracker — tracks token usage and USD cost per query across pipelines.
"""

from __future__ import annotations


import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from loguru import logger
from src.pipelines.base import RAGResponse


MODEL_PRICING_PER_1M = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}


@dataclass
class CostReport:
    pipeline_name: str
    domain: str
    num_queries: int = 0
    total_cost_usd: float = 0.0
    avg_cost_per_query: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_latency_ms: float = 0.0
    ingestion_cost_usd: float = 0.0
    per_query_costs: list[dict] = field(default_factory=list)
    projected_daily_cost_10k: float = 0.0

    def compute_aggregates(self):
        if self.num_queries > 0:
            self.avg_cost_per_query = self.total_cost_usd / self.num_queries
            self.projected_daily_cost_10k = self.avg_cost_per_query * 10_000
            latencies = [q.get("latency_ms", 0) for q in self.per_query_costs]
            self.avg_latency_ms = sum(latencies) / len(latencies) if latencies else 0


class CostTracker:
    def __init__(self, model="gpt-4o", log_path=None):
        self.model = model
        self.pricing = MODEL_PRICING_PER_1M.get(model, {"input": 0, "output": 0})
        self.log_path = log_path
        self._queries = []

    def track(self, response: RAGResponse, question: str = ""):
        tokens = response.tokens_used
        cost = (
            tokens.get("input", 0) * self.pricing["input"] / 1_000_000
            + tokens.get("output", 0) * self.pricing["output"] / 1_000_000
        )
        entry = {
            "question": question[:100],
            "input_tokens": tokens.get("input", 0),
            "output_tokens": tokens.get("output", 0),
            "cost_usd": cost,
            "latency_ms": response.latency_ms,
        }
        self._queries.append(entry)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        return cost

    def get_report(self, pipeline_name: str, domain: str) -> CostReport:
        report = CostReport(
            pipeline_name=pipeline_name, domain=domain,
            num_queries=len(self._queries),
            total_cost_usd=sum(q["cost_usd"] for q in self._queries),
            total_input_tokens=sum(q["input_tokens"] for q in self._queries),
            total_output_tokens=sum(q["output_tokens"] for q in self._queries),
            per_query_costs=self._queries,
        )
        report.compute_aggregates()
        return report

    def reset(self):
        self._queries = []
