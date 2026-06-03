"""
RAGAS Evaluator — wraps the RAGAS framework to evaluate RAG pipelines
on faithfulness, answer relevancy, context precision, and context recall.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class RAGASResult:
    """Results from RAGAS evaluation for a single pipeline."""

    pipeline_name: str
    domain: str
    num_samples: int
    metrics: dict[str, float]       # metric_name -> score (0-1)
    per_sample: list[dict] = field(default_factory=list)

    @property
    def faithfulness(self) -> float:
        return self.metrics.get("faithfulness", 0.0)

    @property
    def answer_relevancy(self) -> float:
        return self.metrics.get("answer_relevancy", 0.0)

    @property
    def context_precision(self) -> float:
        return self.metrics.get("context_precision", 0.0)

    @property
    def context_recall(self) -> float:
        return self.metrics.get("context_recall", 0.0)


class RAGASEvaluator:
    """
    Evaluate RAG pipelines using the RAGAS framework.

    Computes:
    - Faithfulness: Is the answer grounded in retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved docs actually relevant?
    - Context Recall: Did we retrieve all needed information?

    Usage:
        evaluator = RAGASEvaluator()
        result = evaluator.evaluate(pipeline, golden_qa)
        print(result.metrics)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def evaluate(
        self,
        pipeline_name: str,
        domain: str,
        responses: list[dict],
    ) -> RAGASResult:
        """
        Run RAGAS evaluation on pipeline responses.

        Args:
            pipeline_name: Name of the pipeline being evaluated.
            domain: Domain of the Q&A set.
            responses: List of dicts with keys:
                - question, answer, contexts, ground_truth

        Returns:
            RAGASResult with aggregate and per-sample metrics.
        """
        logger.info(
            f"Running RAGAS evaluation for {pipeline_name} "
            f"on {domain} ({len(responses)} samples)"
        )

        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            # Prepare dataset in RAGAS format
            eval_data = {
                "question": [r["question"] for r in responses],
                "answer": [r["answer"] for r in responses],
                "contexts": [r["contexts"] for r in responses],
                "ground_truth": [r["ground_truth"] for r in responses],
            }

            dataset = Dataset.from_dict(eval_data)

            # Run evaluation
            result = ragas_evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            metrics = {
                "faithfulness": float(result.get("faithfulness", 0.0)),
                "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
                "context_precision": float(result.get("context_precision", 0.0)),
                "context_recall": float(result.get("context_recall", 0.0)),
            }

            # Per-sample results
            per_sample = []
            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                per_sample = df.to_dict(orient="records")

        except ImportError:
            logger.warning("RAGAS not installed, using fallback evaluation")
            metrics = self._fallback_evaluation(responses)
            per_sample = []

        ragas_result = RAGASResult(
            pipeline_name=pipeline_name,
            domain=domain,
            num_samples=len(responses),
            metrics=metrics,
            per_sample=per_sample,
        )

        logger.info(f"RAGAS results for {pipeline_name}:")
        for k, v in metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        return ragas_result

    def _fallback_evaluation(self, responses: list[dict]) -> dict[str, float]:
        """
        Simple fallback metrics when RAGAS is not available.
        Uses basic string matching as a crude approximation.
        """
        from difflib import SequenceMatcher

        faithfulness_scores = []
        relevancy_scores = []

        for r in responses:
            answer = r.get("answer", "").lower()
            ground_truth = r.get("ground_truth", "").lower()
            contexts = " ".join(r.get("contexts", [])).lower()

            # Crude faithfulness: how much of the answer appears in contexts
            if contexts:
                matcher = SequenceMatcher(None, answer, contexts)
                faithfulness_scores.append(matcher.ratio())
            else:
                faithfulness_scores.append(0.0)

            # Crude relevancy: similarity between answer and ground truth
            matcher = SequenceMatcher(None, answer, ground_truth)
            relevancy_scores.append(matcher.ratio())

        def avg(lst):
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "faithfulness": avg(faithfulness_scores),
            "answer_relevancy": avg(relevancy_scores),
            "context_precision": 0.0,  # Can't compute without RAGAS
            "context_recall": 0.0,
        }

    def save(self, result: RAGASResult, output_dir: str) -> None:
        """Save RAGAS results to JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / f"ragas_{result.pipeline_name}_{result.domain}.json"
        with open(path, "w") as f:
            json.dump({
                "pipeline_name": result.pipeline_name,
                "domain": result.domain,
                "num_samples": result.num_samples,
                "metrics": result.metrics,
                "per_sample": result.per_sample,
            }, f, indent=2)

        logger.info(f"Saved RAGAS results to {path}")
