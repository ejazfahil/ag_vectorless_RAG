"""
Evaluation Runner — orchestrates all 4 evaluation dimensions
across all pipelines and domains.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from src.corpus.qa_generator import GoldenQAGenerator
from src.evaluation.cost_tracker import CostTracker
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.maintenance import MaintenanceEvaluator
from src.evaluation.ragas_eval import RAGASEvaluator
from src.pipelines.base import RAGPipeline


class EvaluationRunner:
    """
    Orchestrates the full benchmark: runs each pipeline against golden QA,
    then evaluates across all 4 dimensions.
    """

    def __init__(self, config: dict):
        self.config = config
        self.ragas = RAGASEvaluator(config.get("evaluation", {}))
        self.judge = LLMJudge(
            model=config.get("evaluation", {}).get("llm_judge", {}).get("model", "gpt-4o")
        )
        self.maintenance = MaintenanceEvaluator(config.get("evaluation", {}))
        self.output_dir = config.get("results", {}).get("output_dir", "results/")

    def run(
        self,
        pipelines: list[RAGPipeline],
        golden_qa_path: str,
        corpus_path: str,
        domain: str,
        new_doc_paths: list[str] | None = None,
    ) -> dict:
        """
        Run full evaluation for all pipelines on one domain.

        Returns dict mapping pipeline_name -> {ragas, judge, maintenance, cost}.
        """
        # Load golden QA
        qa_gen = GoldenQAGenerator()
        qa_set = qa_gen.load(golden_qa_path)
        logger.info(f"Loaded {qa_set.total_pairs} golden Q&A pairs for {domain}")

        all_results = {}

        for pipeline in pipelines:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {pipeline.display_name}")
            logger.info(f"{'='*60}")

            cost_tracker = CostTracker(
                model=pipeline.config.get("pipeline", {}).get("generation", {}).get("model", "gpt-4o")
            )

            # 1. Maintenance eval (includes ingestion)
            maint_report = self.maintenance.evaluate(
                pipeline, corpus_path, new_doc_paths, domain
            )

            # 2. Query all golden QA pairs
            responses = []
            for qa in qa_set.pairs:
                try:
                    rag_resp = pipeline.query(qa.question)
                    cost_tracker.track(rag_resp, qa.question)
                    responses.append({
                        "question": qa.question,
                        "answer": rag_resp.answer,
                        "contexts": rag_resp.retrieved_contexts,
                        "ground_truth": qa.ground_truth,
                    })
                except Exception as e:
                    logger.error(f"Query failed: {qa.question[:50]}... -> {e}")
                    responses.append({
                        "question": qa.question,
                        "answer": f"ERROR: {e}",
                        "contexts": [],
                        "ground_truth": qa.ground_truth,
                    })

            # 3. RAGAS evaluation
            ragas_result = self.ragas.evaluate(pipeline.name, domain, responses)

            # 4. LLM-as-a-Judge
            judge_report = self.judge.evaluate(pipeline.name, domain, responses)

            # 5. Cost report
            cost_report = cost_tracker.get_report(pipeline.name, domain)

            all_results[pipeline.name] = {
                "ragas": ragas_result,
                "judge": judge_report,
                "maintenance": maint_report,
                "cost": cost_report,
            }

            # Save per-pipeline results
            self._save_results(pipeline.name, domain, all_results[pipeline.name])

        # Save comparison summary
        self._save_comparison(domain, all_results)

        return all_results

    def _save_results(self, pipeline_name, domain, results):
        out_dir = Path(self.output_dir) / "raw" / pipeline_name
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ragas.save(results["ragas"], str(out_dir))
        self.judge.save(results["judge"], str(out_dir))

    def _save_comparison(self, domain, all_results):
        out_dir = Path(self.output_dir) / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {}
        for name, res in all_results.items():
            summary[name] = {
                "ragas": res["ragas"].metrics,
                "judge": {
                    "accuracy": res["judge"].accuracy,
                    "hallucination_rate": res["judge"].hallucination_rate,
                    "error_distribution": res["judge"].error_distribution,
                },
                "maintenance": {
                    "ingestion_time_s": res["maintenance"].ingestion_time_seconds,
                    "update_time_s": res["maintenance"].update_time_seconds,
                    "index_size_mb": res["maintenance"].index_size_mb,
                    "requires_reindex": res["maintenance"].requires_full_reindex,
                },
                "cost": {
                    "avg_cost_per_query": res["cost"].avg_cost_per_query,
                    "total_cost": res["cost"].total_cost_usd,
                    "avg_latency_ms": res["cost"].avg_latency_ms,
                },
            }

        path = out_dir / f"comparison_{domain}.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved comparison to {path}")
