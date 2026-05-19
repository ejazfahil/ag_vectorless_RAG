#!/usr/bin/env python3
from __future__ import annotations
"""
Main benchmark runner — entry point for the Vectorless RAG Benchmark.

Usage:
    python scripts/run_benchmark.py --domain finance
    python scripts/run_benchmark.py --domain all
    python scripts/run_benchmark.py --pipeline three_stage_hybrid --domain finance
    python scripts/run_benchmark.py --pipeline bm25 --domain finance --max-questions 10
"""

import argparse
import json
import sys
import yaml
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logger
from src.utils.telemetry import TelemetryTracker
from src.evaluation.string_metrics import compute_metrics_batch  # new.md C.3


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load base configuration."""
    with open(PROJECT_ROOT / config_path, "r") as f:
        return yaml.safe_load(f)


def get_pipelines(pipeline_filter: str | None = None):
    """Instantiate all (or filtered) RAG pipelines."""
    from src.pipelines.pageindex_rag import PageIndexRAG
    from src.pipelines.roaming_rag import RoamingRAG
    from src.pipelines.bm25_rag import BM25RAG
    from src.pipelines.agentic_rag import AgenticRAG
    from src.pipelines.hybrid_sota import HybridSoTARAG
    from src.pipelines.embedding_free_rag import EmbeddingFreeRAG
    from src.pipelines.three_stage_hybrid import ThreeStageHybridRAG
    from src.pipelines.knn_rag import KNNInMemoryRAG  # new.md D.7

    PIPELINE_CLASSES = {
        "pageindex": PageIndexRAG,
        "roaming": RoamingRAG,
        "bm25": BM25RAG,
        "agentic": AgenticRAG,
        "hybrid_sota": HybridSoTARAG,
        "embedding_free": EmbeddingFreeRAG,
        "three_stage_hybrid": ThreeStageHybridRAG,
        "knn": KNNInMemoryRAG,   # new.md D.7
    }

    pipeline_configs = {
        "pageindex": "configs/pageindex.yaml",
        "roaming": "configs/roaming.yaml",
        "bm25": "configs/bm25.yaml",
        "agentic": "configs/agentic.yaml",
        "hybrid_sota": "configs/hybrid_sota.yaml",
        "embedding_free": "configs/embedding_free.yaml",
        "three_stage_hybrid": "configs/three_stage_hybrid.yaml",
        "knn": "configs/knn.yaml",  # new.md D.7
    }

    pipelines = []
    for name, config_path in pipeline_configs.items():
        if pipeline_filter and name != pipeline_filter:
            continue

        config_file = PROJECT_ROOT / config_path
        if config_file.exists():
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
            pipeline_class = PIPELINE_CLASSES[name]
            pipelines.append(pipeline_class(config))
            logger.info(f"Loaded pipeline: {name}")
        else:
            logger.warning(f"Config not found: {config_path}")

    return pipelines


def load_golden_qa(qa_path: str, max_questions: int = 0) -> list[dict]:
    """Load golden Q&A from JSONL."""
    full_path = PROJECT_ROOT / qa_path
    if not full_path.exists():
        logger.warning(f"Golden Q&A not found: {full_path}")
        return []

    qa_pairs = []
    with open(full_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                qa_pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if max_questions > 0:
        qa_pairs = qa_pairs[:max_questions]

    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs from {full_path.name}")
    return qa_pairs


def run_benchmark(
    pipeline, domain: str, qa_pairs: list[dict],
    corpus_path: str, output_dir: str,
    mlflow_enabled: bool = False,
):
    """
    Run a single pipeline against a domain's Q&A set.
    Implements the evaluation skeleton from blueprint C.4.
    Adds F1/EM metrics (C.3) and MLflow tracking (F.3).
    """
    pipeline_name = pipeline.name
    logger.info(f"\n{'='*50}")
    logger.info(f"  Pipeline: {pipeline_name} | Domain: {domain}")
    logger.info(f"  Questions: {len(qa_pairs)}")
    logger.info(f"{'='*50}")

    # Ingest corpus
    try:
        report = pipeline.ingest(str(PROJECT_ROOT / corpus_path))
        logger.info(
            f"  Ingested: {report.num_documents} docs in "
            f"{report.ingestion_time_seconds:.2f}s"
        )
    except Exception as e:
        logger.error(f"  Ingestion failed: {e}")
        return None

    # Initialize telemetry (blueprint C.4)
    tracker = TelemetryTracker(pipeline_name, domain)

    # Track predictions + references for F1/EM (blueprint C.3)
    predictions: list[str] = []
    ground_truths: list[str] = []
    # Error cross-tab: paradigm × query-type (blueprint C.6)
    error_crosstab: dict[str, dict[str, int]] = {}

    # Run queries with per-query telemetry
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        reference = qa.get("ground_truth_answer", qa.get("ground_truth", ""))
        q_type = qa.get("question_type", "unknown")

        logger.info(f"  [{i+1}/{len(qa_pairs)}] {question[:70]}...")

        with tracker.track_query(question, reference, q_type) as ctx:
            try:
                result = pipeline.query(question)
                ctx.set_result(result)
                predictions.append(result.answer)
                ground_truths.append(reference)
                # Cross-tab: no error
                error_crosstab.setdefault(q_type, {}).setdefault("correct", 0)
                error_crosstab[q_type]["correct"] += 1
            except Exception as e:
                logger.error(f"  ✗ Query failed: {e}")
                ctx.set_error("format_failure")
                predictions.append("")
                ground_truths.append(reference)
                error_crosstab.setdefault(q_type, {}).setdefault("format_failure", 0)
                error_crosstab[q_type]["format_failure"] += 1

    # Save telemetry
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tracker.save(str(results_dir / f"{pipeline_name}_{domain}_telemetry.jsonl"))

    # Compute F1/EM metrics (blueprint C.3)
    string_metrics: dict = {}
    if predictions and ground_truths:
        string_metrics = compute_metrics_batch(predictions, ground_truths)
        logger.info(
            f"  String metrics — EM: {string_metrics['exact_match']:.4f} | "
            f"F1: {string_metrics['f1']:.4f} | "
            f"P: {string_metrics['precision']:.4f} | "
            f"R: {string_metrics['recall']:.4f}"
        )

    # Save F1/EM results
    metrics_path = results_dir / f"{pipeline_name}_{domain}_string_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "pipeline": pipeline_name,
            "domain": domain,
            "n": len(predictions),
            "metrics": string_metrics,
        }, f, indent=2)

    # Save error cross-tab (blueprint C.6)
    crosstab_path = results_dir / f"{pipeline_name}_{domain}_error_crosstab.json"
    with open(crosstab_path, "w") as f:
        json.dump({
            "pipeline": pipeline_name,
            "domain": domain,
            "error_crosstab": error_crosstab,
        }, f, indent=2)

    # Print summary
    summary = tracker.get_summary()
    summary["string_metrics"] = string_metrics
    summary["error_crosstab"] = error_crosstab

    logger.info(f"\n  ── Results: {pipeline_name} on {domain} ──")
    logger.info(f"  Questions: {summary['n']}")
    logger.info(f"  Success rate: {summary.get('success_rate', 0):.1%}")
    if summary.get("latency"):
        lat = summary["latency"]
        logger.info(f"  Latency: p50={lat['p50_s']:.2f}s | p95={lat['p95_s']:.2f}s | mean={lat['mean_s']:.2f}s")
    if summary.get("memory"):
        mem = summary["memory"]
        logger.info(f"  Memory: peak RSS={mem['peak_rss_mb']:.0f}MB | mean Δ={mem['mean_delta_mb']:.1f}MB")
    if summary.get("tokens"):
        tok = summary["tokens"]
        logger.info(f"  Tokens: total={tok['total']:,} | mean/query={tok['mean_per_query']:.0f}")
    if summary.get("cost"):
        logger.info(f"  Cost: ${summary['cost']['total_usd']:.6f} total")

    # MLflow tracking (blueprint F.3)
    if mlflow_enabled:
        _log_to_mlflow(pipeline_name, domain, summary, string_metrics, report)

    return summary


def _log_to_mlflow(pipeline_name: str, domain: str, summary: dict,
                   string_metrics: dict, ingest_report) -> None:
    """Log benchmark results to local MLflow (blueprint F.3)."""
    try:
        import mlflow
        mlflow.set_tracking_uri(f"file:{PROJECT_ROOT}/mlruns")
        run_name = f"{pipeline_name}_{domain}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "pipeline": pipeline_name,
                "domain": domain,
                "n_questions": summary.get("n", 0),
                "n_docs": getattr(ingest_report, "num_documents", 0),
            })
            metrics_to_log = {
                "success_rate": summary.get("success_rate", 0),
                "f1": string_metrics.get("f1", 0),
                "exact_match": string_metrics.get("exact_match", 0),
                "precision": string_metrics.get("precision", 0),
                "recall": string_metrics.get("recall", 0),
                "ingestion_time_s": getattr(ingest_report, "ingestion_time_seconds", 0),
            }
            if summary.get("latency"):
                metrics_to_log["p50_latency_s"] = summary["latency"]["p50_s"]
                metrics_to_log["p95_latency_s"] = summary["latency"]["p95_s"]
            if summary.get("memory"):
                metrics_to_log["peak_rss_mb"] = summary["memory"]["peak_rss_mb"]
            if summary.get("cost"):
                metrics_to_log["total_cost_usd"] = summary["cost"]["total_usd"]
            mlflow.log_metrics(metrics_to_log)
        logger.info(f"  MLflow: logged run '{run_name}'")
    except ImportError:
        logger.debug("MLflow not installed, skipping experiment tracking")
    except Exception as e:
        logger.warning(f"MLflow logging failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Vectorless RAG Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py --domain finance --pipeline bm25
  python scripts/run_benchmark.py --domain all --pipeline three_stage_hybrid
  python scripts/run_benchmark.py --domain finance --max-questions 10 -v
        """,
    )
    parser.add_argument(
        "--domain", type=str, default="all",
        choices=["finance", "legal", "technical", "all"],
        help="Domain to evaluate (default: all)",
    )
    parser.add_argument(
        "--pipeline", type=str, default=None,
        choices=[
            "pageindex", "roaming", "bm25", "agentic",
            "hybrid_sota", "embedding_free", "three_stage_hybrid",
            "knn",  # new.md D.7
        ],
        help="Run only a specific pipeline (default: all)",
    )
    parser.add_argument(
        "--max-questions", type=int, default=0,
        help="Max questions per domain (0 = all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/",
        help="Directory for output results",
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Enable MLflow experiment tracking (blueprint F.3)",
    )

    args = parser.parse_args()

    # Setup
    setup_logger(level="DEBUG" if args.verbose else "INFO")

    logger.info("=" * 60)
    logger.info("  Vectorless RAG Benchmark Suite")
    logger.info("  Fully Local — Zero API Cost")
    logger.info("=" * 60)
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Pipeline: {args.pipeline or 'all (8 pipelines)'}")
    logger.info(f"  Max questions: {args.max_questions or 'all'}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info(f"  MLflow: {'enabled' if args.mlflow else 'disabled'}")
    logger.info("=" * 60)

    # Domain paths
    DOMAIN_CONFIGS = {
        "finance": {
            "corpus": "data/processed/finance",
            "golden_qa": "data/golden_qa/finance_golden_qa.jsonl",
        },
        "legal": {
            "corpus": "data/processed/legal",
            "golden_qa": "data/golden_qa/legal_golden_qa.jsonl",
        },
        "technical": {
            "corpus": "data/processed/technical",
            "golden_qa": "data/golden_qa/technical_golden_qa.jsonl",
        },
    }

    # Get pipelines
    pipelines = get_pipelines(args.pipeline)
    if not pipelines:
        logger.error("No pipelines loaded. Check config files exist.")
        return

    domains = ["finance", "legal", "technical"] if args.domain == "all" else [args.domain]
    all_summaries = []

    for domain in domains:
        domain_cfg = DOMAIN_CONFIGS[domain]

        # Load Q&A
        qa_pairs = load_golden_qa(domain_cfg["golden_qa"], args.max_questions)
        if not qa_pairs:
            logger.warning(f"No Q&A pairs for {domain}, skipping")
            continue

        for pipeline in pipelines:
            summary = run_benchmark(
                pipeline, domain, qa_pairs,
                domain_cfg["corpus"], args.output_dir,
                mlflow_enabled=args.mlflow,
            )
            if summary:
                all_summaries.append(summary)

    # Save combined summary
    if all_summaries:
        summary_path = Path(args.output_dir) / "benchmark_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        logger.info(f"\n📊 Combined summary: {summary_path}")

    logger.info("\n✅ Benchmark complete! Results in: " + args.output_dir)


if __name__ == "__main__":
    main()
