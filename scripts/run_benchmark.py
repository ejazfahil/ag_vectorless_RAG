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
from pathlib import Path

import yaml
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger_setup import setup_logger
from src.utils.telemetry import TelemetryTracker


def load_config(config_path: str = "configs/base.yaml") -> dict:
    """Load base configuration."""
    with open(PROJECT_ROOT / config_path, "r") as f:
        return yaml.safe_load(f)


def get_pipelines(pipeline_filter: str | None = None):
    """Instantiate all (or filtered) RAG pipelines."""
    from src.pipelines.agentic_rag import AgenticRAG
    from src.pipelines.bm25_rag import BM25RAG
    from src.pipelines.embedding_free_rag import EmbeddingFreeRAG
    from src.pipelines.hybrid_sota import HybridSoTARAG
    from src.pipelines.pageindex_rag import PageIndexRAG
    from src.pipelines.roaming_rag import RoamingRAG
    from src.pipelines.three_stage_hybrid import ThreeStageHybridRAG

    PIPELINE_CLASSES = {
        "pageindex": PageIndexRAG,
        "roaming": RoamingRAG,
        "bm25": BM25RAG,
        "agentic": AgenticRAG,
        "hybrid_sota": HybridSoTARAG,
        "embedding_free": EmbeddingFreeRAG,
        "three_stage_hybrid": ThreeStageHybridRAG,
    }

    pipeline_configs = {
        "pageindex": "configs/pageindex.yaml",
        "roaming": "configs/roaming.yaml",
        "bm25": "configs/bm25.yaml",
        "agentic": "configs/agentic.yaml",
        "hybrid_sota": "configs/hybrid_sota.yaml",
        "embedding_free": "configs/embedding_free.yaml",
        "three_stage_hybrid": "configs/three_stage_hybrid.yaml",
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
):
    """
    Run a single pipeline against a domain's Q&A set.
    Implements the evaluation skeleton from blueprint C.4.
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
    import uuid
    run_id = str(uuid.uuid4())
    tracker = TelemetryTracker(pipeline_name, domain, run_id=run_id)

    # Log skeleton run to SQLite database to satisfy FOREIGN KEY constraint for queries
    try:
        from datetime import datetime

        from src.utils.database import db_manager
        skeleton_run = {
            "id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_name": pipeline_name,
            "domain": domain,
        }
        db_manager.insert_run(skeleton_run)
    except Exception as e:
        logger.error(f"Failed to log skeleton run to SQLite database: {e}")

    # Run queries with per-query telemetry
    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        reference = qa.get("ground_truth_answer", "")
        q_type = qa.get("question_type", "")

        logger.info(f"  [{i+1}/{len(qa_pairs)}] {question[:70]}...")

        with tracker.track_query(question, reference, q_type) as ctx:
            try:
                result = pipeline.query(question)
                ctx.set_result(result)
            except Exception as e:
                logger.error(f"  ✗ Query failed: {e}")
                ctx.set_error("format_failure")

    # Save telemetry
    results_dir = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tracker.save(str(results_dir / f"{pipeline_name}_{domain}_telemetry.jsonl"))

    # Print summary
    summary = tracker.get_summary()

    # Log run metadata to SQLite database
    try:
        from datetime import datetime

        from src.utils.database import db_manager

        latency = summary.get("latency", {})
        memory = summary.get("memory", {})
        tokens = summary.get("tokens", {})
        cost = summary.get("cost", {})

        run_data = {
            "id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "pipeline_name": pipeline_name,
            "domain": domain,
            "num_questions": summary.get("n", 0),
            "success_rate": summary.get("success_rate", 0.0),
            "mean_latency": latency.get("mean_s", 0.0),
            "p50_latency": latency.get("p50_s", 0.0),
            "p95_latency": latency.get("p95_s", 0.0),
            "peak_rss": memory.get("peak_rss_mb", 0.0),
            "total_tokens": tokens.get("total", 0),
            "total_cost": cost.get("total_usd", 0.0),
        }
        db_manager.insert_run(run_data)
    except Exception as e:
        logger.error(f"Failed to log run metadata to SQLite database: {e}")
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

    return summary


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

    args = parser.parse_args()

    # Setup
    setup_logger(level="DEBUG" if args.verbose else "INFO")

    logger.info("=" * 60)
    logger.info("  Vectorless RAG Benchmark Suite")
    logger.info("  Fully Local — Zero API Cost")
    logger.info("=" * 60)
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Pipeline: {args.pipeline or 'all (7 pipelines)'}")
    logger.info(f"  Max questions: {args.max_questions or 'all'}")
    logger.info(f"  Output: {args.output_dir}")
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
