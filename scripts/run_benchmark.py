#!/usr/bin/env python3
from __future__ import annotations
"""
Main benchmark runner — entry point for the Vectorless RAG Benchmark.

Usage:
    python scripts/run_benchmark.py --domain finance
    python scripts/run_benchmark.py --domain all
    python scripts/run_benchmark.py --pipeline hybrid_sota --domain legal
"""

import argparse
import sys
import yaml
from pathlib import Path
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logger
from src.evaluation.runner import EvaluationRunner


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

    PIPELINE_CLASSES = {
        "pageindex": PageIndexRAG,
        "roaming": RoamingRAG,
        "bm25": BM25RAG,
        "agentic": AgenticRAG,
        "hybrid_sota": HybridSoTARAG,
    }

    pipeline_configs = {
        "pageindex": "configs/pageindex.yaml",
        "roaming": "configs/roaming.yaml",
        "bm25": "configs/bm25.yaml",
        "agentic": "configs/agentic.yaml",
        "hybrid_sota": "configs/hybrid_sota.yaml",
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


def main():
    parser = argparse.ArgumentParser(
        description="Vectorless RAG Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_benchmark.py --domain finance
  python scripts/run_benchmark.py --domain all --pipeline hybrid_sota
  python scripts/run_benchmark.py --domain legal --verbose
        """,
    )
    parser.add_argument(
        "--domain", type=str, default="all",
        choices=["finance", "legal", "technical", "all"],
        help="Domain to evaluate (default: all)",
    )
    parser.add_argument(
        "--pipeline", type=str, default=None,
        choices=["pageindex", "roaming", "bm25", "agentic", "hybrid_sota"],
        help="Run only a specific pipeline (default: all)",
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
    config = load_config()
    config["results"]["output_dir"] = args.output_dir

    logger.info("=" * 60)
    logger.info("  Vectorless RAG Benchmark Suite")
    logger.info("=" * 60)
    logger.info(f"  Domain: {args.domain}")
    logger.info(f"  Pipeline: {args.pipeline or 'all'}")
    logger.info(f"  Output: {args.output_dir}")
    logger.info("=" * 60)

    # Get pipelines
    pipelines = get_pipelines(args.pipeline)
    if not pipelines:
        logger.warning(
            "No pipelines instantiated yet. "
            "Implement pipeline classes in src/pipelines/ first."
        )
        logger.info("Project scaffolding is complete. Next steps:")
        logger.info("  1. Add documents to data/raw/{domain}/")
        logger.info("  2. Implement pipeline classes")
        logger.info("  3. Generate golden Q&A pairs")
        logger.info("  4. Run this benchmark again")
        return

    # Run evaluation
    runner = EvaluationRunner(config)
    domains = ["finance", "legal", "technical"] if args.domain == "all" else [args.domain]

    for domain in domains:
        domain_config = next(
            (d for d in config["corpus"]["domains"] if d["name"] == domain), None
        )
        if not domain_config:
            logger.error(f"Domain {domain} not found in config")
            continue

        logger.info(f"\n--- Domain: {domain} ---")
        runner.run(
            pipelines=pipelines,
            golden_qa_path=domain_config["golden_qa_path"],
            corpus_path=domain_config["processed_path"],
            domain=domain,
        )

    logger.info("\n✅ Benchmark complete! Results in: " + args.output_dir)


if __name__ == "__main__":
    main()
