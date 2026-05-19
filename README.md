# 🚀 Vectorless RAG Benchmark

> A **State-of-the-Art Vectorless RAG system** and rigorous benchmark comparing 5 embedding-free retrieval paradigms across 4 evaluation dimensions.

## 🎯 Overview

This project builds and benchmarks **embedding-free Retrieval-Augmented Generation** systems — RAG pipelines that operate **without vector databases or dense embeddings**. We implement 5 distinct paradigms and compare them on Performance, Error Analysis, System Maintenance, and Cost Efficiency.

### Pipelines Compared

| # | Pipeline | Approach | Source |
|---|----------|----------|--------|
| 1 | **PageIndex** | Hierarchical tree reasoning | VectifyAI (2026) |
| 2 | **Roaming RAG** | Agentic document navigation | Arcturus Labs (2025) |
| 3 | **BM25 Lexical** | TF-IDF keyword search (Elasticsearch) | Classic IR |
| 4 | **Agentic Search-First** | Multi-agent with massive context | OpenAI paradigm (2025) |
| 5 | **Hybrid Vectorless (Ours)** | Adaptive routing + tree + BM25 + verification | Novel |

### Evaluation Dimensions

| Dimension | Tool/Method | Metrics |
|-----------|-------------|---------|
| **Performance** | RAGAS | Faithfulness, Answer Relevancy, Context Precision/Recall |
| **Error Analysis** | LLM-as-a-Judge | Error categorization, hallucination rate, severity |
| **Maintenance** | Custom | Ingestion time, update time, storage, reindex flag |
| **Cost** | Token tracking | $/query, tokens/query, projected daily cost |

## 🏗️ Project Structure

```
vectorless-rag-benchmark/
├── configs/          # YAML configs for each pipeline
├── data/             # Corpus and golden Q&A sets
├── src/
│   ├── corpus/       # Document preprocessing & QA generation
│   ├── pipelines/    # 5 RAG pipeline implementations
│   ├── evaluation/   # 4-dimension evaluation harness
│   └── utils/        # LLM client, token counter, logging
├── scripts/          # CLI entry points
├── results/          # Experiment outputs & figures
└── tests/            # Unit tests
```

## 🚀 Quick Start

```bash
# 1. Clone & setup
git clone <repo-url>
cd vectorless-rag-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Configure
cp .env.example .env
# Edit .env with your API keys

# 3. Add documents to data/raw/{finance,legal,technical}/

# 4. Run benchmark
python scripts/run_benchmark.py --domain finance --verbose
```

## 📊 Results

Results will be populated after running the benchmark suite.

## 📖 References

1. VectifyAI/PageIndex — [github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)
2. Roaming RAG — Arcturus Labs (2025)
3. Maghakian et al., "Embedding-Free RAG", EMNLP 2025
4. RAGAS Framework — [docs.ragas.io](https://docs.ragas.io)

## 📄 License

MIT
