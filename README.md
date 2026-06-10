# 🔍 Vectorless RAG Benchmark

[![CI](https://github.com/ejazfahil/ag_vectorless_RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/ejazfahil/ag_vectorless_RAG/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A rigorous benchmark of **5 embedding-free retrieval paradigms** for production RAG systems — comparing BM25, TF-IDF, LLM-as-Reranker, ColBERT late-interaction, and Hybrid approaches across latency, cost, accuracy, and maintainability.

## 🎯 Motivation

Vector databases add infrastructure cost and operational complexity. This project answers: **when can you skip embeddings entirely** — and what do you trade off?

## 📊 Benchmark Results

| Method | RAGAS Score | Latency (p50) | Cost/1K queries | Maintenance |
|--------|-------------|---------------|-----------------|-------------|
| BM25 | 0.71 | 12ms | $0.00 | Low |
| TF-IDF | 0.68 | 8ms | $0.00 | Low |
| LLM-as-Reranker | 0.84 | 340ms | $0.42 | Medium |
| ColBERT | 0.81 | 55ms | $0.00 | High |
| Hybrid BM25+LLM | **0.87** | 360ms | $0.44 | Medium |

## 🏗️ Architecture

```
Query
  │
  ├── BM25Backend ──────────────────────► Ranked docs
  ├── TFIDFBackend ─────────────────────► Ranked docs
  ├── LLMReranker (GPT-4o-mini) ────────► Reranked docs
  ├── ColBERTBackend ───────────────────► Late-interaction scores
  └── HybridBackend (BM25 + LLMJudge) ─► Best of both worlds
          │
          └── RAGAS Evaluator ──────────► Faithfulness, Answer Relevancy, Context Precision
```

## 🚀 Quickstart

```bash
git clone https://github.com/ejazfahil/ag_vectorless_RAG.git
cd ag_vectorless_RAG
pip install -r requirements.txt

# Run BM25 retrieval
python -m src.backends.bm25_backend

# Run full benchmark
python benchmark.py --method all --dataset squad_mini
```

### 🐳 Containerised + offline (Docker + local LLM)

The repo ships a multi-stage `Dockerfile` and a `docker-compose.yml` that brings up the
app alongside **Elasticsearch** (the BM25 / hybrid backend). The LLM client auto-detects
a local [Ollama](https://ollama.com) server, so the whole benchmark runs **offline with
no API keys and zero per-token cost** — data never leaves the machine:

```bash
ollama pull qwen3:8b          # models referenced by the configs (also: llama3.2:3b)
docker compose up -d --build  # app :8501 (Streamlit) + Elasticsearch :9200
```

Inside a container the host's Ollama is reached via `OLLAMA_BASE_URL`
(default `http://localhost:11434`; set to `http://host.docker.internal:11434` in compose
for macOS). Verified end-to-end against `qwen3:8b`.

## 📁 Project Structure

```
ag_vectorless_RAG/
├── src/
│   └── backends/
│       ├── bm25_backend.py      # Okapi BM25 retrieval
│       ├── tfidf_backend.py     # TF-IDF retrieval
│       └── llm_reranker.py      # GPT-4o-mini reranking
├── tests/
│   └── test_bm25.py
├── .github/workflows/ci.yml
├── benchmark.py                 # End-to-end benchmark runner
└── README.md
```

## 🔬 Evaluation Methodology

All backends evaluated on a 500-question subset of SQuAD using **RAGAS** metrics:
- **Faithfulness** — does the answer come from the retrieved context?
- **Answer Relevancy** — is the answer relevant to the question?
- **Context Precision** — how much of the retrieved context is useful?

## 📌 Key Findings

- **BM25 wins on cost**: zero embedding cost, 71% of hybrid quality
- **Hybrid wins on quality**: worth it when answer precision is critical
- **LLM reranking is bottleneck**: ~300ms latency overhead
- **For keyword-heavy domains** (legal, medical codebooks): BM25 ≈ vector search

## 🔭 Next Steps
- [ ] Add SPLADE sparse retrieval backend
- [ ] Benchmark on domain-specific datasets (legal, medical)
- [ ] Add streaming retrieval evaluation
- [ ] Docker compose for reproducible benchmarks

## 📄 License
MIT
