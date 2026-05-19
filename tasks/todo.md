# Vectorless RAG Benchmark — Task Tracker

## Current Sprint — Full System Build (new.md explicit)

### 🔴 Critical (new.md explicit requirements)
- [x] Project scaffolding (Phase 0)
- [x] 7 RAG pipeline implementations (Phases 2-4)
- [x] Dataset download + preprocessing: 30 docs, 89K words
- [x] 370 golden Q&A pairs generated deterministically
- [x] LLM client: Ollama + Gemini + OpenAI + Anthropic
- [x] Telemetry module: peak RSS, p50/p95 latency, error taxonomy
- [x] String metrics: F1/EM/Precision/Recall
- [x] **[new.md D.7]** In-Memory KNN pipeline — PCA+PQ+HNSW (Khan) → `src/pipelines/knn_rag.py`
- [x] **[new.md D.7]** `configs/knn.yaml` config
- [x] **[new.md E.4]** Adaptive Router — ML heuristic + LLM fallback (hybrid mode) → `src/router.py`
- [x] **[new.md E.4]** `configs/router.yaml` config
- [x] **[new.md C.1]** RAGAS pointed at Ollama via OpenAI-compat endpoint → `src/evaluation/ragas_eval.py`
- [x] **[new.md C.3]** Wire F1/EM metrics into benchmark runner output
- [x] **[new.md A/D]** Added qwen3:4b/8b/14b/32b + deepseek-r1 family to llm_client OLLAMA_MODELS
- [x] **[new.md B.4]** bm25s library fast-path in BM25RAG (graceful fallback to InMemoryBM25)
- [x] **[new.md F.3]** MLflow local experiment tracking (`--mlflow` flag in benchmark runner)
- [x] **[new.md C.6]** Error cross-tab analysis → `scripts/analyze_errors.py`
- [x] Update benchmark runner with KNN + router + metrics + MLflow
- [x] Update pyproject.toml with new dependencies
- [x] Update src/pipelines/__init__.py exports (8 pipelines total)

### 🟡 Important
- [ ] Pull qwen3:8b model (retrying — network failed at 6%)
- [ ] Run full benchmark with qwen3:8b on all 3 domains
- [ ] Download LegalBench-RAG real data (B.2)
- [ ] Download FRAMES multi-hop data (B.2)

### 🟢 Future
- [ ] MCP server pattern (D.5)
- [ ] Pareto frontier figure: latency × F1 × peak RSS (E.6)
- [ ] Bootstrap CI for statistical significance (C.5)

## Completed
- [x] All 7 pipelines: PageIndex, Roaming, BM25, Agentic, HybridSoTA, EmbeddingFree, ThreeStageHybrid
- [x] Full evaluation framework: RAGAS, LLM judge, maintenance, cost tracker
- [x] BM25 smoke test: 100% success, p50=18.46s, peak RSS=31MB, $0 cost
- [x] Git: 4 commits pushed

## BM25 Benchmark Results (5Q smoke test)
```
Success rate: 100%
Latency: p50=18.46s | p95=19.39s | mean=17.89s
Memory: peak RSS=31MB
Tokens: total=21,730 | mean/query=4,346
Cost: $0.000000
```
