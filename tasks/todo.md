# Vectorless RAG Benchmark — Task Tracker

## Current Sprint (Session 3)

### 🔴 Critical
- [x] Install Ollama + start service
- [x] Pull llama3.2:3b model
- [ ] Pull qwen3:8b model (retrying — network failed at 6%)
- [x] Implement Embedding-Free RAG pipeline (Maghakian et al.)
- [x] Implement Three-Stage Hybrid pipeline (E.1 novel contribution)
- [x] Add psutil memory/latency telemetry (C.3/C.4)
- [x] Run BM25 benchmark on finance (5 questions — passed ✅)
- [ ] Run BM25 benchmark full (all 337 finance questions)
- [ ] Update all configs to use qwen3:8b once downloaded
- [ ] Run full benchmark with qwen3:8b

### 🟡 Important
- [ ] Add F1/EM string-match metrics (C.3)
- [ ] Point RAGAS at Ollama endpoint (C.1)
- [ ] Error analysis cross-tab: paradigm × query-type (C.6)
- [ ] Download LegalBench-RAG real data (B.2)
- [ ] Download FRAMES multi-hop data (B.2)

### 🟢 Future
- [ ] In-Memory KNN pipeline — Khan's PCA+PQ+HNSW (D.7)
- [ ] MCP server pattern (D.5)
- [ ] MLflow experiment tracking (F.3)
- [ ] uv migration from pip (F.2)
- [ ] Adaptive router on RAGRouter-Bench labels (E.4)
- [ ] Pareto frontier figure: latency × F1 × peak RSS (E.6)

## Completed
- [x] Project scaffolding (Phase 0)
- [x] 5 original RAG pipeline implementations (Phase 2)
- [x] Dataset download + preprocessing: 30 docs, 89K words (Phase 1)
- [x] 370 golden Q&A pairs generated deterministically
- [x] LLM client rewrite: Ollama + Gemini + OpenAI + Anthropic
- [x] Embedding-Free RAG (6th pipeline)
- [x] Three-Stage Hybrid (7th pipeline — novel contribution)
- [x] Telemetry module: peak RSS, p50/p95 latency, error taxonomy
- [x] Benchmark runner rewrite with telemetry integration
- [x] Git: 4 commits pushed to github.com/ejazfahil/ag_vectorless_RAG

## BM25 Benchmark Results (5Q smoke test)
```
Success rate: 100%
Latency: p50=18.46s | p95=19.39s | mean=17.89s
Memory: peak RSS=31MB
Tokens: total=21,730 | mean/query=4,346
Cost: $0.000000
```
