# Benchmark Methodology — 2026-05-25

## Retrieval Backends Compared
1. BM25 (Okapi) — sparse keyword
2. TF-IDF — classical IR
3. LLM-as-Reranker — GPT-4o-mini reranking
4. ColBERT late-interaction
5. Hybrid: BM25 + LLM judge

## Evaluation Dataset
- SQuAD mini (500 Q&A pairs)
- RAGAS metrics: faithfulness, answer relevancy, context precision

## Key Finding
Hybrid BM25+LLM: RAGAS=0.87, wins on quality
BM25 alone: RAGAS=0.71, zero embedding cost
