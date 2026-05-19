# Building a Fully Local, Zero-Cost, Vectorless RAG System on an Apple M4 MacBook Air: A Technical Blueprint

## TL;DR

- **Recommended stack**: Run an Ollama-hosted MLX-backed (or LM Studio MLX) **Qwen3-8B / Qwen3-14B-MLX-4bit** as the reasoning LLM, with **bm25s** for sparse lexical retrieval, **PageIndex** (self-hosted via `pip install pageindex` with `--model` pointed at your local OpenAI-compatible Ollama endpoint) for hierarchical/Roaming RAG, and a custom **Embedding-Free RAG** implementation following Maghakian et al. (EMNLP 2025) on top of LangChain — all on a 16GB or 24GB M4 MacBook Air. Total cost: $0.
- **The four paradigms are complementary, not competitive**: BM25 (bm25s/Elasticsearch) is your sub-millisecond first-stage filter; PageIndex/Roaming RAG gives you human-style hierarchical reasoning over a single long document; Embedding-Free RAG (Maghakian et al., reaching 82% on FinanceBench vs. a 50% Traditional RAG baseline and 2.6× average F1 improvement on LegalBench-RAG) gives you a one-size-fits-all zero-tuning pipeline; agentic/MCP pipelines let the LLM choose which of the above to invoke. Build an **adaptive router** on top.
- **Evaluation**: Run **RAGAS pointed at Ollama via its OpenAI-compatible endpoint** (`base_url="http://localhost:11434/v1"`) and **DeepEval with `deepeval set-ollama`** for LLM-as-a-judge. Benchmark head-to-head on **FRAMES (824 multi-hop QA), LegalBench-RAG, FinanceBench, MuSiQue, and 2WikiMultiHopQA**. Target metrics: Faithfulness, Context Precision/Recall, F1, hallucination rate, p50/p95 latency, peak RSS.

---

## A. Local LLM Stack Options for Apple Silicon M4

### A.1 The Five Runtimes That Matter

| Runtime | Strengths | M4 Air Verdict |
|---|---|---|
| **Ollama** | One-command install, OpenAI-compatible API on `localhost:11434`, huge model library, native tool-calling | **Default choice.** Use this for everything unless you need MLX-specific speed |
| **LM Studio** | GUI; can run MLX-optimized models on 16GB Macs (Ollama's MLX backend currently requires 32GB+) | **Best for the 16GB M4 Air** when you want MLX speed |
| **MLX-LM** (`pip install mlx-lm`) | Apple's native framework purpose-built for unified memory; in independent benchmarks on Apple M2 Ultra with Qwen-2.5-7B-Instruct 4-bit, MLX reached ~230 tok/s vs. llama.cpp's ~150 tok/s (a ~53% throughput advantage); on M4 Max with vllm-mlx, the gap for models under 14B parameters ranges 20–87% | Use for raw benchmarking and the `mlx_lm.server` OpenAI-compatible server |
| **llama.cpp** | Reference implementation; widest cross-platform support; GGUF format is de facto standard | Use for scripting/batch and parameter tuning (`n_gpu_layers=-1`) |
| **Jan.ai / GPT4All** | Polished GUIs | Useful for non-coding stakeholders; not for the research pipeline |

Ollama 0.19 (March 30, 2026 release; testing March 29, 2026) shipped a preview **MLX backend** activated by `OLLAMA_MLX=1 ollama serve`; on Apple's reference M5 Max with Qwen3.5-35B-A3B, decode rate jumped from **58 → 112 tokens/sec** (a near-doubling), and a higher-performance int4 mode reached **1,851 tok/s prefill and 134 tok/s decode**. Crucially, the MLX backend has a **hard 32GB unified-memory requirement** — so on a 16GB M4 MacBook Air it will silently fall back to the llama.cpp backend. If you bought the 16GB SKU, use **LM Studio's MLX runtime** instead to get MLX speed.

### A.2 Models — Sized for the M4 MacBook Air Memory Tiers

Memory-bandwidth specs (the binding constraint for token-generation speed): base M4 chip ~120 GB/s; M4 Pro **exactly 273 GB/s** (Apple's official October 2024 spec); M4 Max **exactly 546 GB/s** in the 16-core CPU / 40-core GPU configuration (or 410 GB/s in the 14-core/32-core trim). The base M4 is usable but slower than M4 Pro/Max for sustained generation.

**8 GB M4 Air** (very tight; keep ≤4B params at Q4):
- **Qwen3-4B-MLX-4bit** — best reasoning-per-byte; supports `enable_thinking=True` for chain-of-thought
- **Phi-3-mini 3.8B Q4** (fallback)
- **Llama-3.2-3B-Instruct Q4_K_M**

**16 GB M4 Air** (the sweet spot for this project):
- **Qwen3-8B-MLX-4bit** — primary reasoning model. Strong reasoning, native tool-calling, 32K native context (131K with YaRN)
- **Llama-3.1-8B-Instruct Q4_K_M** — proven for RAG; ~60-80 tok/s at Q4
- **Mistral-7B-Instruct-v0.3 Q4_K_M** — fast, strong baseline
- **DeepSeek-R1-Distill-Qwen-7B** — for reasoning ablation studies
- **Gemma-2-9B-it Q4_K_M** — Google alternative, good instruction following
- **Phi-3-medium 14B Q4** — borderline; may swap

**24 GB M4 Air**:
- **Qwen3-14B-MLX-4bit** — top reasoning model in this tier; the F1=0.2821 leader on PrivacyQA in the Embedding-Free RAG ablation (Maghakian et al., Table 2) used Qwen 2.5-14B, foreshadowing Qwen3-14B's likely strength
- **DeepSeek-R1-Distill-Qwen-14B** — for chain-of-thought traceability in PageIndex tree navigation
- **Llama-3.1-8B at Q8_0** (better quality than Q4)

### A.3 Best Models for the *Reasoning* Tasks Required by Vectorless RAG

Vectorless RAG is reasoning-bound, not memory-bound. The LLM must navigate trees, generate JSON-structured navigation decisions, and self-verify. Empirical pick order, from Maghakian et al.'s Table 2 (PrivacyQA retrieval F1):

1. **Qwen2.5-14B (F1=0.2821, Precision=0.1787, Recall=0.6684)** — best F1 in their ablation
2. **Llama-3.1-70B (F1=0.2717)** — too big for M4 Air
3. **DeepSeek-R1-Distill-Llama-70B (Recall=0.7456, highest)** — too big for M4 Air; the distilled 14B variant is the M4 substitute
4. **Llama-3.1-8B (F1=0.1714, Recall=0.6315)** — the smallest tested. Paper's verbatim claim: *"The smallest Llama 3.1 models still achieved recall twice that of the SOTA Traditional RAG pipeline."*

The practical M4 Air picks therefore are **Qwen3-14B-MLX-4bit** (24GB SKU) or **Qwen3-8B-MLX-4bit** (16GB SKU). For PageIndex specifically, prioritize models that produce reliable JSON; Qwen3 and DeepSeek-R1 distills are best.

### A.4 Setup Commands

```bash
# Ollama — default
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:8b           # or qwen3:14b on 24GB
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct
ollama serve                   # exposes OpenAI-compatible API on :11434

# Optional MLX backend (24GB+ only)
OLLAMA_MLX=1 ollama serve

# MLX-LM (native)
python -m venv .venv && source .venv/bin/activate
pip install mlx-lm
mlx_lm.server --model mlx-community/Qwen3-8B-MLX-4bit --port 8080

# LM Studio — best 16GB MLX option
# Download from lmstudio.ai; in-app: search "Qwen3-8B", filter "MLX", click Download
```

### A.5 Which Stack for Which Paradigm

| Paradigm | Recommended Runtime | Recommended Model |
|---|---|---|
| PageIndex / Roaming RAG | Ollama or LM Studio MLX (long context) | Qwen3-8B/14B (32K+ context, JSON-reliable) |
| BM25 lexical | Any (LLM only used post-retrieval) | Llama-3.1-8B or Mistral-7B |
| Agentic / MCP | Ollama (best tool-calling support) | Qwen3-8B (native tools) |
| Embedding-Free RAG | Two-tier: small model for quote extraction + larger for answer | Qwen3-4B (extract) + Qwen3-14B (answer) — mirrors Maghakian's Gemini-Flash+GPT-4-Turbo pattern |
| In-memory KNN (Khan) | MLX-LM for embeddings + Ollama for generation | all-MiniLM-L6-v2 (384-D) + Llama-3.1-8B |

---

## B. Test Corpus Preparation

### B.1 Choosing the Benchmark Corpus

For evaluating **vectorless / embedding-free** systems specifically, the corpus must satisfy three criteria: (1) documents are **long and hierarchically structured** (so PageIndex/Roaming RAG have something to roam), (2) the gold answers are **span-level**, not document-level (so retrieval precision actually moves), and (3) at least some queries are **multi-hop** (so simple BM25 fails and the LLM-reasoning paradigms can shine).

### B.2 Dataset Survey and Paradigm Mapping

| Dataset | Size / Type | Best For Testing |
|---|---|---|
| **FRAMES** (Google/Harvard, 2024) | 824 multi-hop Wikipedia questions with temporal disambiguation | **Adaptive router + agentic pipelines.** Baseline LLM accuracy without retrieval = **0.408**; multi-step retrieval brings this to **0.66** (>50% improvement) per Krishna et al., arXiv:2409.12941 |
| **LegalBench-RAG** (Pipitone & Alami, 2024; on GitHub at `zeroentropy-cc/legalbenchrag`) | 4 subsets (PrivacyQA, ContractNLI, CUAD, MAUD); span-level gold; ~6929 NDA-related queries | **Embedding-Free RAG, PageIndex.** This is the dataset where Maghakian et al. achieved 2.6× average F1 over the best of 28 SOTA RAG pipelines; MAUD precision/recall jumped from (0.01, 0.31) baseline to (0.08, 0.66) with Embedding-Free RAG |
| **FinanceBench** | SEC filings / earnings reports QA | **PageIndex (VectifyAI's Mafin 2.5 hit 98.7% accuracy here vs. 60–80% for traditional vector RAG) and Embedding-Free RAG (82% correct vs. 50% Traditional, 79% long-context Gemini-1.5-Flash, 85% oracle)** |
| **HotpotQA** | 113K Wikipedia multi-hop | **BM25 + reranking baselines; agentic multi-hop** |
| **2WikiMultiHopQA** | 192K multi-hop with explicit reasoning paths | **Roaming RAG with section IDs** |
| **MuSiQue** | 25K 2-4 hop, compositional | **PageIndex tree-search; one of the four corpora in RAGRouter-Bench** |
| **Natural Questions** | Google search queries → Wikipedia | BM25 baselining; the Khan (Manual RAG) paper reports MRR@5=0.87 on this |
| **TriviaQA** | 95K trivia | **Embedding-Free RAG ablations** (factual recall) |
| **SQuAD / SQuAD 2.0** | 100K+ reading comprehension | Single-doc PageIndex sanity check |
| **QASPER** | 1585 scientific paper QA | **PageIndex / Roaming RAG over arXiv-style long-form** |
| **SCROLLS** | Long-document QA / summarization | **PageIndex sweet spot** |
| **BEIR** (18 datasets across 9 task types) | TREC-COVID, BioASQ, NFCorpus, SciFact, etc. | **BM25 baselining + lexical pipelines**; BM25 is the famously robust BEIR baseline. Available via `pip install beir` |
| **RAGRouter-Bench** (Wang et al., 2026) | 7,727 queries × 4 domains (Wikipedia, literature, legal, medical) × 5 RAG paradigms | **Adaptive router training** |
| **RAGTruth** | ~18,000 annotated responses | **Hallucination rate evaluation** |
| **RULER** | Synthetic NIAH-style long-context | Stress-testing PageIndex tree depth |

### B.3 Recommended Concrete Test Corpus for This Project

For the M4 MacBook Air with limited compute, the pragmatic test corpus is:

1. **FinanceBench** (10K-token SEC filings, 150 questions) — your PageIndex / Embedding-Free RAG headline benchmark
2. **LegalBench-RAG PrivacyQA + MAUD subsets** — span-level gold to measure precision/recall properly
3. **FRAMES (subset of 200 questions)** — for the adaptive router and multi-hop
4. **A custom 50-question gold set you author yourself** over **3–5 of your own domain PDFs** — gives you real-world signal; use the LlamaIndex `DatasetGenerator` to bootstrap, then hand-validate

### B.4 Document Preparation Pipelines

**For BM25:**
```python
# pip install bm25s langchain-text-splitters PyStemmer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bm25s, Stemmer
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
chunks = splitter.split_documents(docs)
stemmer = Stemmer.Stemmer("english")
retriever = bm25s.BM25()
retriever.index(bm25s.tokenize([c.page_content for c in chunks], stemmer=stemmer))
```
`bm25s` benchmarks within striking distance of Elasticsearch on a single node while being a pure-Python, low-dependency library.

**For PageIndex (hierarchical tree):**
Use **PageIndex OCR** to convert PDFs to Markdown preserving hierarchy. If you start from already-clean Markdown, the VectifyAI repo's recommended path is `pip install pageindex` and the `python3 examples/agentic_vectorless_rag_demo.py` script — pointed at your local Ollama with `base_url="http://localhost:11434/v1"`. The IBM "Bob" demo at `dev.to/aairom` is a working reference for the Ollama-backed integration.

**For Roaming RAG:**
Convert source documents to **llms.txt** format (Jeremy Howard's standard: a `/llms.txt` Markdown file with project name, summary, and section links). Then assign each section a unique ID and provide the LLM with two tools: `outline()` (returns the table of contents) and `expand_section(id)` (returns full content of a section). Arcturus Labs' reference implementation is ~300 lines.

**For Embedding-Free RAG (Maghakian et al.):**
Split documents into **3000-word subdocuments** (the paper's default). Sentence-split each subdocument; the algorithm operates on sentence indices. Prepend a **2-3 sentence summary built from the first 5,000 words** to each subdocument for global context during parallelized quotation generation.

**For in-memory KNN (Khan):**
Use **sentence-transformers/all-MiniLM-L6-v2** (384-dim, ~80 MB, runs at 1000+ sentences/sec on M4). Apply **PCA to 256-dim**, then **Product Quantization to ~0.5 KB per embedding**, then build an **HNSW graph** index using `hnswlib`. Khan's paper reports 60% memory reduction and 45% latency improvement for 10K–1M document knowledge bases, and MRR@5 = 0.87 on Natural Questions and TriviaQA.

### B.5 Gold-Standard QA Generation (Local, Zero-Cost)

```python
# Use a strong local LLM to *propose* questions, then hand-validate
from llama_index.llms.ollama import Ollama
from llama_index.core.evaluation import DatasetGenerator
llm = Ollama(model="qwen3:14b", request_timeout=300.0)
generator = DatasetGenerator.from_documents(docs, llm=llm, num_questions_per_chunk=2)
eval_dataset = generator.generate_dataset_from_nodes()
# Persist to JSONL, then manually review every question + ground-truth span
```

Best practice: **generate 5×** the questions you need with the strongest local model (Qwen3-14B), then prune to the cleanest 20%. Always hand-validate ground-truth spans — never auto-trust LLM-generated answers for your benchmark.

### B.6 Tooling — Document Loaders

```bash
# Apple Silicon native install (no Docker required)
brew install libmagic poppler tesseract libxml2 libxslt
pip install "unstructured[pdf]" langchain-unstructured pypdf pymupdf
pip install llama-index llama-index-readers-file
```

Avoid `detectron2` on M4 — it's painful to build. The `fast` strategy in Unstructured falls back to `pdfminer.six` and is sufficient for most academic, legal, and financial documents.

### B.7 Domain-Specific Corpus Picks

- **Legal**: LegalBench-RAG (covers NDAs, M&A agreements, commercial contracts, privacy policies); CUAD (Contract Understanding)
- **Medical**: PubMedQA, BioASQ (in BEIR), GraphRAG-Bench medical subset
- **Technical/API docs**: arXiv subsets via QASPER; the Angular/Vercel `llms.txt` files for Roaming RAG dogfooding
- **General knowledge**: FRAMES, MuSiQue, 2WikiMultiHopQA

---

## C. Evaluation Framework — Fully Local, Zero-Cost

### C.1 RAGAS Pointed at Ollama

```python
from openai import OpenAI
from ragas.llms import llm_factory
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
evaluator_llm = llm_factory("qwen3:14b", provider="openai", client=client)
```

This pattern is directly from the RAGAS quickstart; Ollama doesn't require a real key, so we pass `"ollama"`. The judge model should be **stronger than the system-under-test** — if your RAG uses Qwen3-8B, judge with Qwen3-14B; if it uses Qwen3-14B, judge with Qwen3-14B in `enable_thinking=True` mode (chain-of-thought reasoning).

### C.2 DeepEval Alternative

```bash
ollama run qwen3:14b
deepeval set-ollama qwen3:14b
```

DeepEval now natively supports Ollama as judge — its metrics (`AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`) all run locally.

### C.3 Metrics To Measure

| Metric | What It Tells You | How to Compute Locally |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | RAGAS extracts statements via the V1 statement-generator prompt, then LLM-judges entailment from context; score = fraction supported |
| **Answer Relevancy** | Does the answer address the question? | RAGAS embeds answer + reverse-generated questions; pure local with `nomic-embed-text` via Ollama |
| **Context Precision** | Of retrieved chunks, how many are useful? | LLM-judge returns binary "useful/not useful" per chunk, then Average Precision |
| **Context Recall** | Did we retrieve everything needed for the ground-truth answer? | NLI-style entailment over reference answer |
| **F1 / EM** | String-match precision/recall on spans | Compute directly; for LegalBench-RAG follow the official precision/recall@k script |
| **Hallucination rate** | Fraction of answers containing unsupported claims | Faithfulness < 1.0 threshold counts as hallucination; cross-check against RAGTruth's four-category taxonomy |
| **p50 / p95 latency** | User-facing speed | `time.perf_counter()` around each query; record both retrieval and generation phases separately |
| **Cost per query** | Compute cost on M4 | Estimate from `(input_tokens + output_tokens) / tokens_per_sec * watts`; M4 base draws ~20W at sustained load → ~$0.000001 per query at retail electricity |
| **Memory (peak RSS)** | Did you blow up unified memory? | `psutil.Process().memory_info().rss` + macOS `vm_stat`; alarm at >85% of unified memory |

### C.4 Automated Evaluation Pipeline Skeleton

```python
import time, json, psutil
from pathlib import Path
results = []
for q in eval_set:
    p = psutil.Process()
    mem_before = p.memory_info().rss
    t0 = time.perf_counter()
    ans, ctx = rag_pipeline.run(q["question"])   # your pipeline
    t1 = time.perf_counter()
    mem_peak = p.memory_info().rss
    results.append({
        "question": q["question"], "answer": ans, "context": ctx,
        "reference": q["answer"], "latency_s": t1-t0,
        "mem_delta_mb": (mem_peak - mem_before)/1e6,
    })
Path("runs.jsonl").write_text("\n".join(json.dumps(r) for r in results))
# Then RAGAS over runs.jsonl → metrics.csv
```

### C.5 Head-to-Head Paradigm Comparison

For each of the four paradigms, run the **same 200-question evaluation set** through identical prompts where possible. Aggregate by query type (factual / multi-hop / summarization / numerical) per RAGRouter-Bench's taxonomy. Statistical significance: bootstrap 1000 samples; report 95% CIs. Track results in MLflow or in CSV; a single Python script comparing all four lets you produce the publication-ready table.

### C.6 Error Analysis

For each failure, log:
1. **Retrieval failure** (context didn't contain answer)
2. **Reasoning failure** (context had answer, LLM ignored it)
3. **Hallucination** (LLM invented unsupported facts)
4. **Format failure** (couldn't parse LLM's JSON in PageIndex/Roaming)

Cross-tab failures by paradigm × query type. This drives the adaptive router design (Section E.4).

---

## D. System Architecture & Implementation Guide

### D.1 Project Skeleton (uv-managed)

```
vectorless-rag-m4/
├── pyproject.toml
├── uv.lock
├── .env                       # OLLAMA_HOST=http://localhost:11434
├── data/
│   ├── corpora/{finance,legal,wiki}/
│   └── eval/{frames,legalbench,custom}.jsonl
├── src/
│   ├── paradigms/
│   │   ├── pageindex_runner.py
│   │   ├── roaming_rag.py
│   │   ├── bm25_runner.py
│   │   ├── agentic_mcp.py
│   │   ├── embedding_free.py   # Maghakian et al.
│   │   └── inmem_knn.py        # Khan
│   ├── router.py               # adaptive paradigm selector
│   ├── eval/
│   │   ├── ragas_runner.py
│   │   ├── deepeval_runner.py
│   │   └── metrics.py
│   └── utils/{loaders,chunkers,telemetry}.py
├── notebooks/
└── results/{runs.jsonl,metrics.csv,figures/}
```

Use **uv** instead of pip/conda — it's significantly faster on Apple Silicon. `uv init && uv add langchain langchain-ollama llama-index ragas deepeval bm25s rapidfuzz hnswlib scikit-learn faiss-cpu pageindex unstructured pypdf streamlit`.

### D.2 Paradigm 1 — PageIndex (Vectorless / Hierarchical)

Self-hosted PageIndex points at your local Ollama. The VectifyAI repo (**30,800 GitHub stars / 2,600+ forks as of May 2026**, having hit #1 on GitHub Trending in early 2026) ships `examples/agentic_vectorless_rag_demo.py` as the canonical starter for self-hosted vectorless RAG with the OpenAI Agents SDK.

```python
# Pseudo, following VectifyAI's API
from pageindex import PageIndex
client = PageIndex(base_url="http://localhost:11434/v1", model="qwen3:14b", api_key="ollama")
tree = client.build_tree("data/corpora/finance/10K.md")  # JSON tree of TOC
def query(q):
    plan = client.reason_over_tree(tree, q)              # LLM picks nodes
    sections = [client.get_node_content(tree, nid) for nid in plan.node_ids]
    return client.generate_answer(q, sections)
```

Two key implementation tricks: **(1)** prepend each tree node's `summary` field to help the LLM choose — don't make it read every node's full content during navigation; **(2)** allow the LLM to expand multiple nodes at the same level in parallel ("breadth-first scan") then deepen, which mirrors AlphaGo-style MCTS. PageIndex's MCP server (`VectifyAI/pageindex-mcp`) exposes the tree to any MCP-compatible client.

### D.3 Paradigm 1b — Roaming RAG

```python
# After parsing document into sections with unique IDs
TOOLS = [
    {"name": "outline", "description": "Returns the document table of contents."},
    {"name": "expand_section", "parameters": {"id": "str"}},
]
SYSTEM = """You are a research assistant. Use outline() first, then expand_section(id) 
to read relevant sections. You can expand multiple sections at the same level in 
parallel. When you have enough information, answer the question with citations."""
# Loop: LLM call → tool call → tool result back to LLM → ... until final answer
```

Roaming RAG is **best for `llms.txt`-formatted documents** and structured manuals. Arcturus Labs' reference is the canonical ~300-line implementation.

### D.4 Paradigm 2 — BM25

```python
# pip install bm25s PyStemmer
import bm25s, Stemmer
stemmer = Stemmer.Stemmer("english")
corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)
retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
retriever.index(corpus_tokens)
results, scores = retriever.retrieve(bm25s.tokenize(query, stemmer=stemmer), k=5)
```

For LangChain integration, use `from langchain_community.retrievers import BM25Retriever` (now supporting **BM25Plus** which reduces bias against short documents — pass `bm25_variant="plus"`). For metadata filtering at scale, use **Elasticsearch in single-node mode** via Docker (`docker run -p 9200:9200 -e discovery.type=single-node elasticsearch:8.x`) with the LangChain `ElasticsearchStore` and `BM25Strategy` per the Unstructured.io tutorial.

For hybrid retrieval as the **first-pass filter for vectorless reasoning paradigms** (recommended): BM25 → top-30 sections → PageIndex tree reasoning over just those. This is the "lightweight vector search for pruning" pattern.

### D.5 Paradigm 3 — Agentic & MCP

**MCP server pattern:**
```python
# pip install mcp[cli]
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("rag-server")
@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> list[dict]:
    """BM25 search over local corpus."""
    return bm25_retrieve(query, top_k)
@mcp.tool()
def get_section(doc_id: str, section_id: str) -> str:
    """Roaming RAG-style section fetch."""
    ...
if __name__ == "__main__":
    mcp.run()
```

Register in `claude_desktop_config.json` or call from a local Ollama-backed agent loop. The `shinpr/mcp-local-rag` MCP server is a working reference offering 7 tools (`ingest_file`, `query_documents`, `read_chunk_neighbors`, etc.) and supports both MCP and CLI mode.

**Multi-agent without cloud APIs:**
- **CrewAI** or **AutoGen** point both at a local Ollama URL
- For the "RAG without the R" massive-context pattern: Qwen3's native 32K (131K with YaRN) lets you stuff entire SEC 10-Ks into context and skip retrieval — but on a 16GB M4 Air this will swap aggressively; budget for ~20-24GB SKU if you go this route

**Search-First (web): use a free local web-search proxy.** Tavily and Brave require paid API keys; for fully-free, use **SearXNG** (self-host) + a custom LangChain `Tool` wrapper. Alternatively, **Firecrawl's open-source self-hosted** version for crawling.

### D.6 Paradigm 4a — Embedding-Free RAG (Maghakian et al., EMNLP 2025)

Reference implementation (no public code exists from the authors; this is a faithful re-implementation from Algorithm 1 in the paper):

```python
from rapidfuzz.distance import Levenshtein
from langchain_ollama import ChatOllama

QUOTE_LLM = ChatOllama(model="qwen3:4b",  temperature=0.0)   # fast extractor
ANSWER_LLM = ChatOllama(model="qwen3:14b", temperature=0.1)  # strong synthesizer

REF_QUOTE_PROMPT = """Given the question and document below, return JSON list of 
verbatim quotations from the document that contain information necessary to answer.
Question: {q}
Document: {doc}
Return: {{"quotes": [...]}}"""

def embedding_free_rag(question, document, w=5, subdoc_words=3000):
    sentences = sent_tokenize(document)
    subdocs = group_into_subdocs(sentences, max_words=subdoc_words)
    summary = generate_summary(sentences, first_n_words=5000)   # 2-3 sentences
    
    all_anchors = []
    # parallelize across subdocs
    for sub in subdocs:
        prompted = summary + "\n\n" + sub
        quotes = QUOTE_LLM.invoke(REF_QUOTE_PROMPT.format(q=question, doc=prompted))
        for r in quotes:
            # Levenshtein anchor: find closest sentence in original
            idx = min(range(len(sentences)), 
                      key=lambda i: Levenshtein.distance(r, sentences[i]))
            all_anchors.append(idx)
    
    # build chunks ± w sentences around each anchor, then merge overlaps
    chunks = merge_overlapping([(max(0,a-w), min(len(sentences),a+w+1)) 
                                for a in all_anchors])
    context = "\n".join(" ".join(sentences[s:e]) for s,e in chunks)
    return ANSWER_LLM.invoke(f"Question: {question}\nContext: {context}\nAnswer:")
```

Per the EMNLP paper, the fuzzy-matching cost is **0.005 ± 0.009 seconds per quote** and **34.64 ± 14.01 KB** of memory on a 16GB machine — so the Levenshtein step is essentially free. Use **RapidFuzz** (not `python-Levenshtein`) — it's the library cited by the authors.

The two-LLM separation (`QUOTE_LLM` fast/small + `ANSWER_LLM` strong) is **explicitly endorsed by the paper (Section 2.4.2)** and dramatically improves M4 Air latency vs. using one large model for both phases.

### D.7 Paradigm 4b — In-Memory KNN with PCA + PQ + HNSW (Khan)

```python
# pip install sentence-transformers scikit-learn hnswlib numpy faiss-cpu
import numpy as np, hnswlib
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # 384-D
emb = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True)

# 1) PCA 384 -> 256
pca = PCA(n_components=256).fit(emb)
emb_pca = pca.transform(emb).astype("float32")

# 2) Product Quantization via FAISS
import faiss
pq = faiss.IndexPQ(256, 32, 8)   # 32 subvectors, 8 bits each = 32 bytes per vec
pq.train(emb_pca); pq.add(emb_pca)

# 3) HNSW over the PCA-reduced space
index = hnswlib.Index(space="cosine", dim=256)
index.init_index(max_elements=len(emb_pca), ef_construction=200, M=16)
index.add_items(emb_pca, np.arange(len(emb_pca)))
index.set_ef(64)

def retrieve(query, k=5):
    q = pca.transform(model.encode([query])).astype("float32")
    ids, dists = index.knn_query(q, k=k)
    return [chunks[i] for i in ids[0]]
```

This is fully local, fully in-memory, and per Khan's preprint claims **60% memory reduction, 45% latency improvement, and MRR@5 = 0.87** on Natural Questions / TriviaQA for knowledge bases of **10K–1M documents**. Caveat: Khan's paper is a self-published Figshare/ResearchGate preprint without external peer review or code release — treat the headline numbers as author-reported and re-validate yourself.

### D.8 Native vs. Docker on M4

**Native everything except Elasticsearch.** Apple Silicon native wheels exist for `bm25s`, `rapidfuzz`, `hnswlib`, `faiss-cpu`, `sentence-transformers`, `ollama`, `mlx-lm`, `llama-cpp-python` (with `LLAMA_METAL=1`). Use Docker only for Elasticsearch single-node and SearXNG (if used) — both have ARM64 images.

---

## E. State-of-the-Art Innovations & Groundbreaking Features

### E.1 Hybrid Sparse → Reasoning Pipeline

The most defensible **novel contribution** for this project is a **three-stage hybrid**:

1. **bm25s top-50** (~1 ms, lexical recall)
2. **PageIndex tree-reasoning over only the chapters containing any top-50 hit** (~5 seconds, semantic precision)
3. **Embedding-Free verbatim quote extraction + Levenshtein anchoring** (~2 seconds, span precision)

No published system combines all three. This pattern is theoretically superior because each stage handles a failure mode of the next: BM25 catches keyword recall, PageIndex catches structural relevance, Embedding-Free catches paraphrase-tolerant span localization.

### E.2 Iterative Retrieval & Multi-Hop Reasoning

After the first pass, ask the LLM: *"Do you have enough information? If not, propose a follow-up retrieval query."* Loop until done or hop-budget exhausted (typically 3-5 hops). FRAMES baseline accuracy jumps from **0.408 to 0.66** with multi-step retrieval — a >50% gain. The `Search-P1` (Feb 2026), `DualRAG`, and `TreePS-RAG` lines are all variants of this idea.

### E.3 Self-Evaluation Loops (CRAG-style)

After generating an answer, the same LLM (or a smaller judge) scores faithfulness. If <threshold (e.g., 0.7), re-retrieve with a query-rewrite. This adds latency but trades off against hallucination rate; on a local M4 it's free.

### E.4 Adaptive Retrieval Router

Train a lightweight router (logistic regression on TF-IDF + query-length features, or a 0.6B Qwen3 in JSON-classification mode) on RAGRouter-Bench labels (`single_hop`, `multi_hop`, `summary`):

- **single_hop / factual** → BM25 only (cheap)
- **multi_hop / reasoning** → PageIndex or hybrid
- **summary / aggregation** → Roaming RAG with parallel section expansion
- **needs current info** → Agentic with SearXNG tool

Per the Lightweight Query Routing baseline study on RAGRouter-Bench, the best high-accuracy router achieved **28.1% token savings** while matching always-expensive baselines. This is your headline production efficiency claim.

### E.5 Latest 2025-2026 Research to Cite/Borrow

- **PageIndex** (VectifyAI, 2026) — 98.7% on FinanceBench via tree-reasoning, 30,800 GitHub stars
- **Embedding-Free RAG** (Maghakian et al., EMNLP 2025) — 2.6× F1 on LegalBench-RAG
- **Manual RAG without Vector DB** (Khan, June 2025) — PCA+PQ+HNSW, 60% memory cut
- **RAGRouter-Bench** (Wang et al., 2026) — 7,727 queries × 5 paradigms, the canonical routing benchmark
- **Adaptive-RAG** (Jeong et al., 2024) — T5-Large complexity classifier
- **Probing-RAG** / **Self-RAG** / **CRAG** — retrieval-trigger and post-retrieval-correction
- **Reasoning in Trees** (Jan 2026) — multi-hop tree reasoning for QA
- **TreePS-RAG** (Jan 2026) — process supervision via RL
- **DynaRAG** (Feb 2026) — bridging static and dynamic knowledge

### E.6 Potential Academic Contribution

The strongest publishable angle for an undergraduate or independent researcher: **"A Comparative Evaluation of Four Vectorless RAG Paradigms on Consumer Apple Silicon."** Original contributions could include:

1. **First reproducible benchmark of Maghakian's Embedding-Free RAG against PageIndex and Roaming RAG** on FRAMES + LegalBench-RAG using only open-source local LLMs (no Gemini/GPT-4).
2. **An adaptive router trained on RAGRouter-Bench labels** that selects among all four paradigms on-device.
3. **A latency-quality Pareto frontier** for the four paradigms × four model sizes (Qwen3-4B/8B/14B + Llama-3.1-8B) on a 16GB M4 MacBook Air.

A workshop paper (e.g., the EMNLP RAG workshop, or VLDB's LLM+Graph workshop where related work appeared) is realistic.

---

## F. Tooling & Development Environment

### F.1 Editor and AI Coding Setup

**Claude Code** (Anthropic's terminal-native agent) is free for limited use and pairs well with this project: it can read the entire repo, execute the Python files, and edit them iteratively. Set it up alongside one of:

- **Cursor** — strong on multi-file refactors; can be pointed at a local Ollama via "Custom Model" → OpenAI-compatible endpoint
- **VS Code + Continue.dev** — fully free, fully local; point Continue at Ollama
- **VS Code + Cline** — agentic; local-friendly

For Claude Code specifically:
1. Initialize a `CLAUDE.md` in the repo root listing the four paradigms, their files, and your conventions.
2. Use Claude Code to scaffold each paradigm's `runner.py` skeleton, then iterate.
3. Use the slash command `/cost` to track token usage; for pure-local development the cost is the inference cost, which is electricity.

### F.2 Python Environment — uv

`uv` is **the** modern Python package manager for Apple Silicon. Significantly faster than pip/conda and produces lockfiles for reproducibility.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd vectorless-rag-m4
uv init
uv python pin 3.12
uv add langchain langchain-ollama llama-index ragas deepeval bm25s rapidfuzz \
       hnswlib scikit-learn faiss-cpu sentence-transformers \
       unstructured pypdf pymupdf streamlit jupyter mlflow rich
uv sync
uv run python src/paradigms/embedding_free.py
```

### F.3 Experiment Tracking

For a solo / academic project, **MLflow local** is the right scale — it's a single `pip install mlflow` + `mlflow ui` and gives you the run grid, metric plots, and artifact storage without any cloud account. Avoid Weights & Biases unless you specifically want the team features (and even W&B Local is overkill here). For minimal needs, a CSV+`pandas.read_csv` workflow is fine.

```python
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
with mlflow.start_run(run_name="pageindex_qwen3_8b_finbench"):
    mlflow.log_params({"paradigm": "pageindex", "llm": "qwen3:8b", "k": 5})
    mlflow.log_metrics({"faithfulness": 0.81, "f1": 0.66, "p50_latency_s": 3.2})
    mlflow.log_artifact("results/runs.jsonl")
```

### F.4 Git & Reproducibility

- Commit your `uv.lock` (deterministic resolution)
- Commit your eval gold sets to `data/eval/` (or DVC for >100MB)
- Pin model versions: `ollama show qwen3:8b --modelfile > models/qwen3_8b.modelfile`
- Use `dotenv` for `OLLAMA_HOST` and any local paths
- GitHub Actions can run a smoke test on a Linux runner with `ollama pull tinyllama`; full M4 benchmarks must run locally

### F.5 Structure for Academic Publication

If you want to publish, follow this skeleton:
1. **Abstract** — One-line: "We evaluate four vectorless RAG paradigms on consumer Apple Silicon and propose an adaptive router."
2. **Introduction** — Cite Maghakian, VectifyAI, Khan, Arcturus Labs as your four paradigms.
3. **Related Work** — RAG survey (Gao et al., 2023), Agentic RAG survey (arXiv 2501.09136), BEIR (Thakur et al., 2021), FRAMES (Krishna et al., 2024).
4. **Method** — Your hybrid pipeline + adaptive router.
5. **Experiments** — FRAMES, LegalBench-RAG, FinanceBench. Report Faithfulness, F1, latency, peak RSS.
6. **Limitations** — Single-machine results; only English; M4-Air-specific bandwidth bottleneck.
7. **Code & Data Release** — Public GitHub with `uv.lock` for full reproducibility.

---

## Recommendations (Staged & Actionable)

**Week 1 — Foundation**
- Install Ollama + uv. Pull Qwen3-8B, Llama-3.1-8B, Mistral-7B.
- Reproduce the LlamaIndex local-LLM starter (Real Python / KDnuggets tutorials).
- Stand up `bm25s` over a 100-document corpus; measure baseline latency.

**Week 2 — PageIndex + Roaming RAG**
- Clone VectifyAI/PageIndex; run `agentic_vectorless_rag_demo.py` against Ollama.
- Implement Roaming RAG following Arcturus Labs' ~300-line reference, using an llms.txt-formatted corpus.

**Week 3 — Embedding-Free RAG**
- Implement Maghakian et al.'s Algorithm 1 verbatim. Verify on 50 LegalBench-RAG PrivacyQA questions.
- Two-LLM split: Qwen3-4B extractor + Qwen3-14B synthesizer (or 8B if 16GB).

**Week 4 — In-Memory KNN + Evaluation Harness**
- Build the PCA+PQ+HNSW pipeline per Khan's recipe; validate on Natural Questions subset.
- Stand up RAGAS-via-Ollama + DeepEval. Run all four paradigms on a 100-question slice.

**Week 5 — Adaptive Router + Hybrid Pipeline**
- Train a logistic-regression router on RAGRouter-Bench labels.
- Build the BM25 → PageIndex → Embedding-Free three-stage hybrid.

**Week 6 — Benchmark + Writeup**
- Full evaluation on FRAMES + LegalBench-RAG + FinanceBench.
- Produce the Pareto frontier figure (latency × F1 × peak RSS).

**Thresholds that change the plan:**
- If 16GB M4 Air swaps frequently during Qwen3-8B inference → drop to Qwen3-4B or Llama-3.2-3B and accept lower F1.
- If Embedding-Free RAG matches or beats PageIndex on your corpus → skip the hybrid and ship Embedding-Free as your single paradigm.
- If router accuracy is <70% → fall back to the hybrid pipeline (which is the union of paradigms, not the choice among them).

---

## Caveats

- **Khan (Manual RAG)** is a self-published preprint without peer review or public code; replicate the 60%/45%/MRR=0.87 numbers yourself before citing them as established. The paper's stated scale ("10K–1M documents") may not extrapolate to your corpus size.
- **VectifyAI's 98.7% FinanceBench number** is reported on Mafin 2.5, which uses GPT-4o by default for tree-reasoning. Local Qwen3-8B/14B will almost certainly score lower; expect ~15-25 point drops from frontier-model accuracy on hard benchmarks, consistent with Maghakian's ablation showing Qwen2.5-14B at F1=0.28 vs. Llama-3.1-70B at F1=0.27 on PrivacyQA.
- **PageIndex's "vectorless"** claim is partially marketing. Production-grade vectorless RAG still benefits from BM25 or even a lightweight vector pre-filter before tree reasoning, as the Medium "Production Guide" article explicitly notes ("Even in 'vectorless' systems, use vectors only for pruning").
- **Ollama 0.19's MLX backend requires 32GB+** unified memory. On a 16GB M4 Air it will silently fall back to llama.cpp. Use LM Studio's MLX for the 16GB SKU instead.
- **The Embedding-Free RAG paper does not release public code**; the implementation in Section D.6 above is a faithful reconstruction from the paper's Algorithm 1, prompts in Appendix A.1/A.2, and the explicit use of RapidFuzz. Validate your reconstruction by reproducing the paper's MAUD numbers (recall=0.66, precision=0.08) before extending.
- **Speculation markers**: claims about "M5" performance gains and Ollama-on-MLX speedups for 70B-class models are from vendor blog posts (ollama.com/blog/mlx, dated March 30, 2026) and community benchmark sites (llmcheck.net, willitrunai.com); they are not peer-reviewed and the underlying tok/s figures depend heavily on prompt length, batch size, and macOS version. The 58→112 tok/s number specifically was measured on M5 Max, not M4. Reproduce on your own hardware.
- **Brave Search and Tavily are paid APIs** despite some sources framing them as free; for the "fully local, zero-cost" constraint, self-host **SearXNG** as your web-search tool.
- **Costs noted as "$0"** exclude electricity (~20W sustained on M4 Air × your local power rate × hours) and the one-time cost of the MacBook itself.