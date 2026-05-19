"""
Pipeline 5: Hybrid Vectorless RAG (OUR SoTA SYSTEM)
────────────────────────────────────────────────────
Novel hybrid that combines the best of all paradigms:
  1. Adaptive query routing (LLM classifies query type)
  2. Hierarchical tree navigation (for structured queries)
  3. BM25 lexical search (for keyword/exact queries)
  4. Reciprocal Rank Fusion (for complex multi-hop queries)
  5. Agentic verification loop (for self-correction)
"""

from __future__ import annotations


import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.pipelines.base import (
    RAGPipeline, RAGResponse, IngestionReport, UpdateReport,
)
from src.pipelines.pageindex_rag import PageIndexRAG
from src.pipelines.bm25_rag import BM25RAG, InMemoryBM25
from src.utils.llm_client import LLMClient


ROUTER_PROMPT = """Classify this query into exactly one retrieval strategy.

QUERY: {question}

Categories:
- STRUCTURED: Answer is in a specific section/table of a well-organized document
  (e.g., "What is the revenue in Q3?" → needs tree navigation)
- KEYWORD: Query has specific terms, codes, or exact phrases needing precise match
  (e.g., "What does clause 4.2.1 say?" → needs keyword search)
- COMPLEX: Requires multi-hop reasoning across multiple sections or documents
  (e.g., "Compare X in doc A vs Y in doc B" → needs both strategies)

OUTPUT (JSON):
{{"category": "STRUCTURED|KEYWORD|COMPLEX", "reasoning": "brief explanation"}}"""

VERIFY_PROMPT = """You are a verification agent. Check if the proposed answer
adequately addresses the question based on the evidence.

QUESTION: {question}
PROPOSED ANSWER: {answer}
EVIDENCE USED:
{evidence}

OUTPUT (JSON):
{{
  "verdict": "PASS|FAIL|PARTIAL",
  "issues": ["issue1", ...],
  "suggestion": "what to search for if FAIL"
}}"""

ANSWER_PROMPT = """Answer the question using ONLY the provided context.
Cite specific sources (document, section, page) for every claim.
If context is insufficient, say so clearly.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""


class HybridSoTARAG(RAGPipeline):
    """
    Our novel Hybrid Vectorless RAG system.

    Query flow:
      User Query → Router → {STRUCTURED → Tree Nav}
                            {KEYWORD → BM25 Search}
                            {COMPLEX → Both → RRF Fusion}
                  → Verifier → (retry if FAIL) → Generator → Answer
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})
        router_cfg = pipeline_cfg.get("router", {})
        tree_cfg = pipeline_cfg.get("tree", {})
        bm25_cfg = pipeline_cfg.get("bm25", {})
        fusion_cfg = pipeline_cfg.get("fusion", {})
        verifier_cfg = pipeline_cfg.get("verifier", {})
        gen_cfg = pipeline_cfg.get("generation", {})

        # Router
        router_model = router_cfg.get("model", "gpt-4o")
        self._router = LLMClient(model=router_model, temperature=0.0)

        # Tree component (reuse PageIndex internals)
        self._tree_client = LLMClient(
            model=tree_cfg.get("model", "gpt-4o"), temperature=0.0,
        )
        self._trees: dict[str, dict] = {}
        self._documents: dict[str, dict] = {}

        # BM25 component
        self._bm25 = InMemoryBM25()
        self._chunk_size = bm25_cfg.get("chunk_size", 512)
        self._chunk_overlap = bm25_cfg.get("chunk_overlap", 64)
        self._bm25_top_k = bm25_cfg.get("top_k", 5)

        # Fusion config
        self._fusion_strategy = fusion_cfg.get("strategy", "reciprocal_rank")
        self._tree_weight = fusion_cfg.get("tree_weight", 0.6)
        self._bm25_weight = fusion_cfg.get("bm25_weight", 0.4)
        self._max_combined = fusion_cfg.get("max_combined_contexts", 8)

        # Verifier
        self._verifier = LLMClient(
            model=verifier_cfg.get("model", "gpt-4o"), temperature=0.0,
        )
        self._max_retries = verifier_cfg.get("max_retries", 2)

        # Generator
        self._generator = LLMClient(
            model=gen_cfg.get("model", "gpt-4o"), temperature=0.0,
        )
        self._system_prompt = gen_cfg.get("system_prompt", "")

        self._index_dir: str | None = None

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Build BOTH tree index AND BM25 index for adaptive retrieval."""
        start = time.perf_counter()
        corpus_dir = Path(corpus_path)
        self._bm25 = InMemoryBM25()

        manifest_path = corpus_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            doc_files = [corpus_dir / f"{d['doc_id']}.json" for d in manifest["documents"]]
        else:
            doc_files = sorted(corpus_dir.glob("*.json"))

        doc_count = 0
        chunk_count = 0

        for doc_file in doc_files:
            if doc_file.name == "manifest.json":
                continue
            with open(doc_file) as f:
                doc_data = json.load(f)

            doc_id = doc_data.get("doc_id", doc_file.stem)
            self._documents[doc_id] = doc_data
            full_text = doc_data.get("full_text", "")
            title = doc_data.get("title", doc_id)

            # Build tree index
            from src.pipelines.pageindex_rag import TREE_BUILD_PROMPT
            content_preview = full_text[:8000]
            prompt = TREE_BUILD_PROMPT.format(doc_title=title, content=content_preview)
            resp = self._tree_client.generate(prompt, json_mode=True)
            try:
                self._trees[doc_id] = json.loads(resp.content)
            except json.JSONDecodeError:
                self._trees[doc_id] = {"title": title, "children": []}

            # Build BM25 index (chunk the same document)
            words = full_text.split()
            step = max(1, self._chunk_size - self._chunk_overlap)
            for i in range(0, len(words), step):
                chunk_words = words[i:i + self._chunk_size]
                if chunk_words:
                    chunk_text = " ".join(chunk_words)
                    self._bm25.add_document(
                        f"{doc_id}_c{chunk_count}",
                        chunk_text,
                        {"source_doc": doc_id, "title": title},
                    )
                    chunk_count += 1

            doc_count += 1
            logger.debug(f"Hybrid indexed {doc_id}: tree + {chunk_count} BM25 chunks")

        elapsed = time.perf_counter() - start

        # Save index
        self._index_dir = str(corpus_dir / ".hybrid_index")
        Path(self._index_dir).mkdir(exist_ok=True)
        index_size = 0
        for doc_id, tree in self._trees.items():
            tp = Path(self._index_dir) / f"{doc_id}_tree.json"
            with open(tp, "w") as f:
                json.dump(tree, f)
            index_size += tp.stat().st_size

        self._is_ingested = True
        logger.info(
            f"Hybrid SoTA ingestion: {doc_count} docs, "
            f"{len(self._trees)} trees + {chunk_count} BM25 chunks "
            f"in {elapsed:.1f}s"
        )

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=doc_count,
            ingestion_time_seconds=elapsed,
            index_size_bytes=index_size,
            index_artifacts={"index_dir": self._index_dir},
            metadata={"chunk_count": chunk_count, "tree_count": len(self._trees)},
        )

    def query(self, question: str) -> RAGResponse:
        """Adaptive routing → retrieval → verification → generation."""
        start = time.perf_counter()
        total_input = 0
        total_output = 0
        total_cost = 0.0

        # === Step 1: Route the query ===
        route_resp = self._router.generate(
            ROUTER_PROMPT.format(question=question), json_mode=True,
        )
        total_input += route_resp.input_tokens
        total_output += route_resp.output_tokens
        total_cost += route_resp.cost_usd

        try:
            route_result = json.loads(route_resp.content)
            category = route_result.get("category", "COMPLEX").upper()
        except (json.JSONDecodeError, KeyError):
            category = "COMPLEX"

        logger.debug(f"Query routed to: {category}")

        # === Step 2: Retrieve based on category ===
        contexts = []
        references = []

        if category == "STRUCTURED":
            ctx, refs, tokens, cost = self._tree_retrieve(question)
            contexts = ctx
            references = refs
        elif category == "KEYWORD":
            ctx, refs = self._bm25_retrieve(question)
            contexts = ctx
            references = refs
            tokens = {"input": 0, "output": 0}
            cost = 0.0
        else:  # COMPLEX — use both + fusion
            tree_ctx, tree_refs, tree_tokens, tree_cost = self._tree_retrieve(question)
            bm25_ctx, bm25_refs = self._bm25_retrieve(question)
            contexts, references = self._fuse_results(
                tree_ctx, tree_refs, bm25_ctx, bm25_refs,
            )
            tokens = tree_tokens
            cost = tree_cost

        total_input += tokens.get("input", 0)
        total_output += tokens.get("output", 0)
        total_cost += cost

        # === Step 3: Generate initial answer ===
        context_text = "\n\n---\n\n".join(contexts[:self._max_combined])
        gen_prompt = ANSWER_PROMPT.format(question=question, context=context_text)
        gen_resp = self._generator.generate(gen_prompt, system_prompt=self._system_prompt)
        total_input += gen_resp.input_tokens
        total_output += gen_resp.output_tokens
        total_cost += gen_resp.cost_usd

        answer = gen_resp.content

        # === Step 4: Verification loop ===
        for retry in range(self._max_retries):
            verify_resp = self._verifier.generate(
                VERIFY_PROMPT.format(
                    question=question,
                    answer=answer,
                    evidence=context_text[:3000],
                ),
                json_mode=True,
            )
            total_input += verify_resp.input_tokens
            total_output += verify_resp.output_tokens
            total_cost += verify_resp.cost_usd

            try:
                verdict = json.loads(verify_resp.content)
                if verdict.get("verdict") == "PASS":
                    break
                elif verdict.get("verdict") == "FAIL" and retry < self._max_retries - 1:
                    # Expand search: try the other strategy
                    suggestion = verdict.get("suggestion", question)
                    logger.debug(f"Verification FAIL, retry {retry+1}: {suggestion}")
                    if category != "KEYWORD":
                        extra_ctx, extra_refs = self._bm25_retrieve(suggestion)
                        contexts.extend(extra_ctx)
                        references.extend(extra_refs)
                    else:
                        extra_ctx, extra_refs, t, c = self._tree_retrieve(suggestion)
                        contexts.extend(extra_ctx)
                        references.extend(extra_refs)
                        total_input += t.get("input", 0)
                        total_output += t.get("output", 0)
                        total_cost += c

                    # Re-generate with expanded context
                    context_text = "\n\n---\n\n".join(contexts[:self._max_combined])
                    gen_resp = self._generator.generate(
                        ANSWER_PROMPT.format(question=question, context=context_text),
                        system_prompt=self._system_prompt,
                    )
                    total_input += gen_resp.input_tokens
                    total_output += gen_resp.output_tokens
                    total_cost += gen_resp.cost_usd
                    answer = gen_resp.content
                else:
                    break
            except (json.JSONDecodeError, KeyError):
                break

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            answer=answer,
            retrieved_contexts=contexts,
            source_references=references,
            latency_ms=elapsed_ms,
            tokens_used={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            cost_usd=total_cost,
            metadata={"route_category": category},
        )

    def add_documents(self, document_paths: list[str]) -> UpdateReport:
        """Add to both tree and BM25 indices — no full reindex."""
        start = time.perf_counter()

        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            self._documents[doc_id] = doc_data
            full_text = doc_data.get("full_text", "")
            title = doc_data.get("title", "")

            # Add tree
            from src.pipelines.pageindex_rag import TREE_BUILD_PROMPT
            prompt = TREE_BUILD_PROMPT.format(doc_title=title, content=full_text[:8000])
            resp = self._tree_client.generate(prompt, json_mode=True)
            try:
                self._trees[doc_id] = json.loads(resp.content)
            except json.JSONDecodeError:
                self._trees[doc_id] = {"title": title, "children": []}

            # Add BM25 chunks
            words = full_text.split()
            step = max(1, self._chunk_size - self._chunk_overlap)
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + self._chunk_size])
                if chunk:
                    self._bm25.add_document(
                        f"{doc_id}_c{i}", chunk,
                        {"source_doc": doc_id, "title": title},
                    )

        elapsed = time.perf_counter() - start
        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=False,
            new_index_size_bytes=0,
        )

    # ── Private retrieval methods ──────────────────────────────────

    def _tree_retrieve(self, question: str):
        """Retrieve using hierarchical tree navigation."""
        from src.pipelines.pageindex_rag import TREE_NAVIGATE_PROMPT

        contexts = []
        references = []
        total_input = 0
        total_output = 0
        total_cost = 0.0

        for doc_id, tree in self._trees.items():
            tree_json = json.dumps(tree, indent=1)[:4000]
            resp = self._tree_client.generate(
                TREE_NAVIGATE_PROMPT.format(question=question, tree_json=tree_json),
                json_mode=True,
            )
            total_input += resp.input_tokens
            total_output += resp.output_tokens
            total_cost += resp.cost_usd

            try:
                result = json.loads(resp.content)
                selected = result.get("selected_sections", [])
            except (json.JSONDecodeError, KeyError):
                continue

            doc_data = self._documents.get(doc_id, {})
            full_text = doc_data.get("full_text", "")
            sections = doc_data.get("sections", [])

            for sel in selected:
                target = sel.get("title", "").lower()
                content = ""
                for sec in sections:
                    s = sec if isinstance(sec, dict) else {}
                    if target in s.get("title", "").lower():
                        content = s.get("content", "")[:2000]
                        break
                if not content and full_text:
                    content = full_text[:2000]

                if content:
                    contexts.append(content)
                    references.append({
                        "doc_id": doc_id,
                        "title": sel.get("title", ""),
                        "method": "tree",
                    })

        tokens = {"input": total_input, "output": total_output}
        return contexts, references, tokens, total_cost

    def _bm25_retrieve(self, question: str):
        """Retrieve using BM25 keyword search."""
        results = self._bm25.search(question, top_k=self._bm25_top_k)
        contexts = [r["doc"]["text"] for r in results]
        references = [
            {
                "chunk_id": r["doc"]["id"],
                "title": r["doc"]["metadata"].get("title", ""),
                "bm25_score": round(r["score"], 4),
                "method": "bm25",
            }
            for r in results
        ]
        return contexts, references

    def _fuse_results(self, tree_ctx, tree_refs, bm25_ctx, bm25_refs):
        """Reciprocal Rank Fusion of tree and BM25 results."""
        k = 60  # RRF constant
        scored = {}

        # Score tree results
        for i, (ctx, ref) in enumerate(zip(tree_ctx, tree_refs)):
            key = ctx[:100]  # Use content prefix as key
            rrf_score = self._tree_weight / (k + i + 1)
            scored[key] = {"ctx": ctx, "ref": ref, "score": rrf_score}

        # Score BM25 results
        for i, (ctx, ref) in enumerate(zip(bm25_ctx, bm25_refs)):
            key = ctx[:100]
            rrf_score = self._bm25_weight / (k + i + 1)
            if key in scored:
                scored[key]["score"] += rrf_score
                scored[key]["ref"]["method"] = "fused"
            else:
                scored[key] = {"ctx": ctx, "ref": ref, "score": rrf_score}

        # Sort by fused score
        ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
        ranked = ranked[:self._max_combined]

        return (
            [r["ctx"] for r in ranked],
            [r["ref"] for r in ranked],
        )
