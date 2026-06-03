"""
Pipeline 1: PageIndex RAG (VectifyAI)
─────────────────────────────────────
Hierarchical tree-based vectorless RAG. Builds a JSON tree index
from documents, then uses LLM reasoning to navigate the tree and
retrieve exact page ranges for answer generation.

Reference: https://github.com/VectifyAI/PageIndex
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.pipelines.base import (
    IngestionReport,
    RAGPipeline,
    RAGResponse,
    UpdateReport,
)
from src.utils.llm_client import LLMClient

# ── Prompts ─────────────────────────────────────────────────────────

TREE_BUILD_PROMPT = """You are a document indexing system. Given the following document content,
create a hierarchical JSON tree that acts as an intelligent table of contents.

Each node must have:
- "title": short descriptive title
- "summary": 1-2 sentence summary of what this section covers
- "page_range": [start_page, end_page]
- "content_preview": first 200 characters of the section
- "children": array of child nodes (recursive)

Build a tree with 2-4 levels of depth. Group content logically by topic.

DOCUMENT TITLE: {doc_title}
DOCUMENT CONTENT (truncated):
{content}

OUTPUT: Return ONLY valid JSON representing the tree."""

TREE_NAVIGATE_PROMPT = """You are a research assistant navigating a document's hierarchical index.

QUESTION: {question}

DOCUMENT INDEX TREE:
{tree_json}

Your task:
1. Reason about which branches of the tree are most likely to contain the answer
2. Select the most relevant leaf nodes (sections) that should be retrieved
3. Explain your reasoning

OUTPUT (JSON):
{{
  "reasoning": "step-by-step explanation of your navigation",
  "selected_sections": [
    {{"title": "...", "page_range": [start, end], "relevance": "why this section is relevant"}}
  ]
}}"""

ANSWER_PROMPT = """Answer the following question using ONLY the provided context.
If the context is insufficient, say so explicitly.
Cite specific sections and page numbers.

QUESTION: {question}

CONTEXT:
{context}

ANSWER:"""


class PageIndexRAG(RAGPipeline):
    """
    PageIndex-style hierarchical tree RAG.

    Ingestion: Parse documents → build hierarchical JSON tree per document
    Retrieval: LLM reasons through tree → selects relevant sections
    Generation: LLM answers using selected section content
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})
        gen_cfg = pipeline_cfg.get("generation", {})
        ret_cfg = pipeline_cfg.get("retrieval", {})

        self.tree_model = pipeline_cfg.get("ingestion", {}).get("model", "gpt-4o")
        self.retrieval_model = ret_cfg.get("model", "gpt-4o")
        self.generation_model = gen_cfg.get("model", "gpt-4o")
        self.system_prompt = gen_cfg.get("system_prompt", "")
        self.max_tree_depth = ret_cfg.get("max_tree_depth", 5)
        self.max_retrieved_pages = ret_cfg.get("max_retrieved_pages", 10)

        self._tree_client = LLMClient(model=self.tree_model, temperature=0.0)
        self._nav_client = LLMClient(model=self.retrieval_model, temperature=0.0)
        self._gen_client = LLMClient(model=self.generation_model, temperature=0.0)

        # State
        self._trees: dict[str, dict] = {}      # doc_id -> tree JSON
        self._documents: dict[str, dict] = {}   # doc_id -> full doc data
        self._index_dir: str | None = None

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Build hierarchical tree index for each document."""
        start = time.perf_counter()
        corpus_dir = Path(corpus_path)
        self._index_dir = str(corpus_dir / ".pageindex_trees")
        os.makedirs(self._index_dir, exist_ok=True)

        # Load processed documents
        manifest_path = corpus_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            doc_files = [corpus_dir / f"{d['doc_id']}.json" for d in manifest["documents"]]
        else:
            doc_files = sorted(corpus_dir.glob("*.json"))

        ingestion_cost = 0.0
        index_size = 0
        for doc_file in doc_files:
            if doc_file.name == "manifest.json":
                continue
            with open(doc_file) as f:
                doc_data = json.load(f)

            doc_id = doc_data.get("doc_id", doc_file.stem)
            self._documents[doc_id] = doc_data

            # Check if saved tree exists
            tree_path = os.path.join(self._index_dir, f"{doc_id}_tree.json")
            if os.path.exists(tree_path):
                logger.debug(f"Loading cached PageIndex tree for {doc_id} from {tree_path}...")
                with open(tree_path) as tf:
                    tree = json.load(tf)
            else:
                # Build tree using LLM
                content = doc_data.get("full_text", "")[:8000]  # Truncate for API
                tree = self._build_tree(doc_data.get("title", doc_file.stem), content)
                with open(tree_path, "w") as tf:
                    json.dump(tree, tf, indent=2)
                logger.debug(f"Built tree for {doc_id} and saved to {tree_path}: {len(json.dumps(tree))} chars")

            self._trees[doc_id] = tree
            index_size += os.path.getsize(tree_path)
            ingestion_cost += self._tree_client.cumulative_cost

        elapsed = time.perf_counter() - start
        self._is_ingested = True
        logger.info(f"PageIndex ingestion: {len(self._trees)} trees built in {elapsed:.1f}s")

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=len(self._trees),
            ingestion_time_seconds=elapsed,
            index_size_bytes=index_size,
            index_artifacts={"trees_dir": self._index_dir},
            metadata={"ingestion_cost_usd": ingestion_cost},
        )

    def query(self, question: str) -> RAGResponse:
        """Navigate trees to find relevant sections, then generate answer."""
        start = time.perf_counter()
        total_input = 0
        total_output = 0

        # Step 1: Navigate each document's tree
        all_contexts = []
        all_references = []

        for doc_id, tree in self._trees.items():
            tree_json = json.dumps(tree, indent=1)[:4000]
            nav_prompt = TREE_NAVIGATE_PROMPT.format(
                question=question, tree_json=tree_json,
            )
            nav_resp = self._nav_client.generate(nav_prompt, json_mode=True)
            total_input += nav_resp.input_tokens
            total_output += nav_resp.output_tokens

            try:
                nav_result = json.loads(nav_resp.content)
                selected = nav_result.get("selected_sections", [])
            except (json.JSONDecodeError, KeyError):
                selected = []

            # Step 2: Fetch content for selected sections
            doc_data = self._documents.get(doc_id, {})
            full_text = doc_data.get("full_text", "")
            sections = doc_data.get("sections", [])

            for sel in selected:
                # Match selected section to actual content
                section_content = self._fetch_section_content(
                    sel, sections, full_text
                )
                if section_content:
                    all_contexts.append(section_content)
                    all_references.append({
                        "doc_id": doc_id,
                        "title": sel.get("title", ""),
                        "page_range": sel.get("page_range", []),
                        "relevance": sel.get("relevance", ""),
                    })

        # Step 3: Generate answer
        context_text = "\n\n---\n\n".join(all_contexts[:5])  # Top 5 contexts
        gen_prompt = ANSWER_PROMPT.format(question=question, context=context_text)
        gen_resp = self._gen_client.generate(gen_prompt, system_prompt=self.system_prompt)
        total_input += gen_resp.input_tokens
        total_output += gen_resp.output_tokens

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            answer=gen_resp.content,
            retrieved_contexts=all_contexts,
            source_references=all_references,
            latency_ms=elapsed_ms,
            tokens_used={
                "input": total_input,
                "output": total_output,
                "total": total_input + total_output,
            },
            cost_usd=nav_resp.cost_usd + gen_resp.cost_usd,
        )

    def add_documents(self, document_paths: list[str]) -> UpdateReport:
        """Add new documents — just build trees for them (no reindex needed)."""
        start = time.perf_counter()

        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            self._documents[doc_id] = doc_data

            content = doc_data.get("full_text", "")[:8000]
            tree = self._build_tree(doc_data.get("title", ""), content)
            self._trees[doc_id] = tree

            if self._index_dir:
                tree_path = os.path.join(self._index_dir, f"{doc_id}_tree.json")
                with open(tree_path, "w") as f:
                    json.dump(tree, f, indent=2)

        elapsed = time.perf_counter() - start

        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=False,  # Key advantage of PageIndex
            new_index_size_bytes=0,
        )

    # ── Private helpers ─────────────────────────────────────────────

    def _build_tree(self, doc_title: str, content: str) -> dict:
        """Use LLM to build hierarchical tree from document content."""
        prompt = TREE_BUILD_PROMPT.format(doc_title=doc_title, content=content)
        resp = self._tree_client.generate(prompt, json_mode=True)
        try:
            return json.loads(resp.content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tree JSON for {doc_title}")
            return {"title": doc_title, "summary": "Parse failed", "children": []}

    def _fetch_section_content(
        self, selected: dict, sections: list, full_text: str
    ) -> str:
        """Fetch actual content for a selected section from the document."""
        target_title = selected.get("title", "").lower()

        # Try matching by title
        for section in sections:
            sec_data = section if isinstance(section, dict) else {}
            if target_title in sec_data.get("title", "").lower():
                return sec_data.get("content", "")[:2000]

        # Fallback: extract by page range approximation
        page_range = selected.get("page_range", [])
        if page_range and len(page_range) == 2 and full_text:
            total_chars = len(full_text)
            start_frac = max(0, (page_range[0] - 1)) / max(1, page_range[1])
            start_idx = int(start_frac * total_chars)
            chunk_size = min(2000, total_chars // max(1, page_range[1]))
            return full_text[start_idx:start_idx + chunk_size]

        return ""
