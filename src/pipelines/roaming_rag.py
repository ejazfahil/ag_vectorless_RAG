"""
Pipeline 2: Roaming RAG (Agentic Navigation)
─────────────────────────────────────────────
The LLM acts as an agent that "roams" through a document's structure.
It starts with an outline view, then selectively expands sections
using tool calls until it finds the information needed.

Reference: Arcturus Labs / TheUnwindAI (2025)
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
from src.utils.llm_client import LLMClient


# ── Prompts ─────────────────────────────────────────────────────────

OUTLINE_PROMPT = """You are a research agent navigating a structured document.

QUESTION: {question}

Here is the document outline (table of contents). Each section has an ID
you can use to expand it and read its full content.

OUTLINE:
{outline}

INSTRUCTIONS:
1. Read the outline carefully
2. Identify which sections are MOST LIKELY to contain the answer
3. Select up to {max_sections} sections to expand (read in full)
4. Explain your reasoning

OUTPUT (JSON):
{{
  "reasoning": "Why you chose these sections",
  "sections_to_expand": ["section_id_1", "section_id_2", ...]
}}"""

REFINE_PROMPT = """You are a research agent. You've expanded some sections but
haven't found a complete answer yet.

QUESTION: {question}

SECTIONS ALREADY READ:
{read_sections}

REMAINING OUTLINE:
{remaining_outline}

Should you expand more sections? If yes, which ones?

OUTPUT (JSON):
{{
  "found_answer": true/false,
  "sections_to_expand": ["section_id", ...],
  "partial_answer": "what you know so far"
}}"""

ANSWER_PROMPT = """Answer the question using ONLY the expanded sections below.
Cite section IDs for each claim. If insufficient, say so.

QUESTION: {question}

EXPANDED SECTIONS:
{sections_content}

ANSWER:"""


class RoamingRAG(RAGPipeline):
    """
    Roaming RAG: The LLM agent navigates document structure iteratively.

    Ingestion: Parse documents into sections with unique IDs
    Retrieval: Agent views outline → expands sections → refines until done
    Generation: Answer from expanded sections
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})
        ret_cfg = pipeline_cfg.get("retrieval", {})
        gen_cfg = pipeline_cfg.get("generation", {})

        self.model = ret_cfg.get("model", "gpt-4o")
        self.max_steps = ret_cfg.get("max_roaming_steps", 8)
        self.gen_model = gen_cfg.get("model", "gpt-4o")
        self.system_prompt = gen_cfg.get("system_prompt", "")

        self._nav_client = LLMClient(model=self.model, temperature=0.0)
        self._gen_client = LLMClient(model=self.gen_model, temperature=0.0)

        # State: doc_id -> {outline: str, sections: {id: {title, content}}}
        self._documents: dict[str, dict] = {}
        self._index_dir: str | None = None

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Parse documents into hierarchical sections with IDs."""
        start = time.perf_counter()
        corpus_dir = Path(corpus_path)

        manifest_path = corpus_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            doc_files = [corpus_dir / f"{d['doc_id']}.json" for d in manifest["documents"]]
        else:
            doc_files = sorted(corpus_dir.glob("*.json"))

        for doc_file in doc_files:
            if doc_file.name == "manifest.json":
                continue
            with open(doc_file) as f:
                doc_data = json.load(f)

            doc_id = doc_data.get("doc_id", doc_file.stem)

            # Build section map with IDs
            sections_map = {}
            outline_lines = []
            raw_sections = doc_data.get("sections", [])

            for i, sec in enumerate(raw_sections):
                sec_data = sec if isinstance(sec, dict) else {"id": str(i+1), "title": f"Section {i+1}", "content": str(sec)}
                sec_id = sec_data.get("id", str(i + 1))
                level = sec_data.get("level", 1)
                title = sec_data.get("title", f"Section {sec_id}")
                content = sec_data.get("content", "")

                sections_map[sec_id] = {
                    "title": title,
                    "content": content,
                    "level": level,
                    "word_count": len(content.split()),
                }

                indent = "  " * max(0, level - 1)
                outline_lines.append(
                    f"{indent}[{sec_id}] {title} ({len(content.split())} words)"
                )

            self._documents[doc_id] = {
                "title": doc_data.get("title", doc_id),
                "outline": "\n".join(outline_lines),
                "sections": sections_map,
            }

            logger.debug(f"Indexed {doc_id}: {len(sections_map)} sections")

        elapsed = time.perf_counter() - start

        # Save index
        self._index_dir = str(corpus_dir / ".roaming_index")
        Path(self._index_dir).mkdir(exist_ok=True)
        index_size = 0
        for doc_id, doc_info in self._documents.items():
            idx_path = Path(self._index_dir) / f"{doc_id}_index.json"
            with open(idx_path, "w") as f:
                json.dump({
                    "title": doc_info["title"],
                    "outline": doc_info["outline"],
                    "section_ids": list(doc_info["sections"].keys()),
                }, f, indent=2)
            index_size += idx_path.stat().st_size

        self._is_ingested = True
        logger.info(f"Roaming RAG ingestion: {len(self._documents)} docs in {elapsed:.1f}s")

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=len(self._documents),
            ingestion_time_seconds=elapsed,
            index_size_bytes=index_size,
            index_artifacts={"index_dir": self._index_dir},
        )

    def query(self, question: str) -> RAGResponse:
        """Agent roams through documents, expanding sections iteratively."""
        start = time.perf_counter()
        total_input = 0
        total_output = 0
        total_cost = 0.0

        all_contexts = []
        all_references = []

        for doc_id, doc_info in self._documents.items():
            expanded_sections = {}
            remaining_ids = set(doc_info["sections"].keys())

            # Step 1: View outline and select initial sections
            prompt = OUTLINE_PROMPT.format(
                question=question,
                outline=doc_info["outline"],
                max_sections=min(4, len(remaining_ids)),
            )
            resp = self._nav_client.generate(prompt, json_mode=True)
            total_input += resp.input_tokens
            total_output += resp.output_tokens
            total_cost += resp.cost_usd

            try:
                result = json.loads(resp.content)
                to_expand = result.get("sections_to_expand", [])
            except (json.JSONDecodeError, KeyError):
                to_expand = list(remaining_ids)[:3]

            # Expand selected sections
            for sec_id in to_expand:
                sec_id = str(sec_id)
                if sec_id in doc_info["sections"]:
                    sec = doc_info["sections"][sec_id]
                    expanded_sections[sec_id] = sec
                    remaining_ids.discard(sec_id)

            # Step 2: Iterative refinement (up to max_steps)
            for step in range(self.max_steps - 1):
                if not remaining_ids:
                    break

                read_text = "\n".join(
                    f"[{sid}] {s['title']}:\n{s['content'][:500]}"
                    for sid, s in expanded_sections.items()
                )
                remaining_outline = "\n".join(
                    f"[{sid}] {doc_info['sections'][sid]['title']}"
                    for sid in remaining_ids
                    if sid in doc_info["sections"]
                )

                refine_prompt = REFINE_PROMPT.format(
                    question=question,
                    read_sections=read_text,
                    remaining_outline=remaining_outline,
                )
                resp = self._nav_client.generate(refine_prompt, json_mode=True)
                total_input += resp.input_tokens
                total_output += resp.output_tokens
                total_cost += resp.cost_usd

                try:
                    result = json.loads(resp.content)
                    if result.get("found_answer", False):
                        break
                    more = result.get("sections_to_expand", [])
                except (json.JSONDecodeError, KeyError):
                    break

                for sec_id in more:
                    sec_id = str(sec_id)
                    if sec_id in doc_info["sections"]:
                        expanded_sections[sec_id] = doc_info["sections"][sec_id]
                        remaining_ids.discard(sec_id)

            # Collect contexts from this document
            for sec_id, sec in expanded_sections.items():
                all_contexts.append(f"[{sec_id}] {sec['title']}:\n{sec['content']}")
                all_references.append({
                    "doc_id": doc_id,
                    "section_id": sec_id,
                    "title": sec["title"],
                })

        # Step 3: Generate answer
        sections_text = "\n\n---\n\n".join(all_contexts[:6])
        gen_prompt = ANSWER_PROMPT.format(
            question=question, sections_content=sections_text,
        )
        gen_resp = self._gen_client.generate(gen_prompt, system_prompt=self.system_prompt)
        total_input += gen_resp.input_tokens
        total_output += gen_resp.output_tokens
        total_cost += gen_resp.cost_usd

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
            cost_usd=total_cost,
        )

    def add_documents(self, document_paths: list[str]) -> UpdateReport:
        """Add new documents — parse sections, no reindex needed."""
        start = time.perf_counter()

        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            sections_map = {}
            outline_lines = []
            for i, sec in enumerate(doc_data.get("sections", [])):
                sec_data = sec if isinstance(sec, dict) else {"id": str(i+1), "content": str(sec)}
                sec_id = sec_data.get("id", str(i + 1))
                sections_map[sec_id] = {
                    "title": sec_data.get("title", f"Section {sec_id}"),
                    "content": sec_data.get("content", ""),
                    "level": sec_data.get("level", 1),
                }
                outline_lines.append(f"[{sec_id}] {sections_map[sec_id]['title']}")

            self._documents[doc_id] = {
                "title": doc_data.get("title", doc_id),
                "outline": "\n".join(outline_lines),
                "sections": sections_map,
            }

        elapsed = time.perf_counter() - start
        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=False,
            new_index_size_bytes=0,
        )
