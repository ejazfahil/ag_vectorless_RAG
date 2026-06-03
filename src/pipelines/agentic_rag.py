"""
Pipeline 4: Agentic Search-First RAG
─────────────────────────────────────
Multi-agent pipeline where specialized agents decompose, search,
synthesize, and verify — using massive context windows instead of
vector databases. "RAG without the R."
"""

from __future__ import annotations

import json
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

DECOMPOSE_PROMPT = """Break this question into simpler sub-questions that can be
answered independently by searching through documents.

QUESTION: {question}

OUTPUT (JSON):
{{
  "sub_questions": ["sub_q1", "sub_q2", ...],
  "reasoning": "why you decomposed it this way"
}}

If the question is simple enough, return just the original question."""

SEARCH_PROMPT = """You are a search agent. Find information to answer this question
by reading the provided document content.

QUESTION: {question}

DOCUMENT: {doc_title}
CONTENT:
{content}

Extract ALL relevant information. Be thorough.

OUTPUT (JSON):
{{
  "relevant_passages": ["passage 1...", "passage 2..."],
  "relevance_score": 0.0-1.0,
  "found_answer": true/false
}}"""

SYNTHESIZE_PROMPT = """Synthesize findings from multiple sources into a comprehensive answer.

ORIGINAL QUESTION: {question}

FINDINGS:
{findings}

Provide a complete, well-cited answer. Cite document names and sections."""

VERIFY_PROMPT = """Verify this answer against the evidence.

QUESTION: {question}
PROPOSED ANSWER: {answer}
EVIDENCE:
{evidence}

OUTPUT (JSON):
{{
  "is_grounded": true/false,
  "is_complete": true/false,
  "issues": ["issue1", ...],
  "confidence": 0.0-1.0
}}"""


class AgenticRAG(RAGPipeline):
    """
    Agentic Search-First RAG.

    Architecture: Decomposer → Searcher → Synthesizer → Verifier
    No pre-indexing needed — documents loaded on demand with large context.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})
        agents_cfg = pipeline_cfg.get("agents", {})
        workflow_cfg = pipeline_cfg.get("workflow", {})

        self.max_iterations = workflow_cfg.get("max_iterations", 3)

        # Create specialized agent clients
        decompose_model = agents_cfg.get("decomposer", {}).get("model", "gpt-4o")
        search_model = agents_cfg.get("searcher", {}).get("model", "gpt-4o")
        synth_model = agents_cfg.get("synthesizer", {}).get("model", "gpt-4o")
        verify_model = agents_cfg.get("verifier", {}).get("model", "gpt-4o")

        self._decomposer = LLMClient(model=decompose_model, temperature=0.0)
        self._searcher = LLMClient(model=search_model, temperature=0.0, max_tokens=4096)
        self._synthesizer = LLMClient(model=synth_model, temperature=0.0, max_tokens=4096)
        self._verifier = LLMClient(model=verify_model, temperature=0.0)

        # State
        self._documents: dict[str, dict] = {}

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Load documents into memory (no indexing needed for agentic approach)."""
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
            self._documents[doc_id] = doc_data

        elapsed = time.perf_counter() - start
        self._is_ingested = True

        logger.info(f"Agentic RAG: loaded {len(self._documents)} docs in {elapsed:.1f}s")

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=len(self._documents),
            ingestion_time_seconds=elapsed,
            index_size_bytes=0,  # No index created
            index_artifacts={},
        )

    def query(self, question: str) -> RAGResponse:
        """Multi-agent pipeline: decompose → search → synthesize → verify."""
        start = time.perf_counter()
        total_input = 0
        total_output = 0
        total_cost = 0.0

        # === Agent 1: Decomposer ===
        decompose_resp = self._decomposer.generate(
            DECOMPOSE_PROMPT.format(question=question), json_mode=True,
        )
        total_input += decompose_resp.input_tokens
        total_output += decompose_resp.output_tokens
        total_cost += decompose_resp.cost_usd

        try:
            decomposed = json.loads(decompose_resp.content)
            sub_questions = decomposed.get("sub_questions", [question])
        except (json.JSONDecodeError, KeyError):
            sub_questions = [question]

        # === Agent 2: Searcher (per sub-question, per document) ===
        all_findings = []
        all_contexts = []
        all_references = []

        for sub_q in sub_questions:
            for doc_id, doc_data in self._documents.items():
                full_text = doc_data.get("full_text", "")
                # Use large context: send up to 6000 chars per doc
                content_preview = full_text[:6000]

                search_resp = self._searcher.generate(
                    SEARCH_PROMPT.format(
                        question=sub_q,
                        doc_title=doc_data.get("title", doc_id),
                        content=content_preview,
                    ),
                    json_mode=True,
                )
                total_input += search_resp.input_tokens
                total_output += search_resp.output_tokens
                total_cost += search_resp.cost_usd

                try:
                    result = json.loads(search_resp.content)
                    passages = result.get("relevant_passages", [])
                    relevance = result.get("relevance_score", 0.0)
                except (json.JSONDecodeError, KeyError):
                    passages = []
                    relevance = 0.0

                if passages and relevance > 0.3:
                    for p in passages:
                        all_contexts.append(p)
                        all_references.append({
                            "doc_id": doc_id,
                            "title": doc_data.get("title", ""),
                            "sub_question": sub_q,
                            "relevance_score": relevance,
                        })
                    all_findings.append({
                        "sub_question": sub_q,
                        "document": doc_data.get("title", doc_id),
                        "passages": passages,
                    })

        # === Agent 3: Synthesizer ===
        findings_text = "\n\n".join(
            f"Sub-Q: {f['sub_question']}\nSource: {f['document']}\n"
            + "\n".join(f"- {p}" for p in f["passages"])
            for f in all_findings
        )

        synth_resp = self._synthesizer.generate(
            SYNTHESIZE_PROMPT.format(
                question=question, findings=findings_text or "No relevant findings.",
            )
        )
        total_input += synth_resp.input_tokens
        total_output += synth_resp.output_tokens
        total_cost += synth_resp.cost_usd

        answer = synth_resp.content

        # === Agent 4: Verifier ===
        verify_resp = self._verifier.generate(
            VERIFY_PROMPT.format(
                question=question,
                answer=answer,
                evidence=findings_text[:3000],
            ),
            json_mode=True,
        )
        total_input += verify_resp.input_tokens
        total_output += verify_resp.output_tokens
        total_cost += verify_resp.cost_usd

        try:
            verdict = json.loads(verify_resp.content)
            if not verdict.get("is_grounded", True):
                answer = f"[UNVERIFIED] {answer}"
        except (json.JSONDecodeError, KeyError):
            pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            answer=answer,
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
        """Simply load new documents — no indexing to update."""
        start = time.perf_counter()
        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            self._documents[doc_id] = doc_data
        elapsed = time.perf_counter() - start

        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=False,
            new_index_size_bytes=0,
        )
