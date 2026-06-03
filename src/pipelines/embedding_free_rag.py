"""
Pipeline 6: Embedding-Free RAG (Maghakian et al., EMNLP 2025)
──────────────────────────────────────────────────────────────
Faithful re-implementation of Algorithm 1 from the paper.

Key idea: Use an LLM to extract verbatim quotations from sub-documents,
then use Levenshtein fuzzy matching to anchor those quotes back to exact
sentence indices, forming a ±w context window for answer generation.

Two-LLM architecture (per paper Section 2.4.2):
  - QUOTE_LLM: fast/small model for quotation extraction
  - ANSWER_LLM: strong/large model for answer synthesis

Reference: Maghakian et al., "Embedding-Free RAG", EMNLP 2025
"""

from __future__ import annotations

import json
import re
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

# ── Prompts (faithful to paper's Appendix A.1/A.2) ──────────────

QUOTE_EXTRACTION_PROMPT = """Given the question and document below, return a JSON list of
verbatim quotations from the document that contain information necessary
to answer the question. Extract exact text spans — do not paraphrase.

Question: {question}
Document: {document}

Return ONLY valid JSON: {{"quotes": ["quote1", "quote2", ...]}}
If no relevant information found, return: {{"quotes": []}}"""

ANSWER_PROMPT = """Answer the question using ONLY the provided context passages.
Cite the specific passages. If the context is insufficient, say so.

Question: {question}

Context:
{context}

Answer:"""

SUMMARY_PROMPT = """Provide a 2-3 sentence summary of this document's main topic
and key points.

Document (first 5000 words):
{content}

Summary:"""


def _sent_tokenize(text: str) -> list[str]:
    """Simple sentence tokenizer (no NLTK dependency)."""
    # Split on sentence-ending punctuation followed by space/newline
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,        # insertion
                prev_row[j + 1] + 1,    # deletion
                prev_row[j] + cost,     # substitution
            ))
        prev_row = curr_row

    return prev_row[-1]


try:
    from rapidfuzz.distance import Levenshtein as _RFLev
    def fuzzy_distance(s1: str, s2: str) -> int:
        return _RFLev.distance(s1, s2)
except ImportError:
    fuzzy_distance = _levenshtein_distance


class EmbeddingFreeRAG(RAGPipeline):
    """
    Embedding-Free RAG — Algorithm 1 from Maghakian et al. (EMNLP 2025).

    Ingestion: Split documents into sentences, group into sub-documents
    Retrieval: LLM extracts verbatim quotes → Levenshtein anchor → ±w window
    Generation: Strong LLM answers from anchored context
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})
        quote_cfg = pipeline_cfg.get("quote_extraction", {})
        answer_cfg = pipeline_cfg.get("answer_generation", {})

        # Two-LLM split (paper Section 2.4.2)
        quote_model = quote_cfg.get("model", "llama3.2:3b")
        answer_model = answer_cfg.get("model", "llama3.2:3b")

        self._quote_llm = LLMClient(model=quote_model, temperature=0.0)
        self._answer_llm = LLMClient(model=answer_model, temperature=0.1)

        # Algorithm parameters
        self._subdoc_words = pipeline_cfg.get("subdoc_words", 3000)
        self._window_size = pipeline_cfg.get("window_size", 5)  # ±w sentences
        self._summary_words = pipeline_cfg.get("summary_first_n_words", 5000)

        # State
        self._documents: dict[str, dict] = {}
        self._sentences: dict[str, list[str]] = {}     # doc_id -> sentences
        self._subdocs: dict[str, list[str]] = {}        # doc_id -> sub-documents
        self._summaries: dict[str, str] = {}             # doc_id -> summary

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Split documents into sentences and sub-documents."""
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
            full_text = doc_data.get("full_text", "")

            # Sentence tokenize
            sentences = _sent_tokenize(full_text)
            self._sentences[doc_id] = sentences

            # Group into sub-documents of ~subdoc_words each
            subdocs = []
            current_subdoc = []
            current_words = 0
            for sent in sentences:
                word_count = len(sent.split())
                if current_words + word_count > self._subdoc_words and current_subdoc:
                    subdocs.append(" ".join(current_subdoc))
                    current_subdoc = []
                    current_words = 0
                current_subdoc.append(sent)
                current_words += word_count
            if current_subdoc:
                subdocs.append(" ".join(current_subdoc))
            self._subdocs[doc_id] = subdocs

            # Generate summary from first N words (can be done without LLM)
            first_words = " ".join(full_text.split()[:self._summary_words])
            self._summaries[doc_id] = first_words[:500]  # Use first 500 chars as summary

            logger.debug(
                f"EF-RAG indexed {doc_id}: {len(sentences)} sentences, "
                f"{len(subdocs)} sub-documents"
            )

        elapsed = time.perf_counter() - start
        self._is_ingested = True

        logger.info(
            f"Embedding-Free RAG ingestion: {len(self._documents)} docs in {elapsed:.1f}s"
        )

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=len(self._documents),
            ingestion_time_seconds=elapsed,
            index_size_bytes=0,  # No index — everything is in-memory sentences
            index_artifacts={},
            metadata={
                "total_sentences": sum(len(s) for s in self._sentences.values()),
                "total_subdocs": sum(len(s) for s in self._subdocs.values()),
            },
        )

    def query(self, question: str) -> RAGResponse:
        """
        Algorithm 1: Quote extraction → Levenshtein anchoring → ±w window → answer.
        """
        start = time.perf_counter()
        total_input = 0
        total_output = 0
        total_cost = 0.0

        all_contexts = []
        all_references = []

        for doc_id in self._documents:
            sentences = self._sentences.get(doc_id, [])
            subdocs = self._subdocs.get(doc_id, [])
            summary = self._summaries.get(doc_id, "")

            if not sentences:
                continue

            # Step 1: Extract quotes from each sub-document
            all_anchors = []

            for sub_idx, subdoc in enumerate(subdocs):
                # Prepend summary for global context (paper recommendation)
                prompted = f"Summary: {summary}\n\n{subdoc}"

                resp = self._quote_llm.generate(
                    QUOTE_EXTRACTION_PROMPT.format(
                        question=question, document=prompted[:4000],
                    ),
                    json_mode=True,
                )
                total_input += resp.input_tokens
                total_output += resp.output_tokens
                total_cost += resp.cost_usd

                try:
                    result = json.loads(resp.content)
                    quotes = result.get("quotes", [])
                except (json.JSONDecodeError, KeyError):
                    quotes = []

                # Step 2: Levenshtein anchoring — find closest sentence
                for quote in quotes:
                    if not quote or len(quote) < 10:
                        continue
                    # Find the sentence with minimum edit distance
                    best_idx = min(
                        range(len(sentences)),
                        key=lambda i: fuzzy_distance(
                            quote[:200].lower(), sentences[i][:200].lower()
                        ),
                    )
                    all_anchors.append(best_idx)

            if not all_anchors:
                continue

            # Step 3: Build ±w context windows and merge overlaps
            w = self._window_size
            windows = [
                (max(0, a - w), min(len(sentences), a + w + 1))
                for a in all_anchors
            ]
            merged = self._merge_overlapping(windows)

            # Build context from merged windows
            doc_context_parts = []
            for s, e in merged:
                chunk = " ".join(sentences[s:e])
                doc_context_parts.append(chunk)
                all_references.append({
                    "doc_id": doc_id,
                    "title": self._documents[doc_id].get("title", ""),
                    "sentence_range": [s, e],
                    "method": "embedding_free_levenshtein",
                })

            doc_context = "\n\n".join(doc_context_parts)
            all_contexts.append(doc_context)

        # Step 4: Generate answer with strong LLM
        context_text = "\n\n---\n\n".join(all_contexts[:5])
        gen_resp = self._answer_llm.generate(
            ANSWER_PROMPT.format(question=question, context=context_text),
        )
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
        """Add new documents — just sentence-split them."""
        start = time.perf_counter()
        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            self._documents[doc_id] = doc_data
            full_text = doc_data.get("full_text", "")
            self._sentences[doc_id] = _sent_tokenize(full_text)
            # Build subdocs
            subdocs = []
            current, words = [], 0
            for sent in self._sentences[doc_id]:
                wc = len(sent.split())
                if words + wc > self._subdoc_words and current:
                    subdocs.append(" ".join(current))
                    current, words = [], 0
                current.append(sent)
                words += wc
            if current:
                subdocs.append(" ".join(current))
            self._subdocs[doc_id] = subdocs
            self._summaries[doc_id] = full_text[:500]

        elapsed = time.perf_counter() - start
        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=False,
            new_index_size_bytes=0,
        )

    @staticmethod
    def _merge_overlapping(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping sentence windows."""
        if not windows:
            return []
        sorted_w = sorted(windows, key=lambda x: x[0])
        merged = [sorted_w[0]]
        for s, e in sorted_w[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged
