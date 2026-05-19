"""
Pipeline 7: Three-Stage Hybrid (Blueprint E.1 — Novel Contribution)
═══════════════════════════════════════════════════════════════════
The most defensible novel contribution — no published system combines
all three stages.

Architecture:
  Stage 1: BM25 top-50 (~1 ms, lexical recall)
  Stage 2: PageIndex tree-reasoning over matching chapters (~5 sec, semantic precision)
  Stage 3: Embedding-Free verbatim quote extraction + Levenshtein anchoring (~2 sec, span precision)

Each stage handles a failure mode of the next:
  - BM25 catches keyword recall
  - PageIndex catches structural relevance
  - Embedding-Free catches paraphrase-tolerant span localization

Includes:
  - Iterative multi-hop (E.2): "Do you have enough info?" loop
  - Self-evaluation (E.3): CRAG-style post-generation faithfulness scoring
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from src.pipelines.base import (
    RAGPipeline, RAGResponse, IngestionReport, UpdateReport,
)
from src.utils.llm_client import LLMClient


# ── Prompts ──────────────────────────────────────────────────────

TREE_NAVIGATION_PROMPT = """Given the document tree below, identify which sections
are most likely to contain information relevant to this question.

Question: {question}

Document Tree:
{tree}

Return a JSON list of section IDs to expand: {{"sections": ["id1", "id2", ...]}}
Select at most 5 most relevant sections."""

QUOTE_EXTRACTION_PROMPT = """Given the question and text below, extract verbatim
quotations that contain information needed to answer the question.

Question: {question}
Text: {text}

Return JSON: {{"quotes": ["exact quote 1", "exact quote 2", ...]}}
Extract up to 5 most relevant quotes. If none found, return: {{"quotes": []}}"""

ANSWER_PROMPT = """Answer the question using ONLY the provided evidence passages.
Each passage is labeled with its source. Cite sources for every claim.

Question: {question}

Evidence:
{evidence}

Answer (be precise and cite sources):"""

SUFFICIENCY_CHECK_PROMPT = """You retrieved the following context to answer a question.
Is this context sufficient to provide a complete, accurate answer?

Question: {question}
Context: {context}

Respond with JSON: {{"sufficient": true/false, "follow_up_query": "..." or null}}
If not sufficient, propose a follow-up retrieval query."""

FAITHFULNESS_CHECK_PROMPT = """Score whether the answer is faithful to the provided context.
Every claim in the answer must be supported by the context.

Answer: {answer}
Context: {context}

Return JSON: {{"faithfulness_score": 0.0-1.0, "unsupported_claims": ["..."]}}"""


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(curr_row[j] + 1, prev_row[j + 1] + 1, prev_row[j] + cost))
        prev_row = curr_row
    return prev_row[-1]


try:
    from rapidfuzz.distance import Levenshtein as _RFLev
    def fuzzy_distance(s1: str, s2: str) -> int:
        return _RFLev.distance(s1, s2)
except ImportError:
    fuzzy_distance = _levenshtein_distance


def _sent_tokenize(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


class ThreeStageHybridRAG(RAGPipeline):
    """
    Three-Stage Hybrid RAG — Blueprint E.1 (Novel Contribution).

    Combines:
    1. BM25 lexical retrieval (stage 1: recall)
    2. PageIndex tree reasoning (stage 2: structural precision)
    3. Embedding-Free quote extraction (stage 3: span precision)

    Plus:
    - Iterative multi-hop (E.2)
    - Self-evaluation / CRAG-style (E.3)
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        pipeline_cfg = config.get("pipeline", {})

        model = pipeline_cfg.get("model", "qwen3:8b")
        quote_model = pipeline_cfg.get("quote_model", "llama3.2:3b")

        self._reasoning_llm = LLMClient(model=model, temperature=0.0)
        self._quote_llm = LLMClient(model=quote_model, temperature=0.0)
        self._answer_llm = LLMClient(model=model, temperature=0.1)

        # Stage 1: BM25 parameters
        self._bm25_top_k = pipeline_cfg.get("bm25_top_k", 50)
        self._k1 = pipeline_cfg.get("bm25_k1", 1.5)
        self._b = pipeline_cfg.get("bm25_b", 0.75)

        # Stage 2: PageIndex tree parameters
        self._max_tree_sections = pipeline_cfg.get("max_tree_sections", 5)

        # Stage 3: Embedding-Free parameters
        self._window_size = pipeline_cfg.get("window_size", 5)

        # E.2: Multi-hop parameters
        self._max_hops = pipeline_cfg.get("max_hops", 3)
        self._enable_multi_hop = pipeline_cfg.get("enable_multi_hop", True)

        # E.3: Self-evaluation parameters
        self._enable_self_eval = pipeline_cfg.get("enable_self_eval", True)
        self._faithfulness_threshold = pipeline_cfg.get("faithfulness_threshold", 0.7)

        # Internal state
        self._documents: dict[str, dict] = {}
        self._chunks: list[dict] = []       # BM25 chunks
        self._doc_trees: dict[str, list] = {}  # Document trees
        self._sentences: dict[str, list[str]] = {}  # Per-doc sentences

        # BM25 index
        self._df: dict[str, int] = {}
        self._doc_len: list[int] = []
        self._avg_dl: float = 0.0
        self._tf: list[dict[str, int]] = []
        self._N: int = 0

    def ingest(self, corpus_path: str) -> IngestionReport:
        """Ingest documents for all three stages simultaneously."""
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

            # Stage 1: Build BM25 chunks
            sections = doc_data.get("sections", [])
            for i, section in enumerate(sections):
                if not isinstance(section, dict):
                    continue
                content = section.get("content", "")
                if len(content) < 20:
                    continue
                self._chunks.append({
                    "doc_id": doc_id,
                    "section_idx": i,
                    "title": section.get("title", ""),
                    "content": content,
                    "full_text_offset": full_text.find(content[:50]),
                })

            # Stage 2: Build document tree
            tree = []
            for i, section in enumerate(sections):
                if not isinstance(section, dict):
                    continue
                tree.append({
                    "id": f"{doc_id}_s{i}",
                    "title": section.get("title", ""),
                    "summary": section.get("content", "")[:150],
                    "section_idx": i,
                })
            self._doc_trees[doc_id] = tree

            # Stage 3: Sentence-tokenize for Embedding-Free
            self._sentences[doc_id] = _sent_tokenize(full_text)

        # Build BM25 index over chunks
        self._build_bm25_index()

        elapsed = time.perf_counter() - start
        self._is_ingested = True

        logger.info(
            f"Three-Stage Hybrid ingestion: {len(self._documents)} docs, "
            f"{len(self._chunks)} chunks, "
            f"{sum(len(s) for s in self._sentences.values())} sentences "
            f"in {elapsed:.2f}s"
        )

        return IngestionReport(
            pipeline_name=self.name,
            num_documents=len(self._documents),
            ingestion_time_seconds=elapsed,
            index_size_bytes=0,
            index_artifacts={},
            metadata={
                "num_chunks": len(self._chunks),
                "num_tree_nodes": sum(len(t) for t in self._doc_trees.values()),
                "num_sentences": sum(len(s) for s in self._sentences.values()),
            },
        )

    def _build_bm25_index(self):
        """Build in-memory BM25 index over chunks."""
        self._N = len(self._chunks)
        self._tf = []
        self._df = {}
        self._doc_len = []

        for chunk in self._chunks:
            text = chunk["content"].lower()
            tokens = re.findall(r'\w+', text)
            self._doc_len.append(len(tokens))

            tf: dict[str, int] = {}
            seen = set()
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
                if token not in seen:
                    self._df[token] = self._df.get(token, 0) + 1
                    seen.add(token)
            self._tf.append(tf)

        self._avg_dl = sum(self._doc_len) / max(self._N, 1)

    def _bm25_search(self, query: str, top_k: int = 50) -> list[tuple[int, float]]:
        """BM25 search over chunks."""
        import math
        tokens = re.findall(r'\w+', query.lower())
        scores = []

        for i in range(self._N):
            score = 0.0
            for token in tokens:
                if token not in self._tf[i]:
                    continue
                tf = self._tf[i][token]
                df = self._df.get(token, 0)
                idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1)
                denom = tf + self._k1 * (1 - self._b + self._b * self._doc_len[i] / self._avg_dl)
                score += idf * (tf * (self._k1 + 1)) / denom
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def query(self, question: str) -> RAGResponse:
        """
        Three-stage retrieval + iterative multi-hop + self-evaluation.
        """
        start = time.perf_counter()
        total_input, total_output, total_cost = 0, 0, 0.0
        all_evidence = []
        all_references = []

        current_query = question
        for hop in range(self._max_hops):
            logger.debug(f"Hop {hop+1}/{self._max_hops}: '{current_query[:60]}...'")

            # ═══ STAGE 1: BM25 Lexical Recall ═══════════════════
            bm25_results = self._bm25_search(current_query, top_k=self._bm25_top_k)
            stage1_docs = set()
            stage1_chunks = []

            for chunk_idx, score in bm25_results:
                if score > 0:
                    chunk = self._chunks[chunk_idx]
                    stage1_docs.add(chunk["doc_id"])
                    stage1_chunks.append(chunk)

            logger.debug(f"  Stage 1 (BM25): {len(stage1_chunks)} chunks from {len(stage1_docs)} docs")

            if not stage1_chunks:
                break

            # ═══ STAGE 2: PageIndex Tree Reasoning ══════════════
            stage2_sections = []
            for doc_id in stage1_docs:
                tree = self._doc_trees.get(doc_id, [])
                if not tree:
                    continue

                tree_text = "\n".join(
                    f"- [{n['id']}] {n['title']}: {n['summary']}"
                    for n in tree
                )

                resp = self._reasoning_llm.generate(
                    TREE_NAVIGATION_PROMPT.format(
                        question=current_query, tree=tree_text[:3000],
                    ),
                    json_mode=True,
                )
                total_input += resp.input_tokens
                total_output += resp.output_tokens
                total_cost += resp.cost_usd

                try:
                    nav = json.loads(resp.content)
                    section_ids = nav.get("sections", [])
                    for sid in section_ids[:self._max_tree_sections]:
                        for node in tree:
                            if node["id"] == sid:
                                idx = node["section_idx"]
                                sections = self._documents[doc_id].get("sections", [])
                                if idx < len(sections) and isinstance(sections[idx], dict):
                                    stage2_sections.append({
                                        "doc_id": doc_id,
                                        "section_idx": idx,
                                        "title": sections[idx].get("title", ""),
                                        "content": sections[idx].get("content", ""),
                                    })
                except (json.JSONDecodeError, KeyError):
                    # Fallback to top BM25 chunks
                    stage2_sections.extend(stage1_chunks[:5])

            logger.debug(f"  Stage 2 (Tree): {len(stage2_sections)} sections selected")

            # ═══ STAGE 3: Embedding-Free Quote Extraction ═══════
            stage3_evidence = []
            w = self._window_size

            for section in stage2_sections[:10]:
                doc_id = section["doc_id"]
                sentences = self._sentences.get(doc_id, [])
                if not sentences:
                    continue

                resp = self._quote_llm.generate(
                    QUOTE_EXTRACTION_PROMPT.format(
                        question=current_query,
                        text=section["content"][:3000],
                    ),
                    json_mode=True,
                )
                total_input += resp.input_tokens
                total_output += resp.output_tokens
                total_cost += resp.cost_usd

                try:
                    quotes = json.loads(resp.content).get("quotes", [])
                except (json.JSONDecodeError, KeyError):
                    quotes = []

                # Levenshtein anchoring
                anchors = []
                for quote in quotes:
                    if not quote or len(quote) < 10:
                        continue
                    best_idx = min(
                        range(len(sentences)),
                        key=lambda i: fuzzy_distance(
                            quote[:200].lower(), sentences[i][:200].lower()
                        ),
                    )
                    anchors.append(best_idx)

                # Build ±w windows and merge
                if anchors:
                    windows = [
                        (max(0, a - w), min(len(sentences), a + w + 1))
                        for a in anchors
                    ]
                    merged = self._merge_windows(windows)
                    for s, e in merged:
                        evidence_text = " ".join(sentences[s:e])
                        stage3_evidence.append(evidence_text)
                        all_references.append({
                            "doc_id": doc_id,
                            "title": section.get("title", ""),
                            "sentence_range": [s, e],
                            "stage": "embedding_free_levenshtein",
                        })

            # If Stage 3 produced no evidence, fall back to Stage 2 content
            if not stage3_evidence:
                stage3_evidence = [s["content"][:500] for s in stage2_sections[:5]]
                for s in stage2_sections[:5]:
                    all_references.append({
                        "doc_id": s["doc_id"],
                        "title": s.get("title", ""),
                        "stage": "tree_reasoning_fallback",
                    })

            all_evidence.extend(stage3_evidence)
            logger.debug(f"  Stage 3 (EF): {len(stage3_evidence)} evidence passages")

            # ═══ E.2: Sufficiency Check (Multi-Hop) ═════════════
            if not self._enable_multi_hop or hop == self._max_hops - 1:
                break

            context_preview = "\n".join(all_evidence[:3])[:2000]
            resp = self._reasoning_llm.generate(
                SUFFICIENCY_CHECK_PROMPT.format(
                    question=question, context=context_preview,
                ),
                json_mode=True,
            )
            total_input += resp.input_tokens
            total_output += resp.output_tokens
            total_cost += resp.cost_usd

            try:
                check = json.loads(resp.content)
                if check.get("sufficient", True):
                    logger.debug(f"  Context sufficient after {hop+1} hops")
                    break
                follow_up = check.get("follow_up_query")
                if follow_up:
                    current_query = follow_up
                    logger.debug(f"  Multi-hop: follow-up query: '{follow_up[:60]}...'")
                else:
                    break
            except (json.JSONDecodeError, KeyError):
                break

        # ═══ FINAL: Generate Answer ═══════════════════════════════
        evidence_text = "\n\n---\n\n".join(
            f"[Source {i+1}] {e[:800]}"
            for i, e in enumerate(all_evidence[:8])
        )

        gen_resp = self._answer_llm.generate(
            ANSWER_PROMPT.format(question=question, evidence=evidence_text),
        )
        total_input += gen_resp.input_tokens
        total_output += gen_resp.output_tokens
        total_cost += gen_resp.cost_usd

        answer = gen_resp.content

        # ═══ E.3: Self-Evaluation (CRAG-style) ════════════════════
        if self._enable_self_eval:
            faith_resp = self._reasoning_llm.generate(
                FAITHFULNESS_CHECK_PROMPT.format(
                    answer=answer[:1000], context=evidence_text[:2000],
                ),
                json_mode=True,
            )
            total_input += faith_resp.input_tokens
            total_output += faith_resp.output_tokens
            total_cost += faith_resp.cost_usd

            try:
                faith_result = json.loads(faith_resp.content)
                score = faith_result.get("faithfulness_score", 1.0)
                if score < self._faithfulness_threshold:
                    logger.warning(
                        f"  CRAG: Low faithfulness ({score:.2f}). "
                        f"Unsupported: {faith_result.get('unsupported_claims', [])}"
                    )
                    # Could re-retrieve here; for now just log
            except (json.JSONDecodeError, KeyError):
                pass

        elapsed_ms = (time.perf_counter() - start) * 1000

        return RAGResponse(
            answer=answer,
            retrieved_contexts=all_evidence,
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
        """Add new documents to all three stages."""
        start = time.perf_counter()
        for path in document_paths:
            with open(path) as f:
                doc_data = json.load(f)
            doc_id = doc_data.get("doc_id", Path(path).stem)
            self._documents[doc_id] = doc_data
            full_text = doc_data.get("full_text", "")
            sections = doc_data.get("sections", [])

            for i, sec in enumerate(sections):
                if isinstance(sec, dict) and sec.get("content"):
                    self._chunks.append({
                        "doc_id": doc_id, "section_idx": i,
                        "title": sec.get("title", ""), "content": sec["content"],
                    })
            self._doc_trees[doc_id] = [
                {"id": f"{doc_id}_s{i}", "title": sec.get("title", ""),
                 "summary": sec.get("content", "")[:150], "section_idx": i}
                for i, sec in enumerate(sections) if isinstance(sec, dict)
            ]
            self._sentences[doc_id] = _sent_tokenize(full_text)

        self._build_bm25_index()
        elapsed = time.perf_counter() - start

        return UpdateReport(
            pipeline_name=self.name,
            num_new_documents=len(document_paths),
            update_time_seconds=elapsed,
            required_full_reindex=True,
            new_index_size_bytes=0,
        )

    @staticmethod
    def _merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Merge overlapping windows."""
        if not windows:
            return []
        sorted_w = sorted(windows)
        merged = [sorted_w[0]]
        for s, e in sorted_w[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        return merged
