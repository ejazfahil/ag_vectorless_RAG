"""
Golden Q&A Generator — creates ground-truth question-answer pairs
from processed documents using LLM generation + human validation support.

Generates 3 types of questions per document:
  1. Single-hop factual (30%)
  2. Multi-hop reasoning (40%)
  3. Needle-in-haystack (30%)
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.llm_client import LLMClient


@dataclass
class QAPair:
    """A single question-answer pair with metadata."""

    id: str
    question: str
    ground_truth: str                # The correct answer
    question_type: str               # single_hop | multi_hop | needle
    domain: str
    source_doc_id: str
    source_sections: list[str]       # Section IDs needed to answer
    difficulty: str = "medium"       # easy | medium | hard
    human_validated: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class GoldenQASet:
    """Collection of Q&A pairs for a domain."""

    domain: str
    pairs: list[QAPair]
    total_pairs: int = 0

    def __post_init__(self):
        self.total_pairs = len(self.pairs)


GENERATION_PROMPT = """You are an expert QA pair generator for RAG evaluation.

Given the following document sections, generate {count} question-answer pairs.

DISTRIBUTION:
- {single_hop_count} SINGLE-HOP FACTUAL questions (direct facts from one section)
- {multi_hop_count} MULTI-HOP REASONING questions (require combining info from 2+ sections)
- {needle_count} NEEDLE-IN-HAYSTACK questions (specific detail buried in text)

RULES:
1. Questions must be answerable ONLY from the provided text
2. Answers must be complete and cite the relevant section(s)
3. Include a mix of easy, medium, and hard questions
4. Avoid questions answerable with general knowledge
5. For multi-hop, ensure the answer genuinely requires multiple sections

DOCUMENT TITLE: {doc_title}
DOMAIN: {domain}

SECTIONS:
{sections_text}

OUTPUT FORMAT (JSON array):
[
  {{
    "question": "...",
    "ground_truth": "...",
    "question_type": "single_hop|multi_hop|needle",
    "source_sections": ["section_id_1", "section_id_2"],
    "difficulty": "easy|medium|hard"
  }}
]

Generate exactly {count} Q&A pairs:"""


class GoldenQAGenerator:
    """
    Generate ground-truth Q&A pairs for RAG evaluation.

    Usage:
        generator = GoldenQAGenerator(model="gpt-4o")
        qa_set = generator.generate(corpus, num_pairs=50)
        generator.save(qa_set, "data/golden_qa/finance.json")
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        single_hop_ratio: float = 0.3,
        multi_hop_ratio: float = 0.4,
        needle_ratio: float = 0.3,
    ):
        self.client = LLMClient(model=model, temperature=0.7)
        self.single_hop_ratio = single_hop_ratio
        self.multi_hop_ratio = multi_hop_ratio
        self.needle_ratio = needle_ratio

    def generate(
        self,
        corpus: Any,  # ProcessedCorpus
        num_pairs: int = 50,
    ) -> GoldenQASet:
        """
        Generate Q&A pairs from a processed corpus.

        Distributes questions across all documents proportionally.

        Args:
            corpus: ProcessedCorpus object.
            num_pairs: Total number of Q&A pairs to generate.

        Returns:
            GoldenQASet with all generated pairs.
        """
        all_pairs = []
        pairs_per_doc = max(1, num_pairs // len(corpus.documents))

        logger.info(
            f"Generating {num_pairs} Q&A pairs for domain={corpus.domain} "
            f"({pairs_per_doc} per doc across {len(corpus.documents)} docs)"
        )

        for doc in corpus.documents:
            try:
                doc_pairs = self._generate_for_document(
                    doc, count=pairs_per_doc, domain=corpus.domain
                )
                all_pairs.extend(doc_pairs)
                logger.debug(f"  ✓ {doc.filename}: {len(doc_pairs)} pairs generated")
            except Exception as e:
                logger.error(f"  ✗ Failed for {doc.filename}: {e}")

        # Trim to exact count if we generated too many
        if len(all_pairs) > num_pairs:
            all_pairs = random.sample(all_pairs, num_pairs)

        # Assign unique IDs
        for i, pair in enumerate(all_pairs):
            pair.id = f"{corpus.domain}_{i+1:03d}"

        qa_set = GoldenQASet(domain=corpus.domain, pairs=all_pairs)
        logger.info(f"Generated {qa_set.total_pairs} Q&A pairs for {corpus.domain}")
        return qa_set

    def _generate_for_document(
        self, doc: Any, count: int, domain: str
    ) -> list[QAPair]:
        """Generate Q&A pairs for a single document."""
        # Prepare section text
        sections_text = ""
        for section in doc.sections[:20]:  # Limit to avoid token overflow
            sections_text += f"\n--- Section {section.id}: {section.title} ---\n"
            # Truncate very long sections
            content = section.content[:2000] if len(section.content) > 2000 else section.content
            sections_text += content + "\n"

        # Calculate type distribution
        single_hop_count = max(1, round(count * self.single_hop_ratio))
        multi_hop_count = max(1, round(count * self.multi_hop_ratio))
        needle_count = count - single_hop_count - multi_hop_count

        prompt = GENERATION_PROMPT.format(
            count=count,
            single_hop_count=single_hop_count,
            multi_hop_count=multi_hop_count,
            needle_count=needle_count,
            doc_title=doc.title,
            domain=domain,
            sections_text=sections_text,
        )

        response = self.client.generate(prompt, json_mode=True)

        try:
            raw_pairs = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON response for {doc.filename}")
            return []

        pairs = []
        for raw in raw_pairs:
            pairs.append(QAPair(
                id="",  # Assigned later
                question=raw.get("question", ""),
                ground_truth=raw.get("ground_truth", ""),
                question_type=raw.get("question_type", "single_hop"),
                domain=domain,
                source_doc_id=doc.doc_id,
                source_sections=raw.get("source_sections", []),
                difficulty=raw.get("difficulty", "medium"),
            ))

        return pairs

    def save(self, qa_set: GoldenQASet, output_path: str) -> None:
        """Save Q&A set to JSON file."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "domain": qa_set.domain,
            "total_pairs": qa_set.total_pairs,
            "pairs": [asdict(p) for p in qa_set.pairs],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {qa_set.total_pairs} Q&A pairs to {output_path}")

    def load(self, path: str) -> GoldenQASet:
        """Load Q&A set from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        pairs = [QAPair(**p) for p in data["pairs"]]
        return GoldenQASet(domain=data["domain"], pairs=pairs)
