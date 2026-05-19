#!/usr/bin/env python3
"""
Golden Q&A Generator — creates evaluation sets WITHOUT requiring any LLM.

Three sources:
1. FinanceBench ground truth (real Q&A from PatronusAI)
2. Deterministic extraction from document structure
3. Template-based question generation

Usage:
    python scripts/generate_golden_qa.py --domain all
"""
from __future__ import annotations

import argparse
import json
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_financebench_qa(finance_dir: Path) -> list[dict]:
    """Extract Q&A pairs from FinanceBench JSONL."""
    qa_pairs = []
    jsonl_path = finance_dir / "financebench_qa.jsonl"

    if not jsonl_path.exists():
        return []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                question = entry.get("question", "")
                answer = entry.get("answer", "")
                evidence = entry.get("evidence", "")
                company = entry.get("company", entry.get("ticker", ""))

                if question and answer:
                    qa_pairs.append({
                        "id": hashlib.md5(question.encode()).hexdigest()[:12],
                        "question": question,
                        "ground_truth_answer": answer,
                        "evidence": evidence,
                        "source_document": f"{company}_financial_data",
                        "domain": "finance",
                        "question_type": classify_question_type(question),
                        "difficulty": "medium",
                        "source": "financebench",
                    })
            except json.JSONDecodeError:
                continue

    return qa_pairs


def classify_question_type(question: str) -> str:
    """Classify question type based on content."""
    q = question.lower()

    if any(w in q for w in ["compare", "difference", "vs", "between", "both"]):
        return "multi_hop"
    elif any(w in q for w in ["trend", "change", "increase", "decrease", "growth"]):
        return "trend_analysis"
    elif any(w in q for w in ["how much", "what is the", "total", "revenue", "$", "percent"]):
        return "factual_numeric"
    elif any(w in q for w in ["what does", "define", "explain", "describe"]):
        return "factual_descriptive"
    elif any(w in q for w in ["why", "reason", "because", "cause"]):
        return "reasoning"
    else:
        return "factual"


def generate_from_document_structure(
    doc_data: dict, domain: str
) -> list[dict]:
    """Generate Q&A pairs deterministically from document structure."""
    qa_pairs = []
    doc_id = doc_data.get("doc_id", "unknown")
    title = doc_data.get("title", "Unknown Document")
    sections = doc_data.get("sections", [])
    full_text = doc_data.get("full_text", "")

    # Extract facts from section titles and content
    for i, section in enumerate(sections):
        if not isinstance(section, dict):
            continue

        sec_title = section.get("title", "")
        sec_content = section.get("content", "")

        if not sec_title or not sec_content or len(sec_content) < 50:
            continue

        # Template 1: "What does [section] cover?"
        qa_pairs.append({
            "id": f"{doc_id}_sec{i}_overview",
            "question": f"What information is provided in the '{sec_title}' section of {title}?",
            "ground_truth_answer": sec_content[:500],
            "evidence": sec_content[:500],
            "source_document": doc_id,
            "domain": domain,
            "question_type": "factual_descriptive",
            "difficulty": "easy",
            "source": "deterministic_structure",
        })

        # Template 2: Extract numbers and create questions
        numbers = re.findall(r'\$[\d,.]+[BMK]?\b|\d+(?:\.\d+)?%|\d{1,3}(?:,\d{3})+', sec_content)
        if numbers and len(numbers) >= 1:
            # Find a sentence containing the number
            sentences = re.split(r'(?<=[.!?])\s+', sec_content)
            for sent in sentences:
                found_nums = re.findall(r'\$[\d,.]+[BMK]?\b|\d+(?:\.\d+)?%', sent)
                if found_nums and len(sent) > 30:
                    qa_pairs.append({
                        "id": hashlib.md5(sent[:50].encode()).hexdigest()[:12],
                        "question": _create_numeric_question(sent, title, sec_title),
                        "ground_truth_answer": sent.strip(),
                        "evidence": sent.strip(),
                        "source_document": doc_id,
                        "domain": domain,
                        "question_type": "factual_numeric",
                        "difficulty": "medium",
                        "source": "deterministic_numeric",
                    })
                    break  # One per section

    # Template 3: Cross-section questions (multi-hop)
    if len(sections) >= 3:
        sec_titles = [s.get("title", "") for s in sections if isinstance(s, dict) and s.get("title")]
        if len(sec_titles) >= 2:
            qa_pairs.append({
                "id": f"{doc_id}_multihop",
                "question": f"Based on {title}, what are the key topics covered across the '{sec_titles[0]}' and '{sec_titles[-1]}' sections?",
                "ground_truth_answer": f"The '{sec_titles[0]}' section covers: {sections[0].get('content', '')[:200]}. The '{sec_titles[-1]}' section covers: {sections[-1].get('content', '')[:200]}.",
                "evidence": f"{sections[0].get('content', '')[:200]} ... {sections[-1].get('content', '')[:200]}",
                "source_document": doc_id,
                "domain": domain,
                "question_type": "multi_hop",
                "difficulty": "hard",
                "source": "deterministic_multihop",
            })

    return qa_pairs


def _create_numeric_question(sentence: str, doc_title: str, section_title: str) -> str:
    """Create a question about a numeric fact in a sentence."""
    # Remove the number to form a "what is" question
    nums = re.findall(r'\$[\d,.]+[BMK]?\b|\d+(?:\.\d+)?%', sentence)
    if nums:
        # Try to extract the subject before the number
        parts = sentence.split(nums[0])
        subject = parts[0].strip().rstrip("was were is are of :,").strip()
        if len(subject) > 10:
            return f"According to {doc_title}, what is {subject.lower()}?"
    return f"What numeric data is mentioned in the '{section_title}' section of {doc_title}?"


def generate_legal_qa(doc_data: dict) -> list[dict]:
    """Generate Q&A for legal contracts."""
    qa_pairs = []
    doc_id = doc_data.get("doc_id", "unknown")
    title = doc_data.get("title", "Unknown")
    sections = doc_data.get("sections", [])

    # Legal-specific templates
    legal_templates = [
        ("What is the termination clause?", "term", "TERMINATION"),
        ("What are the payment terms?", "payment", "COMPENSATION"),
        ("What is the liability limitation?", "liab", "LIABILITY"),
        ("What is the governing law?", "gov", "GOVERNING"),
        ("What are the intellectual property provisions?", "ip", "INTELLECTUAL"),
        ("What is the indemnification provision?", "indem", "INDEMNIFICATION"),
        ("What is the confidentiality obligation?", "conf", "CONFIDENTIAL"),
    ]

    for question_template, q_id, keyword in legal_templates:
        for sec in sections:
            if not isinstance(sec, dict):
                continue
            sec_title = sec.get("title", "").upper()
            sec_content = sec.get("content", "")
            if keyword in sec_title and sec_content:
                qa_pairs.append({
                    "id": f"{doc_id}_{q_id}",
                    "question": f"In the {title}, {question_template.lower()}",
                    "ground_truth_answer": sec_content[:500],
                    "evidence": sec_content[:500],
                    "source_document": doc_id,
                    "domain": "legal",
                    "question_type": "factual_descriptive",
                    "difficulty": "medium",
                    "source": "deterministic_legal",
                })
                break

    return qa_pairs


def generate_technical_qa(doc_data: dict) -> list[dict]:
    """Generate Q&A for technical documentation."""
    qa_pairs = []
    doc_id = doc_data.get("doc_id", "unknown")
    title = doc_data.get("title", "Unknown")
    sections = doc_data.get("sections", [])

    # Technical-specific templates
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            continue
        sec_title = sec.get("title", "")
        sec_content = sec.get("content", "")
        if not sec_content or len(sec_content) < 30:
            continue

        # "How do you..." questions for procedural content
        if any(kw in sec_title.lower() for kw in ["setup", "install", "config", "usage", "how"]):
            qa_pairs.append({
                "id": f"{doc_id}_howto_{i}",
                "question": f"How do you {sec_title.lower().strip('.')}?",
                "ground_truth_answer": sec_content[:500],
                "evidence": sec_content[:500],
                "source_document": doc_id,
                "domain": "technical",
                "question_type": "procedural",
                "difficulty": "medium",
                "source": "deterministic_technical",
            })

        # "What is..." for definitional content
        elif any(kw in sec_title.lower() for kw in ["overview", "definition", "architecture", "what"]):
            qa_pairs.append({
                "id": f"{doc_id}_whatis_{i}",
                "question": f"What is {sec_title.lower().strip('.')}?",
                "ground_truth_answer": sec_content[:500],
                "evidence": sec_content[:500],
                "source_document": doc_id,
                "domain": "technical",
                "question_type": "factual_descriptive",
                "difficulty": "easy",
                "source": "deterministic_technical",
            })

    return qa_pairs


def generate_golden_qa(domain: str) -> list[dict]:
    """Generate golden Q&A for a domain."""
    processed_dir = PROJECT_ROOT / "data" / "processed" / domain

    if not processed_dir.exists():
        print(f"  ⚠ No processed data for {domain}")
        return []

    all_qa = []

    # Source 1: FinanceBench ground truth
    if domain == "finance":
        fb_qa = load_financebench_qa(PROJECT_ROOT / "data" / "raw" / "finance")
        all_qa.extend(fb_qa)
        print(f"  ✓ FinanceBench: {len(fb_qa)} Q&A pairs")

    # Source 2: Deterministic extraction from processed docs
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

        for doc_meta in manifest["documents"]:
            doc_file = processed_dir / f"{doc_meta['doc_id']}.json"
            if not doc_file.exists():
                continue
            with open(doc_file) as f:
                doc_data = json.load(f)

            # Domain-specific generators
            if domain == "legal":
                qa = generate_legal_qa(doc_data)
            elif domain == "technical":
                qa = generate_technical_qa(doc_data)
            else:
                qa = generate_from_document_structure(doc_data, domain)

            all_qa.extend(qa)

    # Deduplicate by question
    seen = set()
    unique_qa = []
    for qa in all_qa:
        q_hash = hashlib.md5(qa["question"].encode()).hexdigest()
        if q_hash not in seen:
            seen.add(q_hash)
            unique_qa.append(qa)

    return unique_qa


def save_golden_qa(qa_pairs: list[dict], domain: str):
    """Save golden Q&A to JSONL."""
    output_dir = PROJECT_ROOT / "data" / "golden_qa"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{domain}_golden_qa.jsonl"
    with open(output_file, "w") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa) + "\n")

    print(f"  ✓ Saved: {output_file.name} ({len(qa_pairs)} pairs)")

    # Also save a summary
    type_counts: dict[str, int] = {}
    for qa in qa_pairs:
        qt = qa.get("question_type", "unknown")
        type_counts[qt] = type_counts.get(qt, 0) + 1

    summary = {
        "domain": domain,
        "total_questions": len(qa_pairs),
        "question_types": type_counts,
        "sources": list(set(qa.get("source", "") for qa in qa_pairs)),
    }

    summary_file = output_dir / f"{domain}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden Q&A evaluation sets (no LLM required)"
    )
    parser.add_argument(
        "--domain", choices=["finance", "legal", "technical", "all"],
        default="all",
    )
    args = parser.parse_args()

    print("🎯 Golden Q&A Generator (Deterministic — No LLM Required)")
    print("=" * 55)

    domains = ["finance", "legal", "technical"] if args.domain == "all" else [args.domain]

    for domain in domains:
        print(f"\n📋 Domain: {domain}")
        qa_pairs = generate_golden_qa(domain)
        if qa_pairs:
            save_golden_qa(qa_pairs, domain)
            print(f"  Total: {len(qa_pairs)} evaluation questions")
        else:
            print(f"  ⚠ No Q&A generated for {domain}")

    print("\n✅ Golden Q&A generation complete!")
    print(f"   Output: {PROJECT_ROOT / 'data' / 'golden_qa'}")


if __name__ == "__main__":
    main()
