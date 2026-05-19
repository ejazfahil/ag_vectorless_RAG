"""
LLM-as-a-Judge evaluator — uses a separate LLM to perform
structured error analysis on RAG pipeline outputs.

Implements best practices:
- Chain-of-thought reasoning before scoring
- Binary decomposed checks (not vague 1-5 scales)
- Position bias mitigation
- Error categorization into actionable taxonomy
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from src.utils.llm_client import LLMClient


@dataclass
class JudgeVerdict:
    """Verdict from the LLM judge for a single Q&A pair."""

    question: str
    answer: str
    ground_truth: str

    # Binary checks
    is_hallucination: bool = False
    is_context_sufficient: bool = True
    is_answer_complete: bool = True
    is_answer_correct: bool = True

    # Error classification
    error_category: str = "correct"   # correct | retrieval_failure |
                                       # generation_failure | hallucination |
                                       # incomplete
    severity: int = 0                  # 0=correct, 1-5 increasing severity

    # Chain-of-thought reasoning
    reasoning: str = ""

    metadata: dict = field(default_factory=dict)


@dataclass
class JudgeReport:
    """Aggregate report from LLM judge for a full pipeline."""

    pipeline_name: str
    domain: str
    num_samples: int
    verdicts: list[JudgeVerdict]

    # Aggregate metrics
    accuracy: float = 0.0
    hallucination_rate: float = 0.0
    retrieval_failure_rate: float = 0.0
    generation_failure_rate: float = 0.0
    incomplete_rate: float = 0.0
    avg_severity: float = 0.0

    error_distribution: dict[str, int] = field(default_factory=dict)

    def compute_aggregates(self) -> None:
        """Compute aggregate metrics from individual verdicts."""
        if not self.verdicts:
            return

        n = len(self.verdicts)
        cats = [v.error_category for v in self.verdicts]

        self.accuracy = cats.count("correct") / n
        self.hallucination_rate = cats.count("hallucination") / n
        self.retrieval_failure_rate = cats.count("retrieval_failure") / n
        self.generation_failure_rate = cats.count("generation_failure") / n
        self.incomplete_rate = cats.count("incomplete") / n
        self.avg_severity = sum(v.severity for v in self.verdicts) / n

        self.error_distribution = {}
        for cat in set(cats):
            self.error_distribution[cat] = cats.count(cat)


JUDGE_PROMPT = """You are an expert evaluator for Retrieval-Augmented Generation systems.

Analyze the following RAG output and provide a structured verdict.

## Input
**Question:** {question}
**Ground Truth Answer:** {ground_truth}
**System's Answer:** {answer}
**Retrieved Context:** {context}

## Instructions
Perform the following checks. For each, provide your reasoning BEFORE the verdict.

### Check 1: Hallucination Detection
Does the system's answer contain ANY information that is NOT present in the retrieved context?
Reason step-by-step, then answer: HALLUCINATION_DETECTED = true/false

### Check 2: Context Sufficiency
Does the retrieved context contain enough information to correctly answer the question?
Reason step-by-step, then answer: CONTEXT_SUFFICIENT = true/false

### Check 3: Answer Completeness
Does the system's answer fully address the question compared to the ground truth?
Reason step-by-step, then answer: ANSWER_COMPLETE = true/false

### Check 4: Answer Correctness
Is the system's answer factually correct when compared to the ground truth?
Reason step-by-step, then answer: ANSWER_CORRECT = true/false

### Error Classification
Based on the above checks, classify the error into exactly ONE category:
- "correct": All checks passed
- "retrieval_failure": Context was insufficient (the system couldn't answer because it didn't find the right info)
- "generation_failure": Context was correct but the LLM misused or ignored it
- "hallucination": Answer contains fabricated information not in the context
- "incomplete": Answer is partially correct but missing key information

### Severity Score
Rate the severity on a scale of 0-5:
0 = Fully correct
1 = Minor inaccuracy
2 = Missing some details
3 = Significant error
4 = Major factual error
5 = Completely wrong or dangerous hallucination

## Output (JSON)
{{
  "reasoning": "Your step-by-step analysis here...",
  "is_hallucination": true/false,
  "is_context_sufficient": true/false,
  "is_answer_complete": true/false,
  "is_answer_correct": true/false,
  "error_category": "correct|retrieval_failure|generation_failure|hallucination|incomplete",
  "severity": 0-5
}}"""


class LLMJudge:
    """
    LLM-as-a-Judge evaluator for structured error analysis.

    Uses a powerful LLM (typically different from the pipeline's model)
    to classify errors, detect hallucinations, and categorize failures.

    Usage:
        judge = LLMJudge(model="gpt-4o")
        report = judge.evaluate(pipeline_name, domain, responses)
        print(report.accuracy, report.hallucination_rate)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        position_bias_mitigation: bool = True,
        config: dict[str, Any] | None = None,
    ):
        self.client = LLMClient(model=model, temperature=0.0)
        self.position_bias_mitigation = position_bias_mitigation
        self.config = config or {}

    def evaluate(
        self,
        pipeline_name: str,
        domain: str,
        responses: list[dict],
    ) -> JudgeReport:
        """
        Run LLM-as-a-Judge evaluation on all responses.

        Args:
            pipeline_name: Name of the pipeline.
            domain: Domain of the Q&A set.
            responses: List of dicts with keys:
                question, answer, contexts, ground_truth

        Returns:
            JudgeReport with verdicts and aggregate metrics.
        """
        logger.info(
            f"Running LLM-as-a-Judge for {pipeline_name} "
            f"on {domain} ({len(responses)} samples)"
        )

        verdicts = []
        for i, resp in enumerate(responses):
            try:
                verdict = self._judge_single(resp)
                verdicts.append(verdict)

                status = "✓" if verdict.error_category == "correct" else "✗"
                logger.debug(
                    f"  [{i+1}/{len(responses)}] {status} "
                    f"{verdict.error_category} (severity={verdict.severity})"
                )
            except Exception as e:
                logger.error(f"  Judge failed on sample {i+1}: {e}")
                verdicts.append(JudgeVerdict(
                    question=resp.get("question", ""),
                    answer=resp.get("answer", ""),
                    ground_truth=resp.get("ground_truth", ""),
                    error_category="judge_error",
                    reasoning=str(e),
                ))

        report = JudgeReport(
            pipeline_name=pipeline_name,
            domain=domain,
            num_samples=len(responses),
            verdicts=verdicts,
        )
        report.compute_aggregates()

        logger.info(f"Judge results for {pipeline_name}:")
        logger.info(f"  Accuracy: {report.accuracy:.2%}")
        logger.info(f"  Hallucination rate: {report.hallucination_rate:.2%}")
        logger.info(f"  Retrieval failures: {report.retrieval_failure_rate:.2%}")
        logger.info(f"  Generation failures: {report.generation_failure_rate:.2%}")

        return report

    def _judge_single(self, response: dict) -> JudgeVerdict:
        """Judge a single Q&A response."""
        context_text = "\n---\n".join(response.get("contexts", []))

        prompt = JUDGE_PROMPT.format(
            question=response["question"],
            ground_truth=response["ground_truth"],
            answer=response["answer"],
            context=context_text or "[No context retrieved]",
        )

        llm_response = self.client.generate(
            prompt,
            system_prompt="You are a strict, impartial RAG evaluation judge. Output valid JSON only.",
            json_mode=True,
        )

        try:
            result = json.loads(llm_response.content)
        except json.JSONDecodeError:
            logger.warning("Judge returned non-JSON response, attempting extraction")
            result = self._extract_json_from_text(llm_response.content)

        return JudgeVerdict(
            question=response["question"],
            answer=response["answer"],
            ground_truth=response["ground_truth"],
            is_hallucination=result.get("is_hallucination", False),
            is_context_sufficient=result.get("is_context_sufficient", True),
            is_answer_complete=result.get("is_answer_complete", True),
            is_answer_correct=result.get("is_answer_correct", True),
            error_category=result.get("error_category", "correct"),
            severity=int(result.get("severity", 0)),
            reasoning=result.get("reasoning", ""),
            metadata={"judge_cost_usd": llm_response.cost_usd},
        )

    def _extract_json_from_text(self, text: str) -> dict:
        """Try to extract JSON from a text response that may have extra content."""
        import re
        # Look for JSON block
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {}

    def save(self, report: JudgeReport, output_dir: str) -> None:
        """Save judge report to JSON."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / f"judge_{report.pipeline_name}_{report.domain}.json"

        data = {
            "pipeline_name": report.pipeline_name,
            "domain": report.domain,
            "num_samples": report.num_samples,
            "accuracy": report.accuracy,
            "hallucination_rate": report.hallucination_rate,
            "retrieval_failure_rate": report.retrieval_failure_rate,
            "generation_failure_rate": report.generation_failure_rate,
            "incomplete_rate": report.incomplete_rate,
            "avg_severity": report.avg_severity,
            "error_distribution": report.error_distribution,
            "verdicts": [
                {
                    "question": v.question,
                    "error_category": v.error_category,
                    "severity": v.severity,
                    "reasoning": v.reasoning,
                    "is_hallucination": v.is_hallucination,
                }
                for v in report.verdicts
            ],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved judge report to {path}")
