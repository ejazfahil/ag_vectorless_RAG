"""
Multi-Agent Advocate Courtroom Debate Swarm Engine.
────────────────────────────────────────────────────
Orchestrates turn-by-turn debates between different RAG paradigms (BM25, PageIndex, Embedding-Free)
under the supervision of a Presiding Judge (Qwen3-8B) with a real-time Faithfulness Scorer (Llama3.2-3B).
Logs full transcripts, queries, contexts, and grounding scores to the shared SQLite database.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import yaml
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.bm25_rag import BM25RAG
from src.pipelines.embedding_free_rag import EmbeddingFreeRAG
from src.pipelines.pageindex_rag import PageIndexRAG
from src.utils.llm_client import LLMClient

# Graceful DB logging integration with the DB engineer's DatabaseManager
try:
    from src.utils.database import db_manager
    DB_LOGGING_ENABLED = True
except ImportError as e:
    logger.warning(f"Could not import database logging helpers: {e}. SQLite logging disabled.")
    DB_LOGGING_ENABLED = False

# System prompts for advocates
ADVOCATE_SYSTEM_PROMPTS = {
    "bm25": (
        "You are a professional legal-style courtroom advocate representing the BM25 Lexical RAG system. "
        "You rely on precise keyword matching and exact term references. Highlight the reliability of lexical facts."
    ),
    "pageindex": (
        "You are a professional legal-style courtroom advocate representing the PageIndex Hierarchical Tree RAG system. "
        "You navigate document indexes systematically and refer to hierarchical structures, chapters, and exact page ranges."
    ),
    "embedding_free": (
        "You are a professional legal-style courtroom advocate representing the Embedding-Free RAG system. "
        "You rely on exact verbatim quote extraction and Levenshtein distance anchoring to verify your claims."
    )
}


class CourtroomDebate:
    """
    Manages the multi-turn debate sequence between the RAG advocates,
    mediated by a Presiding Judge.
    """

    def __init__(
        self,
        domain: str,
        topic: str,
        judge_model: str = "qwen3:8b",
        advocate_model: str = "llama3.2:3b",
        db_path: str | None = None
    ):
        self.domain = domain
        self.topic = topic
        self.judge_model = judge_model
        self.advocate_model = advocate_model
        self.db_path = db_path

        # Configure custom db path if provided
        if DB_LOGGING_ENABLED and self.db_path:
            db_manager.db_path = Path(self.db_path)
            db_manager.init_db()

        # Initialize LLM clients
        self.judge_client = LLMClient(model=self.judge_model, temperature=0.2, timeout=300)
        self.eval_client = LLMClient(model=self.advocate_model, temperature=0.0)
        self.advocate_client = LLMClient(model=self.advocate_model, temperature=0.3)

        self.corpus_path = str(PROJECT_ROOT / "data" / "processed" / domain)

        # Load configurations and override models
        bm25_cfg = self._load_and_override_config(PROJECT_ROOT / "configs" / "bm25.yaml")
        pageindex_cfg = self._load_and_override_config(PROJECT_ROOT / "configs" / "pageindex.yaml")
        ef_cfg = self._load_and_override_config(PROJECT_ROOT / "configs" / "embedding_free.yaml")

        # Instantiate pipelines
        logger.info("Initializing RAG pipelines for advocates...")
        self.pipelines = {
            "bm25": BM25RAG(bm25_cfg),
            "pageindex": PageIndexRAG(pageindex_cfg),
            "embedding_free": EmbeddingFreeRAG(ef_cfg)
        }

    def _load_and_override_config(self, path: Path) -> dict:
        """Load YAML config and dynamically inject advocate model."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at {path}")

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # Enforce that all advocate pipeline internal LLMs use the advocate model
        if "pipeline" in config:
            pipeline = config["pipeline"]
            if "generation" in pipeline and isinstance(pipeline["generation"], dict):
                pipeline["generation"]["model"] = self.advocate_model
            if "ingestion" in pipeline and isinstance(pipeline["ingestion"], dict):
                pipeline["ingestion"]["model"] = self.advocate_model
            if "retrieval" in pipeline and isinstance(pipeline["retrieval"], dict):
                pipeline["retrieval"]["model"] = self.advocate_model
            if "quote_extraction" in pipeline and isinstance(pipeline["quote_extraction"], dict):
                pipeline["quote_extraction"]["model"] = self.advocate_model
            if "answer_generation" in pipeline and isinstance(pipeline["answer_generation"], dict):
                pipeline["answer_generation"]["model"] = self.advocate_model

        return config

    def ingest_corpus(self) -> None:
        """Prepare the data indices for all pipelines."""
        logger.info(f"Ingesting domain corpus '{self.domain}' from {self.corpus_path} into RAG pipelines...")
        for name, pipeline in self.pipelines.items():
            logger.info(f"Ingesting into {name} RAG index...")
            try:
                report = pipeline.ingest(self.corpus_path)
                logger.info(f"Ingestion complete for {name}. Documents: {report.num_documents}")
            except Exception as e:
                logger.exception(f"Ingestion failed for pipeline {name}: {e}")

    def _evaluate_faithfulness(self, argument: str, context: str) -> float:
        """Run LLM-as-a-judge prompt locally to rate claim faithfulness against retrieved context."""
        if not context:
            return 0.0

        prompt = f"""You are a rigorous factual auditor. Your task is to evaluate the faithfulness of a RAG advocate's argument based ONLY on the retrieved context they had access to.

Advocate Argument:
{argument}

Retrieved Context:
{context}

Evaluate if every single claim in the argument is 100% supported by the retrieved context. If there are any extrapolations, unsubstantiated claims, or contradictions, penalize the score.

Provide your evaluation in the following JSON format:
{{
  "reasoning": "Explain step-by-step why the argument is or is not fully grounded in the context.",
  "score": <float between 0.0 and 1.0>
}}

Return ONLY the valid JSON object. No other text."""

        try:
            resp = self.eval_client.generate(prompt, json_mode=True)
            data = json.loads(resp.content)
            score = float(data.get("score", 0.5))
            score = max(0.0, min(1.0, score))
            logger.info(f"Faithfulness score: {score:.2f} | Reasoning: {data.get('reasoning', 'N/A')}")
            return score
        except Exception as e:
            logger.warning(f"Faithfulness LLM evaluation failed: {e}. Defaulting to 0.5.")
            return 0.5

    def _generate_introduction(self) -> str:
        """Get Judge opening courtroom remarks."""
        prompt = f"""You are the Presiding Judge (Qwen3-8B) in the Advocate Courtroom Swarm.
Domain of Today's Trial: {self.domain.upper()}
Topic under Debate: '{self.topic}'

Deliver a majestic, formal courtroom address to call the court to order.
Introduce the topic, outline the rules of this debate, and present the three RAG advocates who will compete:
1. BM25 Lexical RAG (Lexical/Keyword paradigm)
2. PageIndex RAG (Hierarchical tree navigation paradigm)
3. Embedding-Free RAG (Verbatim quote extraction paradigm)

Ask each advocate to present their opening argument and ground it strictly in the evidence from their index.
Return ONLY your formal speech. Do not include introductory notes or chat prefix/suffix."""

        resp = self.judge_client.generate(prompt)
        return resp.content

    def _generate_rebuttal_call(self, turn1_transcripts: dict) -> str:
        """Get Judge rebuttal directive."""
        transcripts_str = ""
        for p, data in turn1_transcripts.items():
            transcripts_str += f"\n### {p.upper()} Advocate:\n{data['argument']}\n(Faithfulness Score: {data['score']:.2f})\n"

        prompt = f"""You are the Presiding Judge (Qwen3-8B) in the Advocate Courtroom Swarm.
Topic: '{self.topic}'

Here is the transcript of Turn 1 (Opening Statements):
{transcripts_str}

Deliver a majestic courtroom address calling the court to order for Turn 2: Rebuttal & Cross-Examination.
Summarize the main lines of argument and their contentions. Address each advocate specifically, highlight weaknesses or contradictions in their claims compared to their opponents', and prompt them to retrieve counter-evidence and challenge their opponents.
Return ONLY your formal speech in clean Markdown."""

        resp = self.judge_client.generate(prompt)
        return resp.content

    def _generate_closing_call(self, turn2_transcripts: dict) -> str:
        """Get Judge closing arguments call."""
        transcripts_str = ""
        for p, data in turn2_transcripts.items():
            transcripts_str += f"\n### {p.upper()} Advocate Rebuttal:\n{data['argument']}\n(Faithfulness Score: {data['score']:.2f})\n"

        prompt = f"""You are the Presiding Judge (Qwen3-8B) in the Advocate Courtroom Swarm.
Topic: '{self.topic}'

Here is the transcript of Turn 2 (Rebuttals):
{transcripts_str}

Deliver a courtroom address calling for Turn 3: Closing Arguments.
Direct the advocates to present their final summaries, synthesize their strongest factual evidence, and make their ultimate case for why their RAG paradigm retrieved the most robust and faithful truth.
Return ONLY your formal speech in clean Markdown."""

        resp = self.judge_client.generate(prompt)
        return resp.content

    def _generate_verdict(self, full_transcript: str) -> tuple[str, str, str]:
        """Synthesize final judgment, rate advocates and choose winner."""
        prompt = f"""You are the Presiding Judge (Qwen3-8B) delivering the final verdict in the Advocate Courtroom Swarm.
Domain of Trial: {self.domain}
Topic under Debate: '{self.topic}'

Here is the complete trial record of the debate:

{full_transcript}

Deliver a comprehensive, majestic, and analytical final verdict in highly stylized legal Markdown.
Your verdict must include:
1. COURT DECISION & SCORECARD: An elegant Markdown table rating each advocate (BM25, PageIndex, Embedding-Free RAG) from 0.0 to 10.0 on: Factual Grounding (Faithfulness), Citation Accuracy, Logical Reasoning, and Overall.
2. CRITICAL EVALUATION: A detailed analysis of each advocate's arguments across all three turns, citing specific points they made and verifying their grounding based on their faithfulness scores.
3. VERDICT & WINNER DECLARATION: A formal declaration of the winning RAG paradigm based on who retrieved the most robust, accurate, and faithful truth.

At the very end of your response, write a separate section with ONLY this JSON block:
```json
{{
  "winner": "<BM25 | PageIndex | Embedding-Free>",
  "summary": "<A 2-3 sentence high-level summary of why the winner won.>"
}}
```
Make sure the JSON is perfectly valid and matches the format exactly. No text after the JSON."""

        resp = self.judge_client.generate(prompt)
        verdict_content = resp.content

        winner = "BM25"
        summary = "No summary generated."

        try:
            # Find JSON block in output
            json_match = re.search(r'```json\s*(.*?)\s*```', verdict_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
            else:
                json_match = re.search(r'(\{.*?\})\s*$', verdict_content, re.DOTALL)
                json_str = json_match.group(1).strip() if json_match else "{}"

            data = json.loads(json_str)
            winner = data.get("winner", "BM25")
            summary = data.get("summary", "No summary generated.")
        except Exception as e:
            logger.warning(f"Could not parse verdict JSON metadata: {e}. Performing heuristic extraction.")
            if "pageindex" in verdict_content.lower():
                winner = "PageIndex"
            elif "embedding-free" in verdict_content.lower():
                winner = "Embedding-Free"
            else:
                winner = "BM25"
            summary = "The debate was won based on factual grounding and logic."

        return verdict_content, winner, summary

    def _execute_advocate_turn(
        self,
        paradigm: str,
        query: str,
        system_prompt: str,
        turn_num: int,
        history: str = ""
    ) -> dict:
        """Executes RAG context retrieval and advocate statement formulation."""
        pipeline = self.pipelines[paradigm]
        logger.info(f"Advocate {paradigm.upper()} retrieving evidence with query: '{query}'")

        retrieved_context = ""
        citations = []

        try:
            rag_response = pipeline.query(query)
            retrieved_context = "\n\n---\n\n".join(rag_response.retrieved_contexts)
            citations = rag_response.source_references
        except Exception as e:
            logger.exception(f"RAG query failed for advocate {paradigm}: {e}. Continuing with fallback context.")
            retrieved_context = "No context retrieved due to pipeline runtime error."
            citations = []

        citations_str = json.dumps(citations, indent=2)

        # Turn-specific prompt construction
        if turn_num == 1:
            prompt = f"""You are the {paradigm.upper()} RAG Advocate. The topic is '{self.topic}'.
Your pipeline has retrieved the following context:
{retrieved_context}

Source references:
{citations_str}

Formulate a highly persuasive, factual, and premium opening statement.
- State your position clearly.
- Cite specific document IDs, titles, section names, or page numbers explicitly.
- Ground your argument 100% in the retrieved context. Do not make up facts.
- Use Markdown styling to make your presentation professional and authoritative.

Begin your speech immediately:"""
        elif turn_num == 2:
            prompt = f"""You are the {paradigm.upper()} RAG Advocate. The topic is '{self.topic}'.

History of debate so far:
{history}

Your pipeline has retrieved the following rebuttal context:
{retrieved_context}

Source references:
{citations_str}

Formulate a sharp, fact-based Rebuttal.
- Point out specific inconsistencies or gaps in your opponents' opening statements.
- Support your refutations with the retrieved context.
- Cite specific documents and sections explicitly.
- Use elegant Markdown formatting.

Begin your speech immediately:"""
        else:
            prompt = f"""You are the {paradigm.upper()} RAG Advocate. The topic is '{self.topic}'.

History of debate so far:
{history}

Your pipeline has retrieved the following final context:
{retrieved_context}

Source references:
{citations_str}

Formulate a masterly Closing Argument.
- Summarize your strongest points and factual evidence.
- Make a final appeal to the Judge on why your paradigm has retrieved the most faithful and accurate truth.
- Cite specific sources explicitly.
- Use elegant Markdown formatting.

Begin your speech immediately:"""

        try:
            resp = self.advocate_client.generate(prompt, system_prompt=system_prompt)
            argument = resp.content
        except Exception as e:
            logger.exception(f"Failed to generate speech for advocate {paradigm}: {e}")
            argument = f"Advocate {paradigm} presents evidence in support of the topic, referring to retrieved documents."

        # Evaluate factual faithfulness of speech
        faithfulness_score = self._evaluate_faithfulness(argument, retrieved_context)

        return {
            "query": query,
            "context": retrieved_context,
            "argument": argument,
            "citations": citations,
            "score": faithfulness_score
        }

    def _update_db_verdict(self, db_id: int, verdict: str, summary: str) -> None:
        """Update the debates table with the final verdict and summary."""
        try:
            conn = sqlite3.connect(db_manager.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE debates SET verdict = ?, summary = ? WHERE id = ?",
                (verdict, summary, db_id)
            )
            conn.commit()
            conn.close()
            logger.info(f"Successfully updated final verdict in DB for debate ID: {db_id}")
        except Exception as e:
            logger.error(f"Failed to update verdict in SQLite: {e}")

    def run_debate(self) -> dict:
        """Run the complete 3-turn multi-agent courtroom debate."""
        db_id = None
        if DB_LOGGING_ENABLED:
            try:
                # Insert parent debate record initially (verdict & summary are updated at the end)
                db_id = db_manager.insert_debate({
                    "topic": self.topic,
                    "judge_model": self.judge_model,
                    "verdict": None,
                    "summary": None
                })
                logger.info(f"Logged debate session initialization to database (DB ID: {db_id})")
            except Exception as e:
                logger.error(f"Failed to log debate initialization: {e}")

        logger.info(f"--- DEBATE START | ID: {db_id} | Domain: {self.domain} | Topic: {self.topic} ---")

        # Step 0: Ingest corpus
        self.ingest_corpus()

        db_turn_idx = 0
        transcript_history = []

        # ==========================================
        # INTRODUCTION
        # ==========================================
        logger.info("Presiding Judge opening the court...")
        intro_text = self._generate_introduction()
        print("\n\n=== COURTROOM DEBATE INTRODUCTION ===")
        print(intro_text)

        if DB_LOGGING_ENABLED and db_id is not None:
            try:
                db_manager.insert_debate_turn({
                    "debate_id": db_id,
                    "turn_index": db_turn_idx,
                    "speaker": "Presiding Judge",
                    "paradigm": "Judge",
                    "query_used": None,
                    "retrieved_context": None,
                    "argument": intro_text,
                    "citations": None,
                    "faithfulness_score": None
                })
            except Exception as e:
                logger.error(f"Failed to log Intro turn: {e}")
        db_turn_idx += 1
        transcript_history.append(f"### Presiding Judge (Introduction):\n{intro_text}\n")

        # ==========================================
        # TURN 1: OPENING STATEMENTS
        # ==========================================
        logger.info("Executing Turn 1: Opening Statements...")
        turn1_results = {}

        for name in ["bm25", "pageindex", "embedding_free"]:
            query_prompt = f"""You are the {name.upper()} RAG Advocate. The topic is '{self.topic}'.
Generate a targeted search query tailored specifically for your retrieval mechanism to find the strongest opening evidence on the topic.
- BM25 uses lexical keywords.
- PageIndex navigates an index tree.
- Embedding-Free extracts specific verbatim quotes.

Return ONLY the raw search query. No other text."""
            q_resp = self.eval_client.generate(query_prompt)
            query = q_resp.content.strip().strip('"').strip("'")

            res = self._execute_advocate_turn(
                paradigm=name,
                query=query,
                system_prompt=ADVOCATE_SYSTEM_PROMPTS[name],
                turn_num=1
            )
            turn1_results[name] = res

            print(f"\n--- {name.upper()} Advocate (Opening Statement) ---")
            print(res["argument"])
            print(f"Faithfulness Score: {res['score']:.2f}")

            if DB_LOGGING_ENABLED and db_id is not None:
                try:
                    db_manager.insert_debate_turn({
                        "debate_id": db_id,
                        "turn_index": db_turn_idx,
                        "speaker": f"{name.upper()} Advocate",
                        "paradigm": name,
                        "query_used": res["query"],
                        "retrieved_context": res["context"],
                        "argument": res["argument"],
                        "citations": res["citations"],
                        "faithfulness_score": res["score"]
                    })
                except Exception as e:
                    logger.error(f"Failed to log Turn 1 for {name}: {e}")
            db_turn_idx += 1
            transcript_history.append(
                f"### {name.upper()} Advocate (Turn 1 Opening):\n{res['argument']}\n"
                f"*(Faithfulness Score: {res['score']:.2f})*\n"
            )

        # ==========================================
        # TURN 2: REBUTTALS
        # ==========================================
        logger.info("Executing Turn 2: Rebuttals...")
        rebuttal_call = self._generate_rebuttal_call(turn1_results)
        print("\n=== COURTROOM DEBATE TURN 2: REBUTTALS & CROSS-EXAMINATION ===")
        print(rebuttal_call)

        if DB_LOGGING_ENABLED and db_id is not None:
            try:
                db_manager.insert_debate_turn({
                    "debate_id": db_id,
                    "turn_index": db_turn_idx,
                    "speaker": "Presiding Judge",
                    "paradigm": "Judge",
                    "query_used": None,
                    "retrieved_context": None,
                    "argument": rebuttal_call,
                    "citations": None,
                    "faithfulness_score": None
                })
            except Exception as e:
                logger.error(f"Failed to log Rebuttal Call turn: {e}")
        db_turn_idx += 1
        transcript_history.append(f"### Presiding Judge (Call for Rebuttal):\n{rebuttal_call}\n")

        turn2_results = {}
        for name in ["bm25", "pageindex", "embedding_free"]:
            opponents = [p for p in ["bm25", "pageindex", "embedding_free"] if p != name]
            opponents_args = "\n\n".join(
                f"--- {opp.upper()} Opening Statement ---\n{turn1_results[opp]['argument']}" for opp in opponents
            )

            query_prompt = f"""You are the {name.upper()} RAG Advocate. The topic is '{self.topic}'.
Your opening statement was:
{turn1_results[name]['argument']}

Your opponents argued:
{opponents_args}

The Judge directed:
{rebuttal_call}

Generate a new, highly targeted search query to find counter-evidence or details in the corpus that contradict or fill gaps in the opponents' claims.
Return ONLY the raw search query."""
            q_resp = self.eval_client.generate(query_prompt)
            query = q_resp.content.strip().strip('"').strip("'")

            res = self._execute_advocate_turn(
                paradigm=name,
                query=query,
                system_prompt=ADVOCATE_SYSTEM_PROMPTS[name],
                turn_num=2,
                history=f"=== Turn 1 Opening Statement ===\n{turn1_results[name]['argument']}\n\n=== Opponents Statements ===\n{opponents_args}"
            )
            turn2_results[name] = res

            print(f"\n--- {name.upper()} Advocate (Rebuttal) ---")
            print(res["argument"])
            print(f"Faithfulness Score: {res['score']:.2f}")

            if DB_LOGGING_ENABLED and db_id is not None:
                try:
                    db_manager.insert_debate_turn({
                        "debate_id": db_id,
                        "turn_index": db_turn_idx,
                        "speaker": f"{name.upper()} Advocate",
                        "paradigm": name,
                        "query_used": res["query"],
                        "retrieved_context": res["context"],
                        "argument": res["argument"],
                        "citations": res["citations"],
                        "faithfulness_score": res["score"]
                    })
                except Exception as e:
                    logger.error(f"Failed to log Turn 2 rebuttal for {name}: {e}")
            db_turn_idx += 1
            transcript_history.append(
                f"### {name.upper()} Advocate (Turn 2 Rebuttal):\n{res['argument']}\n"
                f"*(Faithfulness Score: {res['score']:.2f})*\n"
            )

        # ==========================================
        # TURN 3: CLOSING ARGUMENTS
        # ==========================================
        logger.info("Executing Turn 3: Closing Arguments...")
        closing_call = self._generate_closing_call(turn2_results)
        print("\n=== COURTROOM DEBATE TURN 3: CLOSING ARGUMENTS ===")
        print(closing_call)

        if DB_LOGGING_ENABLED and db_id is not None:
            try:
                db_manager.insert_debate_turn({
                    "debate_id": db_id,
                    "turn_index": db_turn_idx,
                    "speaker": "Presiding Judge",
                    "paradigm": "Judge",
                    "query_used": None,
                    "retrieved_context": None,
                    "argument": closing_call,
                    "citations": None,
                    "faithfulness_score": None
                })
            except Exception as e:
                logger.error(f"Failed to log Closing Call turn: {e}")
        db_turn_idx += 1
        transcript_history.append(f"### Presiding Judge (Call for Closing):\n{closing_call}\n")

        turn3_results = {}
        for name in ["bm25", "pageindex", "embedding_free"]:
            history_summary = (
                f"=== Your Turn 1 Opening ===\n{turn1_results[name]['argument']}\n\n"
                f"=== Your Turn 2 Rebuttal ===\n{turn2_results[name]['argument']}\n"
            )

            query_prompt = f"""You are the {name.upper()} RAG Advocate. The topic is '{self.topic}'.
Generate a final search query to fetch the ultimate pieces of context to wrap up your case on this topic.
Return ONLY the raw search query."""
            q_resp = self.eval_client.generate(query_prompt)
            query = q_resp.content.strip().strip('"').strip("'")

            res = self._execute_advocate_turn(
                paradigm=name,
                query=query,
                system_prompt=ADVOCATE_SYSTEM_PROMPTS[name],
                turn_num=3,
                history=history_summary
            )
            turn3_results[name] = res

            print(f"\n--- {name.upper()} Advocate (Closing Argument) ---")
            print(res["argument"])
            print(f"Faithfulness Score: {res['score']:.2f}")

            if DB_LOGGING_ENABLED and db_id is not None:
                try:
                    db_manager.insert_debate_turn({
                        "debate_id": db_id,
                        "turn_index": db_turn_idx,
                        "speaker": f"{name.upper()} Advocate",
                        "paradigm": name,
                        "query_used": res["query"],
                        "retrieved_context": res["context"],
                        "argument": res["argument"],
                        "citations": res["citations"],
                        "faithfulness_score": res["score"]
                    })
                except Exception as e:
                    logger.error(f"Failed to log Turn 3 closing for {name}: {e}")
            db_turn_idx += 1
            transcript_history.append(
                f"### {name.upper()} Advocate (Turn 3 Closing):\n{res['argument']}\n"
                f"*(Faithfulness Score: {res['score']:.2f})*\n"
            )

        # ==========================================
        # VERDICT
        # ==========================================
        logger.info("Delivering final judicial verdict...")
        full_transcript = "\n\n".join(transcript_history)
        verdict, winner, summary = self._generate_verdict(full_transcript)

        print("\n=== FINAL JUDICIAL VERDICT ===")
        print(verdict)

        if DB_LOGGING_ENABLED and db_id is not None:
            try:
                db_manager.insert_debate_turn({
                    "debate_id": db_id,
                    "turn_index": db_turn_idx,
                    "speaker": "Presiding Judge (Verdict)",
                    "paradigm": "Judge",
                    "query_used": None,
                    "retrieved_context": None,
                    "argument": verdict,
                    "citations": None,
                    "faithfulness_score": None
                })
                # Update the parent record in `debates` table with the actual verdict and summary
                self._update_db_verdict(db_id, verdict, summary)
            except Exception as e:
                logger.error(f"Failed to update final verdict / turn: {e}")

        logger.info(f"--- DEBATE COMPLETED | Winner: {winner} | ID: {db_id} ---")

        return {
            "debate_id": db_id,
            "winner": winner,
            "summary": summary,
            "verdict": verdict,
            "turn1": turn1_results,
            "turn2": turn2_results,
            "turn3": turn3_results
        }


def main():
    parser = argparse.ArgumentParser(description="Advocate Courtroom Swarm Engine CLI")
    parser.add_argument(
        "--domain",
        type=str,
        default="finance",
        choices=["finance", "legal", "technical"],
        help="Domain to run the debate on"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="Trade War Tariff Impacts on Global Tech Supply Chain",
        help="Topic under debate"
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="qwen3:8b",
        help="Model name for the Presiding Judge"
    )
    parser.add_argument(
        "--advocate",
        type=str,
        default="llama3.2:3b",
        help="Model name for the Advocates"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Custom database file path"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("        ADVOCATE COURTROOM SWARM ENGINE")
    print("=" * 60)
    print(f"  Domain:    {args.domain}")
    print(f"  Topic:     {args.topic}")
    print(f"  Judge:     {args.judge}")
    print(f"  Advocate:  {args.advocate}")
    print("=" * 60)

    try:
        debate = CourtroomDebate(
            domain=args.domain,
            topic=args.topic,
            judge_model=args.judge,
            advocate_model=args.advocate,
            db_path=args.db_path
        )
        debate.run_debate()
    except Exception as e:
        logger.exception(f"Debate engine failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
