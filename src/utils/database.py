"""
Database utilities — manages SQLite connection, schema initialization, and CRUD operations.
Tracks runs, queries, debates, and debate turns.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from loguru import logger

# Project root calculation
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "vectorless_rag.db"


class DatabaseManager:
    """Manages connection, schema initialization, and transactional CRUD operations for SQLite."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        # Ensure parent directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Establish a connection with a timeout of 30 seconds and Row factory enabled."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # Enable foreign key support in SQLite
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Initialize all schema tables if they do not already exist."""
        runs_sql = """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            pipeline_name TEXT NOT NULL,
            domain TEXT NOT NULL,
            num_questions INTEGER,
            success_rate REAL,
            mean_latency REAL,
            p50_latency REAL,
            p95_latency REAL,
            peak_rss REAL,
            total_tokens INTEGER,
            total_cost REAL
        );
        """

        queries_sql = """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            pipeline_name TEXT NOT NULL,
            domain TEXT NOT NULL,
            answer TEXT,
            retrieved_contexts TEXT, -- JSON-serialized list of strings
            reference_answer TEXT,
            question_type TEXT,
            latency REAL,
            mem_delta REAL,
            mem_peak REAL,
            input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER,
            cost REAL,
            error_type TEXT,
            success INTEGER, -- 0 or 1
            faithfulness_score REAL,
            f1_score REAL,
            em_score REAL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        );
        """

        debates_sql = """
        CREATE TABLE IF NOT EXISTS debates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            topic TEXT NOT NULL,
            judge_model TEXT,
            verdict TEXT,
            summary TEXT
        );
        """

        debate_turns_sql = """
        CREATE TABLE IF NOT EXISTS debate_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id INTEGER,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            paradigm TEXT,
            query_used TEXT,
            retrieved_context TEXT,
            argument TEXT,
            citations TEXT, -- JSON-serialized list or dict
            faithfulness_score REAL,
            FOREIGN KEY (debate_id) REFERENCES debates(id) ON DELETE CASCADE
        );
        """

        try:
            with self.get_connection() as conn:
                conn.execute(runs_sql)
                conn.execute(queries_sql)
                conn.execute(debates_sql)
                conn.execute(debate_turns_sql)
                conn.commit()
            logger.debug(f"Database initialized successfully at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise

    def insert_run(self, data: dict[str, Any]) -> None:
        """Insert or update a benchmark run metadata record."""
        timestamp = data.get("timestamp") or datetime.utcnow().isoformat()
        sql = """
        INSERT INTO runs (
            id, timestamp, pipeline_name, domain, num_questions,
            success_rate, mean_latency, p50_latency, p95_latency,
            peak_rss, total_tokens, total_cost
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            timestamp = excluded.timestamp,
            pipeline_name = excluded.pipeline_name,
            domain = excluded.domain,
            num_questions = excluded.num_questions,
            success_rate = excluded.success_rate,
            mean_latency = excluded.mean_latency,
            p50_latency = excluded.p50_latency,
            p95_latency = excluded.p95_latency,
            peak_rss = excluded.peak_rss,
            total_tokens = excluded.total_tokens,
            total_cost = excluded.total_cost
        """
        params = (
            data.get("id"),
            timestamp,
            data.get("pipeline_name"),
            data.get("domain"),
            data.get("num_questions"),
            data.get("success_rate"),
            data.get("mean_latency"),
            data.get("p50_latency"),
            data.get("p95_latency"),
            data.get("peak_rss"),
            data.get("total_tokens"),
            data.get("total_cost"),
        )
        try:
            with self.get_connection() as conn:
                conn.execute(sql, params)
                conn.commit()
            logger.debug(f"Successfully logged run {data.get('id')} to DB.")
        except Exception as e:
            logger.error(f"Error inserting run record into DB: {e}")
            raise

    def insert_query(self, data: dict[str, Any]) -> int:
        """Insert a single query telemetry record."""
        timestamp = data.get("timestamp") or datetime.utcnow().isoformat()

        # Serialize retrieved_contexts as JSON if it's a list
        contexts = data.get("retrieved_contexts")
        if isinstance(contexts, list):
            contexts_str = json.dumps(contexts)
        else:
            contexts_str = contexts

        sql = """
        INSERT INTO queries (
            run_id, timestamp, question, pipeline_name, domain, answer,
            retrieved_contexts, reference_answer, question_type, latency,
            mem_delta, mem_peak, input_tokens, output_tokens, total_tokens,
            cost, error_type, success, faithfulness_score, f1_score, em_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            data.get("run_id"),
            timestamp,
            data.get("question"),
            data.get("pipeline_name"),
            data.get("domain"),
            data.get("answer"),
            contexts_str,
            data.get("reference_answer"),
            data.get("question_type"),
            data.get("latency"),
            data.get("mem_delta"),
            data.get("mem_peak"),
            data.get("input_tokens"),
            data.get("output_tokens"),
            data.get("total_tokens"),
            data.get("cost"),
            data.get("error_type"),
            1 if data.get("success", True) else 0,
            data.get("faithfulness_score"),
            data.get("f1_score"),
            data.get("em_score"),
        )
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                last_row_id = cursor.lastrowid
                return last_row_id if last_row_id is not None else 0
        except Exception as e:
            logger.error(f"Error inserting query telemetry record: {e}")
            raise

    def insert_debate(self, data: dict[str, Any]) -> int:
        """Insert a debate session record."""
        timestamp = data.get("timestamp") or datetime.utcnow().isoformat()
        sql = """
        INSERT INTO debates (
            timestamp, topic, judge_model, verdict, summary
        ) VALUES (?, ?, ?, ?, ?)
        """
        params = (
            timestamp,
            data.get("topic"),
            data.get("judge_model"),
            data.get("verdict"),
            data.get("summary"),
        )
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                last_row_id = cursor.lastrowid
                return last_row_id if last_row_id is not None else 0
        except Exception as e:
            logger.error(f"Error inserting debate record: {e}")
            raise

    def insert_debate_turn(self, data: dict[str, Any]) -> int:
        """Insert a turn within a debate session."""
        citations = data.get("citations")
        if isinstance(citations, (list, dict)):
            citations_str = json.dumps(citations)
        else:
            citations_str = citations

        sql = """
        INSERT INTO debate_turns (
            debate_id, turn_index, speaker, paradigm, query_used,
            retrieved_context, argument, citations, faithfulness_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            data.get("debate_id"),
            data.get("turn_index"),
            data.get("speaker"),
            data.get("paradigm"),
            data.get("query_used"),
            data.get("retrieved_context"),
            data.get("argument"),
            citations_str,
            data.get("faithfulness_score"),
        )
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)
                conn.commit()
                last_row_id = cursor.lastrowid
                return last_row_id if last_row_id is not None else 0
        except Exception as e:
            logger.error(f"Error inserting debate turn record: {e}")
            raise

    def get_runs(self) -> list[dict[str, Any]]:
        """Fetch all run records, sorted by timestamp descending."""
        sql = "SELECT * FROM runs ORDER BY timestamp DESC"
        try:
            with self.get_connection() as conn:
                rows = conn.execute(sql).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching runs: {e}")
            return []

    def get_queries(self, run_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Fetch query records, optionally filtered by run_id, sorted by timestamp ascending."""
        if run_id:
            sql = "SELECT * FROM queries WHERE run_id = ? ORDER BY timestamp ASC"
            params = (run_id,)
        else:
            sql = "SELECT * FROM queries ORDER BY timestamp ASC"
            params = ()

        try:
            with self.get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    # Convert BOOLEAN success back to bool
                    d["success"] = bool(d["success"])
                    if d.get("retrieved_contexts"):
                        try:
                            d["retrieved_contexts"] = json.loads(d["retrieved_contexts"])
                        except Exception:
                            pass
                    results.append(d)
                return results
        except Exception as e:
            logger.error(f"Error fetching queries: {e}")
            return []

    def get_debates(self) -> list[dict[str, Any]]:
        """Fetch all debate sessions, sorted by timestamp descending."""
        sql = "SELECT * FROM debates ORDER BY timestamp DESC"
        try:
            with self.get_connection() as conn:
                rows = conn.execute(sql).fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching debates: {e}")
            return []

    def get_debate_turns(self, debate_id: int) -> list[dict[str, Any]]:
        """Fetch all turns for a specific debate, sorted by turn_index ascending."""
        sql = "SELECT * FROM debate_turns WHERE debate_id = ? ORDER BY turn_index ASC"
        try:
            with self.get_connection() as conn:
                rows = conn.execute(sql, (debate_id,)).fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    if d.get("citations"):
                        try:
                            d["citations"] = json.loads(d["citations"])
                        except Exception:
                            pass
                    results.append(d)
                return results
        except Exception as e:
            logger.error(f"Error fetching debate turns: {e}")
            return []


# Global convenience instance
db_manager = DatabaseManager()
