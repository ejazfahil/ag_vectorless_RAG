#!/usr/bin/env python3
"""
Answer-quality benchmark — complements ``run_benchmark.py`` (which tracks
latency/tokens/cost telemetry) by scoring *correctness*.

For a seeded sample of the golden Q&A set it runs each selected pipeline plus a
no-retrieval **closed-book baseline** (same LLM, no context) and scores every
answer against the ground truth with token-F1, exact-match, and answer-match
(does the normalised gold string appear in the answer). The gap between a
pipeline and the closed-book baseline is the value retrieval actually adds.

Every number written to ``results/`` comes from a real run — nothing estimated.
Local pipelines (BM25, closed-book) run free on Ollama; heavier pipelines and
frontier models are driven by the same command via ``--model`` / provider env.

Examples
--------
    # Real local baseline (free, Ollama):
    python -m scripts.quality_benchmark --pipelines bm25,closed_book --n 20 --seed 42

    # Frontier model (paid — populates API columns):
    python -m scripts.quality_benchmark --pipelines bm25,closed_book --model gpt-4o --full
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.string_metrics import exact_match, normalize_answer, token_f1  # noqa: E402
from src.utils.llm_client import LLMClient  # noqa: E402

# name -> (class attr in src.pipelines, config file). "closed_book" is handled inline.
PIPELINES = {
    "bm25": ("BM25RAG", "configs/bm25.yaml"),
    "pageindex": ("PageIndexRAG", "configs/pageindex.yaml"),
    "embedding_free": ("EmbeddingFreeRAG", "configs/embedding_free.yaml"),
    "agentic": ("AgenticRAG", "configs/agentic.yaml"),
    "roaming": ("RoamingRAG", "configs/roaming.yaml"),
    "hybrid_sota": ("HybridSoTARAG", "configs/hybrid_sota.yaml"),
    "three_stage_hybrid": ("ThreeStageHybridRAG", "configs/three_stage_hybrid.yaml"),
}


def load_full_config(cfg_file: str, model: str) -> dict:
    """Load a pipeline YAML and force the generation model for reproducible local runs."""
    cfg = yaml.safe_load((REPO_ROOT / cfg_file).read_text())
    cfg.setdefault("pipeline", {}).setdefault("generation", {})["model"] = model
    return cfg


def load_questions(domain: str, n: int, seed: int, use_full: bool) -> list[dict]:
    path = REPO_ROOT / f"data/golden_qa/{domain}_golden_qa.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if use_full or n >= len(rows):
        return rows
    return random.Random(seed).sample(rows, n)


class ClosedBookBaseline:
    """No-retrieval control: the same LLM answers from parametric memory only."""

    def __init__(self, model: str):
        self._client = LLMClient(model=model, temperature=0.0)
        self.provider = self._client.provider

    def query(self, question: str) -> dict:
        start = time.perf_counter()
        resp = self._client.generate(
            prompt=question,
            system_prompt="Answer the question as precisely as possible. If you do not know, say so.",
        )
        return {"answer": resp.content, "latency_ms": (time.perf_counter() - start) * 1000,
                "tokens": resp.total_tokens, "cost_usd": resp.cost_usd}


def build_pipeline(name: str, model: str):
    if name == "closed_book":
        return ClosedBookBaseline(model)
    import src.pipelines as P

    attr, cfg_file = PIPELINES[name]
    return getattr(P, attr)(load_full_config(cfg_file, model))


import re as _re

_NUM = _re.compile(r"-?\$?\s?\d[\d,]*(?:\.\d+)?%?")


def _numbers(s: str) -> set[float]:
    """Extract numeric values from text (handles $, commas, %, decimals)."""
    out = set()
    for tok in _NUM.findall(s):
        t = tok.replace("$", "").replace(",", "").replace("%", "").strip()
        try:
            out.add(round(float(t), 4))
        except ValueError:
            continue
    return out


def numeric_match(answer: str, gold: str) -> float | None:
    """
    Numeric-aware correctness for financial QA: True if every number in the gold
    answer appears in the answer's numbers (0.1% relative tolerance). Returns None
    when the gold has no numbers, so the caller can fall back to string matching.
    """
    g = _numbers(gold)
    if not g:
        return None
    a = _numbers(answer)
    return float(all(any(abs(x - y) <= max(1e-9, abs(x) * 1e-3) for y in a) for x in g))


def score(answer: str, gold: str) -> dict:
    m = token_f1(answer, gold)
    nm = numeric_match(answer, gold)
    # Correct = numeric-aware match for numeric golds, else normalised substring match.
    substr = float(normalize_answer(gold) in normalize_answer(answer))
    return {"f1": m["f1"], "exact_match": exact_match(answer, gold),
            "answer_match": substr,
            "correct": substr if nm is None else nm,
            "is_numeric": float(nm is not None)}


def run_one(name: str, questions: list[dict], corpus_path: str, model: str) -> dict:
    t0 = time.perf_counter()
    pipe = build_pipeline(name, model)
    ingest_s = 0.0
    if hasattr(pipe, "ingest"):
        rep = pipe.ingest(corpus_path)
        ingest_s = getattr(rep, "ingestion_time_seconds", time.perf_counter() - t0)

    records = []
    for i, q in enumerate(questions, 1):
        try:
            r = pipe.query(q["question"])
            if isinstance(r, dict):
                answer, latency, tokens, cost = r["answer"], r["latency_ms"], r["tokens"], r["cost_usd"]
            else:
                answer, latency = r.answer, r.latency_ms
                tokens, cost = r.tokens_used.get("total", 0), r.cost_usd
            s = score(answer, q["ground_truth_answer"])
            records.append({"id": q["id"], "gold": q["ground_truth_answer"], "answer": answer,
                            "latency_ms": latency, "tokens": tokens, "cost_usd": cost, **s})
            print(f"  [{name}] {i}/{len(questions)} F1={s['f1']:.2f} "
                  f"match={s['answer_match']:.0f} {latency:.0f}ms", flush=True)
        except Exception as e:  # noqa: BLE001 — record failures, never fabricate a result
            print(f"  [{name}] {i}/{len(questions)} ERROR {type(e).__name__}: {e}", flush=True)
            records.append({"id": q["id"], "error": f"{type(e).__name__}: {e}"})

    ok = [r for r in records if "error" not in r]
    if not ok:
        return {"pipeline": name, "n": 0, "error": "no successful queries", "records": records}
    lat = [r["latency_ms"] for r in ok]
    return {
        "pipeline": name, "n": len(ok),
        "correct": round(statistics.mean(r["correct"] for r in ok), 4),
        "answer_match": round(statistics.mean(r["answer_match"] for r in ok), 4),
        "f1": round(statistics.mean(r["f1"] for r in ok), 4),
        "exact_match": round(statistics.mean(r["exact_match"] for r in ok), 4),
        "p50_latency_ms": round(float(np.percentile(lat, 50)), 1),
        "p95_latency_ms": round(float(np.percentile(lat, 95)), 1),
        "mean_tokens": round(statistics.mean(r["tokens"] for r in ok), 1),
        "cost_usd": round(sum(r["cost_usd"] for r in ok), 6),
        "ingest_s": round(ingest_s, 2), "records": records,
    }


def make_plots(rows: list[dict], out_dir: Path, meta: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("n")]
    if not rows:
        return
    names = [r["pipeline"] for r in rows]
    sub = f"{meta['domain']} · N={meta['n']} · {meta['model']} · seed {meta['seed']}"
    x = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7, 4)); w = 0.38
    ax.bar(x - w / 2, [r["correct"] for r in rows], w, label="Correct (numeric-aware)")
    ax.bar(x + w / 2, [r["f1"] for r in rows], w, label="Token F1")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15); ax.set_ylim(0, 1)
    ax.set_ylabel("score"); ax.set_title(f"Answer quality\n{sub}"); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "quality.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, [r["p50_latency_ms"] / 1000 for r in rows], color="#c1666b")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=15)
    ax.set_ylabel("p50 latency (s)"); ax.set_title(f"Median query latency\n{sub}")
    fig.tight_layout(); fig.savefig(out_dir / "latency.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Answer-quality benchmark for vectorless RAG")
    ap.add_argument("--pipelines", default="bm25,closed_book", help="comma list, or 'all'")
    ap.add_argument("--domain", default="finance", choices=["finance", "legal", "technical"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--output", default="results")
    args = ap.parse_args()

    selected = (["closed_book", *PIPELINES] if args.pipelines == "all"
                else [p.strip() for p in args.pipelines.split(",") if p.strip()])
    questions = load_questions(args.domain, args.n, args.seed, args.full)
    corpus = str(REPO_ROOT / f"data/processed/{args.domain}")
    out_dir = REPO_ROOT / args.output
    (out_dir / "per_question").mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    print(f"Quality benchmark | {args.domain} | N={len(questions)} | model={args.model} | {selected}")
    rows = []
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        agg = run_one(name, questions, corpus, args.model)
        (out_dir / "per_question" / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in agg.pop("records")) + "\n")
        rows.append(agg)

    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "domain": args.domain,
            "n": len(questions), "seed": args.seed, "model": args.model,
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__}
    (out_dir / "quality_summary.json").write_text(
        json.dumps({"meta": meta, "leaderboard": rows}, indent=2))

    cols = ["pipeline", "n", "correct", "answer_match", "f1", "exact_match",
            "p50_latency_ms", "p95_latency_ms", "mean_tokens", "cost_usd", "ingest_s"]
    with (out_dir / "quality_leaderboard.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    make_plots(rows, out_dir / "plots", meta)

    print("\n=== QUALITY LEADERBOARD ===")
    print(f"{'pipeline':16s} {'correct':>7s} {'F1':>6s} {'EM':>5s} {'p50ms':>7s} {'tok':>6s} {'$':>7s}")
    for r in rows:
        if r.get("n"):
            print(f"{r['pipeline']:16s} {r['correct']:7.2f} {r['f1']:6.2f} "
                  f"{r['exact_match']:5.2f} {r['p50_latency_ms']:7.0f} {r['mean_tokens']:6.0f} "
                  f"{r['cost_usd']:7.4f}")
        else:
            print(f"{r['pipeline']:16s}  (no successful runs — {r.get('error','')})")
    print(f"\nArtifacts -> {out_dir}/")


if __name__ == "__main__":
    main()
