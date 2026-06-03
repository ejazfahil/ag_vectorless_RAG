"""
Basic tests for pipeline base classes and utilities.
"""

import pytest

from src.pipelines.base import RAGPipeline, RAGResponse
from src.utils.token_counter import TokenCounter


class TestRAGResponse:
    def test_default_creation(self):
        resp = RAGResponse(
            answer="Test answer",
            retrieved_contexts=["ctx1", "ctx2"],
            source_references=[{"doc": "test.pdf", "page": 1}],
            latency_ms=150.0,
        )
        assert resp.answer == "Test answer"
        assert len(resp.retrieved_contexts) == 2
        assert resp.cost_usd == 0.0
        assert resp.tokens_used["total"] == 0

    def test_with_tokens(self):
        resp = RAGResponse(
            answer="Answer",
            retrieved_contexts=[],
            source_references=[],
            latency_ms=100.0,
            tokens_used={"input": 500, "output": 100, "total": 600},
            cost_usd=0.003,
        )
        assert resp.tokens_used["total"] == 600
        assert resp.cost_usd == 0.003


class TestTokenCounter:
    def test_empty_string(self):
        counter = TokenCounter(model="gpt-4o")
        assert counter.count("") == 0

    def test_basic_counting(self):
        counter = TokenCounter(model="gpt-4o")
        count = counter.count("Hello, world!")
        assert count > 0

    def test_truncation(self):
        counter = TokenCounter(model="gpt-4o")
        long_text = "word " * 1000
        truncated = counter.truncate_to_tokens(long_text, 10)
        assert counter.count(truncated) <= 10

    def test_fallback_model(self):
        counter = TokenCounter(model="some-unknown-model")
        count = counter.count("Hello, world!")
        assert count > 0  # Should use char approximation


class TestAbstractPipeline:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            RAGPipeline(config={})
