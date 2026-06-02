"""Unit tests for BM25 backend."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.backends.bm25_backend import BM25Backend


def test_bm25_fit_and_retrieve_basic():
    docs = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "cats and dogs are great pets",
    ]
    bm25 = BM25Backend()
    bm25.fit(docs)
    results = bm25.retrieve("cat mat", top_k=1)
    assert results[0][0] == 0, "Expected first doc to rank highest for 'cat mat'"


def test_bm25_scores_are_non_negative():
    docs = ["machine learning is great", "deep learning neural networks"]
    bm25 = BM25Backend()
    bm25.fit(docs)
    results = bm25.retrieve("learning", top_k=2)
    for _, score in results:
        assert score >= 0


def test_bm25_top_k_limit():
    docs = [f"document number {i}" for i in range(10)]
    bm25 = BM25Backend()
    bm25.fit(docs)
    results = bm25.retrieve("document", top_k=3)
    assert len(results) == 3


def test_bm25_empty_query_returns_zeros():
    docs = ["hello world", "foo bar"]
    bm25 = BM25Backend()
    bm25.fit(docs)
    results = bm25.retrieve("", top_k=2)
    for _, score in results:
        assert score == 0.0
