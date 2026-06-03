# Pipelines subpackage
#
# The lightweight base classes are imported eagerly so that
# `from src.pipelines import RAGPipeline, RAGResponse` (and unit tests that only
# touch the base contracts) work without pulling in heavy optional dependencies
# such as langchain, loguru, ragas, or elasticsearch.
#
# The concrete pipelines are loaded lazily via PEP 562 module __getattr__, so
# their (potentially heavy) imports only run when a pipeline is actually used.
from importlib import import_module

from .base import RAGPipeline, RAGResponse

# Public name -> (submodule, attribute)
_LAZY = {
    "AgenticRAG": ("agentic_rag", "AgenticRAG"),
    "BM25RAG": ("bm25_rag", "BM25RAG"),
    "EmbeddingFreeRAG": ("embedding_free_rag", "EmbeddingFreeRAG"),
    "HybridSoTARAG": ("hybrid_sota", "HybridSoTARAG"),
    "PageIndexRAG": ("pageindex_rag", "PageIndexRAG"),
    "RoamingRAG": ("roaming_rag", "RoamingRAG"),
    "ThreeStageHybridRAG": ("three_stage_hybrid", "ThreeStageHybridRAG"),
}

__all__ = ["RAGPipeline", "RAGResponse", *_LAZY.keys()]


def __getattr__(name: str):
    """Lazily import concrete pipelines on first access (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    obj = getattr(import_module(f".{module}", __name__), attr)
    globals()[name] = obj  # cache for subsequent lookups
    return obj


def __dir__():
    return sorted(__all__)
