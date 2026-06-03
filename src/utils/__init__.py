# Utilities subpackage
#
# Submodules are exposed lazily (PEP 562) so importing a single lightweight
# utility (e.g. TokenCounter) does not eagerly pull in heavier siblings such as
# the database layer or the LLM client. Their imports run only on first access.
from importlib import import_module

# Public name -> (submodule, attribute)
_LAZY = {
    "LLMClient": ("llm_client", "LLMClient"),
    "TokenCounter": ("token_counter", "TokenCounter"),
    "setup_logger": ("logger_setup", "setup_logger"),
    "DatabaseManager": ("database", "DatabaseManager"),
    "db_manager": ("database", "db_manager"),
}

__all__ = list(_LAZY.keys())


def __getattr__(name: str):
    """Lazily import utilities on first access (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attr = target
    obj = getattr(import_module(f".{module}", __name__), attr)
    globals()[name] = obj  # cache for subsequent lookups
    return obj


def __dir__():
    return sorted(__all__)
