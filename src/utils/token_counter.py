"""
Token counter utility — uses tiktoken for OpenAI models
and a character-based approximation for Anthropic models.
"""

import tiktoken
from loguru import logger


class TokenCounter:
    """
    Count tokens for text using the appropriate tokenizer.

    Usage:
        counter = TokenCounter(model="gpt-4o")
        n = counter.count("Hello, world!")
        print(f"{n} tokens")
    """

    # Mapping of model families to tiktoken encoding names
    _ENCODING_MAP = {
        "gpt-4o": "o200k_base",
        "gpt-4.1": "o200k_base",
        "gpt-4": "cl100k_base",
        "gpt-3.5": "cl100k_base",
    }

    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self._encoder = None

        # Try to load tiktoken encoder
        for prefix, encoding in self._ENCODING_MAP.items():
            if model.startswith(prefix):
                try:
                    self._encoder = tiktoken.get_encoding(encoding)
                    logger.debug(f"TokenCounter using {encoding} for {model}")
                except Exception:
                    logger.warning(f"Failed to load tiktoken encoding {encoding}")
                break

        if self._encoder is None:
            logger.info(
                f"No tiktoken encoder for {model}, using char-based approximation"
            )

    def count(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: Input text string.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        if self._encoder is not None:
            return len(self._encoder.encode(text))

        # Approximation: ~4 chars per token for English text
        return max(1, len(text) // 4)

    def count_messages(self, messages: list[dict]) -> int:
        """
        Count tokens in a list of chat messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.

        Returns:
            Total estimated token count including message overhead.
        """
        total = 0
        for msg in messages:
            total += 4  # Message overhead tokens
            total += self.count(msg.get("content", ""))
            total += self.count(msg.get("role", ""))
        total += 2  # Priming tokens
        return total

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within a token budget.

        Args:
            text: Input text.
            max_tokens: Maximum number of tokens allowed.

        Returns:
            Truncated text.
        """
        if self._encoder is not None:
            tokens = self._encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return self._encoder.decode(tokens[:max_tokens])

        # Approximation fallback
        max_chars = max_tokens * 4
        return text[:max_chars]
