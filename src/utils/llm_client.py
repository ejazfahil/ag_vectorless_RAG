"""
Unified LLM client — wraps OpenAI / Anthropic / LiteLLM behind a
single interface with automatic cost tracking and token counting.
"""

import os
import json
import time
from typing import Any
from dataclasses import dataclass

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@dataclass
class LLMResponse:
    """Standardized LLM response with metadata."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    raw_response: Any = None


# Pricing per 1M tokens (as of May 2026)
MODEL_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}


class LLMClient:
    """
    Unified LLM client supporting OpenAI and Anthropic models.

    Usage:
        client = LLMClient(model="gpt-4o")
        response = client.generate("What is vectorless RAG?")
        print(response.content, response.cost_usd)
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0,
                 max_tokens: int = 4096, timeout: int = 120):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Determine provider
        if model.startswith("claude"):
            self.provider = "anthropic"
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        else:
            self.provider = "openai"
            import openai
            self._client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        # Cost tracking
        self._pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        self._total_cost = 0.0
        self._total_tokens = 0
        self._call_count = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            prompt: User message / query.
            system_prompt: System-level instructions.
            json_mode: If True, request JSON output format.

        Returns:
            LLMResponse with content, token counts, and cost.
        """
        start = time.perf_counter()

        if self.provider == "openai":
            response = self._call_openai(prompt, system_prompt, json_mode)
        else:
            response = self._call_anthropic(prompt, system_prompt)

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Compute cost
        cost = (
            (response.input_tokens * self._pricing["input"] / 1_000_000)
            + (response.output_tokens * self._pricing["output"] / 1_000_000)
        )

        response.cost_usd = cost
        response.latency_ms = elapsed_ms

        # Update cumulative tracking
        self._total_cost += cost
        self._total_tokens += response.total_tokens
        self._call_count += 1

        logger.debug(
            f"LLM call #{self._call_count} | {self.model} | "
            f"{response.total_tokens} tokens | ${cost:.6f} | "
            f"{elapsed_ms:.0f}ms"
        )

        return response

    def _call_openai(self, prompt: str, system_prompt: str,
                     json_mode: bool) -> LLMResponse:
        """Make an OpenAI API call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)

        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cost_usd=0.0,  # Filled by caller
            latency_ms=0.0,
            raw_response=response,
        )

    def _call_anthropic(self, prompt: str, system_prompt: str) -> LLMResponse:
        """Make an Anthropic API call."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        content = response.content[0].text if response.content else ""

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=0.0,
            latency_ms=0.0,
            raw_response=response,
        )

    @property
    def cumulative_cost(self) -> float:
        """Total USD spent across all calls."""
        return self._total_cost

    @property
    def cumulative_tokens(self) -> int:
        """Total tokens consumed across all calls."""
        return self._total_tokens

    def get_usage_summary(self) -> dict:
        """Return a summary of all LLM usage for this client instance."""
        return {
            "model": self.model,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
        }
