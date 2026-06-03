"""
Unified LLM client — wraps Ollama (local/free), OpenAI, Anthropic,
and Google Gemini behind a single interface with automatic cost tracking.

Provider priority: Ollama (free) > Gemini (free tier) > OpenAI > Anthropic
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

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


# Pricing per 1M tokens (as of May 2026) — Ollama/Gemini Free = $0
MODEL_PRICING = {
    # --- Free (local) ---
    "llama3.1": {"input": 0.0, "output": 0.0},
    "llama3.2": {"input": 0.0, "output": 0.0},
    "mistral": {"input": 0.0, "output": 0.0},
    "qwen2.5": {"input": 0.0, "output": 0.0},
    "gemma2": {"input": 0.0, "output": 0.0},
    "phi3": {"input": 0.0, "output": 0.0},
    # --- Google Gemini (free tier) ---
    "gemini-2.5-flash": {"input": 0.0, "output": 0.0},
    "gemini-2.0-flash": {"input": 0.0, "output": 0.0},
    "gemini-1.5-flash": {"input": 0.0, "output": 0.0},
    # --- Paid APIs ---
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}

# Ollama model aliases (short name -> full pull name)
OLLAMA_MODELS = {
    "llama3.1", "llama3.2", "llama3.1:8b", "llama3.2:3b",
    "mistral", "mistral:7b", "qwen2.5", "qwen2.5:7b",
    "gemma2", "gemma2:9b", "phi3", "phi3:mini",
}


def detect_best_provider() -> tuple[str, str]:
    """
    Auto-detect the best available LLM provider.
    Returns (provider, model) tuple.

    Priority: Ollama (free) > Gemini (free) > OpenAI > Anthropic
    """
    # 1. Check Ollama
    if _ollama_available():
        models = _ollama_list_models()
        if models:
            model = models[0]
            logger.info(f"Auto-detected: Ollama with model '{model}' (FREE)")
            return "ollama", model
        else:
            logger.info("Ollama running but no models. Will pull llama3.2:3b")
            return "ollama", "llama3.2:3b"

    # 2. Check Google Gemini
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        logger.info("Auto-detected: Google Gemini (FREE tier)")
        return "gemini", "gemini-2.0-flash"

    # 3. Check OpenAI
    if os.getenv("OPENAI_API_KEY"):
        logger.info("Auto-detected: OpenAI")
        return "openai", "gpt-4o"

    # 4. Check Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Auto-detected: Anthropic")
        return "anthropic", "claude-3-5-sonnet-20241022"

    # Fallback
    logger.warning("No LLM provider found! Checking if Ollama can be started...")
    return "ollama", "llama3.2:3b"


def _ollama_available() -> bool:
    """Check if Ollama is running locally."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_list_models() -> list[str]:
    """List models available in Ollama."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return models
    except Exception:
        return []


class LLMClient:
    """
    Unified LLM client supporting Ollama (free/local), Google Gemini
    (free tier), OpenAI, and Anthropic.

    Usage:
        # Auto-detect best free provider
        client = LLMClient.auto()
        response = client.generate("What is vectorless RAG?")

        # Or specify explicitly
        client = LLMClient(model="llama3.2", provider="ollama")
    """

    def __init__(self, model: str = "llama3.2", temperature: float = 0.0,
                 max_tokens: int = 4096, timeout: int = 300,
                 provider: str = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client = None

        # Auto-detect provider if not specified
        if provider:
            self.provider = provider
        elif model.startswith("claude"):
            self.provider = "anthropic"
        elif model.startswith("gemini"):
            self.provider = "gemini"
        elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
            self.provider = "openai"
        elif model in OLLAMA_MODELS or _ollama_available():
            self.provider = "ollama"
        else:
            # Try auto-detection
            detected_provider, _ = detect_best_provider()
            self.provider = detected_provider

        # Initialize provider-specific client
        if self.provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "gemini":
            pass  # Uses REST API directly
        elif self.provider == "ollama":
            pass  # Uses REST API directly

        # Cost tracking
        self._pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
        self._total_cost = 0.0
        self._total_tokens = 0
        self._call_count = 0

    @classmethod
    def auto(cls, temperature: float = 0.0, max_tokens: int = 4096) -> "LLMClient":
        """Auto-detect the best available free provider and create a client."""
        provider, model = detect_best_provider()
        return cls(model=model, provider=provider, temperature=temperature,
                   max_tokens=max_tokens)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion from the LLM."""
        start = time.perf_counter()

        if self.provider == "ollama":
            response = self._call_ollama(prompt, system_prompt, json_mode)
        elif self.provider == "gemini":
            response = self._call_gemini(prompt, system_prompt, json_mode)
        elif self.provider == "openai":
            response = self._call_openai(prompt, system_prompt, json_mode)
        elif self.provider == "anthropic":
            response = self._call_anthropic(prompt, system_prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

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
            f"LLM call #{self._call_count} | {self.provider}/{self.model} | "
            f"{response.total_tokens} tokens | ${cost:.6f} | "
            f"{elapsed_ms:.0f}ms"
        )

        return response

    # ── Ollama (FREE, LOCAL) ────────────────────────────────────────

    def _call_ollama(self, prompt: str, system_prompt: str,
                     json_mode: bool) -> LLMResponse:
        """Call Ollama local API (http://localhost:11434)."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Ollama not reachable at localhost:11434. "
                f"Start it with: ollama serve\nError: {e}"
            )

        content = result.get("response", "")
        # Ollama provides eval_count and prompt_eval_count
        input_tokens = result.get("prompt_eval_count", len(prompt) // 4)
        output_tokens = result.get("eval_count", len(content) // 4)

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=0.0,  # FREE
            latency_ms=0.0,
            raw_response=result,
        )

    # ── Google Gemini (FREE TIER) ──────────────────────────────────

    def _call_gemini(self, prompt: str, system_prompt: str,
                     json_mode: bool) -> LLMResponse:
        """Call Google Gemini API via REST (free tier)."""
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY in .env for Gemini")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={api_key}"
        )

        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": f"[System Instructions]\n{system_prompt}\n\n[User Query]\n{prompt}"}],
            })
        else:
            contents.append({
                "role": "user",
                "parts": [{"text": prompt}],
            })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            },
        }

        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            result = json.loads(resp.read())

        # Parse Gemini response
        candidates = result.get("candidates", [{}])
        content = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            content = parts[0].get("text", "") if parts else ""

        usage = result.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", len(prompt) // 4)
        output_tokens = usage.get("candidatesTokenCount", len(content) // 4)

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=0.0,  # FREE tier
            latency_ms=0.0,
            raw_response=result,
        )

    # ── OpenAI ─────────────────────────────────────────────────────

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
            cost_usd=0.0,
            latency_ms=0.0,
            raw_response=response,
        )

    # ── Anthropic ──────────────────────────────────────────────────

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

    # ── Properties ─────────────────────────────────────────────────

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
            "provider": self.provider,
            "model": self.model,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
        }
