"""Concrete LLM clients over plain HTTP (no vendor SDKs needed)."""

from __future__ import annotations

from typing import Any

from finsight.config import Settings
from finsight.llm.base import LLMError, post_with_retry


class AnthropicLLM:
    """Claude via the Anthropic Messages API."""

    name = "anthropic"
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, model: str = "claude-sonnet-5-5", temperature: float = 0.2) -> None:
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        self.api_key, self.model, self.temperature = api_key, model, temperature

    def complete(self, system, messages, *, role="", context=None, max_tokens=1500) -> str:
        data = post_with_retry(
            self.URL,
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": self.temperature,
                "system": system,
                "messages": messages,
            },
        )
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


class OpenAICompatLLM:
    """Any OpenAI-compatible chat completions endpoint: OpenAI, Groq, Ollama, OpenRouter, etc."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str, temperature: float = 0.2,
                 reasoning_effort: str = "", min_tokens: int = 0) -> None:
        if not model:
            raise LLMError("Set FINSIGHT_MODEL to a model your OpenAI-compatible provider serves")
        self.api_key, self.model, self.base_url, self.temperature = api_key, model, base_url, temperature
        reasoning = "gpt-oss" in model.lower()  # reasoning models spend output tokens on thinking first
        self.reasoning_effort = reasoning_effort or ("low" if reasoning else "")
        self.min_tokens = min_tokens or (3000 if reasoning else 0)

    def complete(self, system, messages, *, role="", context=None, max_tokens=1500) -> str:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "max_tokens": max(max_tokens, self.min_tokens),
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        data = post_with_retry(f"{self.base_url}/chat/completions", headers, payload)
        try:
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from e
        if not content.strip() and choice.get("finish_reason") == "length":
            raise LLMError("Model used its whole output budget before answering (finish_reason=length). "
                           "Raise FINSIGHT_MAX_TOKENS or set FINSIGHT_REASONING_EFFORT=low.")
        return content


def get_llm(settings: Settings | None = None) -> Any:
    s = settings or Settings.from_env()
    if s.llm == "anthropic":
        return AnthropicLLM(s.anthropic_api_key, s.model or "claude-sonnet-5-5")
    if s.llm == "openai":
        return OpenAICompatLLM(s.openai_api_key, s.model, s.openai_base_url,
                               reasoning_effort=s.reasoning_effort, min_tokens=s.max_tokens)
    if s.llm == "mock":
        from finsight.llm.mock import MockLLM

        return MockLLM()
    raise LLMError(f"Unknown FINSIGHT_LLM backend: {s.llm}")
