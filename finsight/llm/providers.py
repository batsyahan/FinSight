"""Concrete LLM clients over plain HTTP (no vendor SDKs needed)."""

from __future__ import annotations

import json
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

    # ---- native tool calling -------------------------------------------------
    def chat_tools(self, system, messages, tools, *, tool_choice="auto", role="", context=None,
                   max_tokens=1500) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "system": system,
            "messages": _to_anthropic(messages),
            "tools": [{"name": s["name"], "description": s["description"], "input_schema": s["parameters"]}
                      for s in tools],
            "tool_choice": {"required": {"type": "any"}, "none": {"type": "none"}}.get(tool_choice, {"type": "auto"}),
        }
        data = post_with_retry(self.URL, self._headers(), payload)
        blocks = data.get("content", [])
        return {
            "content": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
            "tool_calls": [{"id": b["id"], "name": b["name"], "args": b.get("input") or {}}
                           for b in blocks if b.get("type") == "tool_use"],
        }

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}



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
        msg, finish = self._post(self._payload(system, messages, max_tokens))
        content = msg.get("content") or ""
        if not content.strip() and finish == "length":
            raise LLMError("Model used its whole output budget before answering (finish_reason=length). "
                           "Raise FINSIGHT_MAX_TOKENS or set FINSIGHT_REASONING_EFFORT=low.")
        return content

    # ---- native tool calling -------------------------------------------------
    def chat_tools(self, system, messages, tools, *, tool_choice="auto", role="", context=None,
                   max_tokens=1500) -> dict[str, Any]:
        payload = self._payload(system, _to_openai(messages), max_tokens)
        payload["tools"] = [{"type": "function", "function": s} for s in tools]
        payload["tool_choice"] = tool_choice
        msg, finish = self._post(payload)
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": c.get("id"), "name": fn.get("name"), "args": args if isinstance(args, dict) else {}})
        content = msg.get("content") or ""
        if not content.strip() and not calls and finish == "length":
            raise LLMError("Model used its whole output budget before answering (finish_reason=length). "
                           "Raise FINSIGHT_MAX_TOKENS or set FINSIGHT_REASONING_EFFORT=low.")
        return {"content": content, "tool_calls": calls}

    def _payload(self, system, messages, max_tokens) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "max_tokens": max(max_tokens, self.min_tokens),
            "temperature": self.temperature,
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def _post(self, payload) -> tuple[dict[str, Any], str]:
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        data = post_with_retry(f"{self.base_url}/chat/completions", headers, payload)
        try:
            choice = data["choices"][0]
            return choice["message"], choice.get("finish_reason", "")
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response shape: {str(data)[:300]}") from e


# ---------------------------------------------------------------- message conversion
# The ReAct loop keeps a neutral history:
#   {"role": "user", "content": str}
#   {"role": "assistant", "content": str, "tool_calls": [{"id", "name", "args"}]}
#   {"role": "tool", "tool_call_id": str, "name": str, "content": str}


def _to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def _to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if out and out[-1]["role"] == role:  # Anthropic needs alternating roles: merge
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in messages:
        if m["role"] == "assistant":
            blocks = [{"type": "text", "text": m["content"]}] if (m.get("content") or "").strip() else []
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                       for c in m.get("tool_calls") or []]
            push("assistant", blocks or [{"type": "text", "text": "(no content)"}])
        elif m["role"] == "tool":
            push("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}])
        else:
            push("user", [{"type": "text", "text": m.get("content") or ""}])
    return out


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
