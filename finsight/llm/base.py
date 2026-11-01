"""Provider-agnostic LLM interface plus robust JSON extraction for agent replies."""

from __future__ import annotations

import json
import re
import time
from typing import Any, Protocol

import requests


class LLM(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        role: str = "",
        context: Any = None,
        max_tokens: int = 1500,
    ) -> str:
        """Return the assistant's text reply.

        ``role`` names the calling agent and ``context`` is the live AnalysisState. Real
        LLMs ignore both; the offline MockLLM uses them to produce deterministic output.
        """
        ...


class LLMError(RuntimeError):
    pass


def post_with_retry(url: str, headers: dict, payload: dict, timeout: int = 90, retries: int = 3) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
        else:
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code not in (408, 429, 500, 502, 503, 504, 529):
                break
        time.sleep(2**attempt)
    raise LLMError(f"LLM request failed: {last}")


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of an LLM reply (handles code fences and chatter)."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for c in candidates:
        start = c.find("{")
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(c)):
                ch = c[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                elif ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(c[start : i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except json.JSONDecodeError:
                            pass
                        break
            start = c.find("{", start + 1)
    raise ValueError("No JSON object found in model reply")
