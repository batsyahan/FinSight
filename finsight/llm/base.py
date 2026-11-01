"""Provider-agnostic LLM interface plus robust JSON extraction for agent replies."""

from __future__ import annotations

import json
import re
import sys
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
    def __init__(self, message: str, body: str = "") -> None:
        super().__init__(message)
        self.body = body  # full provider response body, when there was one


_WAIT_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.IGNORECASE)


def retry_delay(response: "requests.Response | None", attempt: int) -> float:
    """Seconds to wait before retrying: the server's Retry-After / message hint, else backoff."""
    if response is not None:
        ra = response.headers.get("retry-after")
        try:
            if ra:
                return min(float(ra) + 1, 90)
        except ValueError:
            pass
        m = _WAIT_RE.search(response.text or "")
        if m:
            return min(int(m.group(1) or 0) * 60 + float(m.group(2)) + 1, 90)
    return float(2**attempt)


def post_with_retry(url: str, headers: dict, payload: dict, timeout: int = 120, retries: int = 6) -> dict:
    last, body = None, ""
    for attempt in range(retries):
        resp = None
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
        else:
            if resp.status_code == 200:
                return resp.json()
            last = f"HTTP {resp.status_code}: {resp.text[:300]}"
            body = resp.text
            if resp.status_code not in (408, 429, 500, 502, 503, 504, 529):
                break
        if attempt == retries - 1:
            break
        wait = retry_delay(resp, attempt)
        if resp is not None and resp.status_code == 429:
            print(f"  ⏳ rate limited by the LLM provider, waiting {wait:.0f}s...", file=sys.stderr, flush=True)
        time.sleep(wait)
    raise LLMError(f"LLM request failed: {last}", body=body)


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
