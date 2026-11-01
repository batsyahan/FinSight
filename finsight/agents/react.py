"""A small, provider-agnostic ReAct loop.

The model replies with JSON only, either
    {"thought": "...", "action": "<tool>", "args": {...}}
or
    {"thought": "...", "final": {"findings": [{"text": "...", "citations": ["S1"]}]}}

Using a JSON protocol instead of vendor tool-calling keeps the loop identical across
Claude, OpenAI-compatible APIs, local models and the offline mock.
"""

from __future__ import annotations

from typing import Any

from finsight.llm.base import extract_json
from finsight.state import AnalysisState, Finding
from finsight.tools.registry import ToolRegistry, to_json

PROTOCOL = """You work in a loop. On each turn reply with ONE JSON object and nothing else.

To use a tool:
{{"thought": "why you need it", "action": "<tool name>", "args": {{...}}}}

When you have enough information:
{{"thought": "...", "final": {{"findings": [{{"text": "one factual finding", "citations": ["S1"]}}]}}}}

Rules:
- Every tool result is stored as evidence with an id like S1. Cite those ids in findings.
- Only state figures that appear in tool results. Never invent numbers.
- Write 3 to 6 concise findings. Use at most {max_steps} tool calls.

Available tools:
{tools}"""


def run_react_agent(
    *,
    agent: str,
    role_prompt: str,
    task: str,
    llm: Any,
    tools: ToolRegistry,
    state: AnalysisState,
    max_steps: int = 6,
) -> list[Finding]:
    system = role_prompt.strip() + "\n\n" + PROTOCOL.format(tools=tools.describe(), max_steps=max_steps)
    messages: list[dict[str, str]] = [{"role": "user", "content": task}]
    state.log(agent, "start", task)

    for step in range(max_steps + 2):
        force_final = step >= max_steps
        if force_final:
            messages.append({"role": "user", "content": "Tool budget used up. Reply now with your final JSON."})
        reply = llm.complete(system, messages, role=agent, context=state)
        messages.append({"role": "assistant", "content": reply})
        try:
            obj = extract_json(reply)
        except ValueError:
            state.log(agent, "parse_error", reply[:200])
            messages.append({"role": "user", "content": "That was not valid JSON. Reply with one JSON object only."})
            continue

        if "final" in obj:
            findings = _parse_findings(agent, obj["final"], state)
            state.log(agent, "final", [f.text for f in findings])
            return findings

        action = obj.get("action")
        if not action or force_final:
            messages.append({"role": "user", "content": "Reply with either an action or a final answer."})
            continue
        state.log(agent, "thought", obj.get("thought", ""))
        result = tools.call(action, obj.get("args") or {})
        state.log(agent, "tool", {"tool": action, "args": obj.get("args"), "result": result.get("evidence_id") or result.get("error")})
        if "error" in result:
            obs = f"Tool {action} failed: {result['error']}"
        else:
            obs = f"Observation from {action} (evidence {result['evidence_id']}): {to_json(result['data'])}"
        messages.append({"role": "user", "content": obs})

    state.errors.append(f"{agent}: did not produce a final answer")
    return []


def _parse_findings(agent: str, final: Any, state: AnalysisState) -> list[Finding]:
    items = final.get("findings", []) if isinstance(final, dict) else []
    out = []
    for it in items:
        if isinstance(it, str):
            it = {"text": it, "citations": []}
        text = str(it.get("text", "")).strip()
        cites = [c for c in it.get("citations", []) if state.evidence.get(str(c))]
        if text:
            out.append(Finding(agent=agent, text=text, citations=cites))
    return out
