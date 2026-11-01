"""A small, provider-agnostic ReAct loop with two transports.

* Native: models that expose ``chat_tools`` (Claude, OpenAI-compatible APIs) use the API's
  own tool calling. The orchestrator runs each agent's core tools first (``seed``), the model
  may call more, and it answers through a ``submit_findings`` tool.
* JSON: other models (the offline mock, plain local models) reply with JSON text, either
      {"thought": "...", "action": "<tool>", "args": {...}}
  or
      {"thought": "...", "final": {"findings": [{"text": "...", "citations": ["S1"]}]}}

Either way the agent ends with findings that cite evidence ids, and uncited findings
are dropped.
"""

from __future__ import annotations

import json
import re
from typing import Any

from finsight.llm.base import LLMError, extract_json
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


NATIVE_PROTOCOL = """Core evidence has already been gathered for you (see the tool results in the
conversation). Every tool result is stored as evidence with an id like S1.
You may call up to {max_steps} more tools if something important is missing.

When you have enough information, call the submit_findings tool. Rules:
- Only state figures that appear in tool results; never use outside knowledge or memory.
- Every finding must cite at least one evidence id. Write 3 to 6 concise findings."""

SUBMIT_TOOL = {
    "name": "submit_findings",
    "description": "Submit your final findings once you have gathered evidence. Call this exactly once, last.",
    "parameters": {
        "type": "object",
        "properties": {
            "findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "one factual finding"},
                        "citations": {"type": "array", "items": {"type": "string"},
                                      "description": "evidence ids such as S1"},
                    },
                    "required": ["text", "citations"],
                },
            }
        },
        "required": ["findings"],
    },
}


def run_react_agent(
    *,
    agent: str,
    role_prompt: str,
    task: str,
    llm: Any,
    tools: ToolRegistry,
    state: AnalysisState,
    max_steps: int = 6,
    seed: list[tuple[str, dict[str, Any]]] | None = None,
) -> list[Finding]:
    """``seed`` lists tool calls the orchestrator runs up front (native path), so the core
    evidence exists no matter how reliably a given model uses tools."""
    if hasattr(llm, "chat_tools"):
        return _run_native(agent, role_prompt, task, llm, tools, state, max_steps, seed or [])
    system = role_prompt.strip() + "\n\n" + PROTOCOL.format(tools=tools.describe(), max_steps=max_steps)
    messages: list[dict[str, str]] = [{"role": "user", "content": task}]
    state.log(agent, "start", task)
    tool_successes = 0

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
            if tool_successes == 0 and not force_final:
                # answering from memory is exactly what FinSight must not do
                state.log(agent, "rejected_final", "no tool evidence yet")
                messages.append({"role": "user", "content": (
                    "You have not called any tool yet, so you have no evidence to cite. "
                    "Do not answer from memory. Call a tool now.")})
                continue
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
            tool_successes += 1
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
        cites = [str(c) for c in it.get("citations", []) if state.evidence.get(str(c))]
        if text and cites:
            out.append(Finding(agent=agent, text=text, citations=cites))
        elif text:
            state.log(agent, "dropped_uncited_finding", text[:150])
    return out


def _run_native(agent: str, role_prompt: str, task: str, llm: Any, tools: ToolRegistry,
                state: AnalysisState, max_steps: int, seed: list[tuple[str, dict[str, Any]]]) -> list[Finding]:
    """Native tool calling, seeded.

    1. The orchestrator runs the seed tools and shows them to the model as its own earlier calls.
    2. The model may call more tools, then answers via submit_findings (or JSON / a bullet list).
    3. Findings without a valid evidence citation are dropped; the Critic checks the memo later.
    """
    system = role_prompt.strip() + "\n\n" + NATIVE_PROTOCOL.format(max_steps=max_steps)
    all_schemas = tools.schemas() + [SUBMIT_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    state.log(agent, "start", task)

    if seed:
        calls = [{"id": f"seed_{i}", "name": n, "args": a} for i, (n, a) in enumerate(seed)]
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        for c in calls:
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                             "content": _run_tool(agent, tools, state, c, seeded=True)})

    have_evidence = lambda: any(ev.source in tools.tools for ev in state.evidence.items.values())  # noqa: E731
    calls_made = 0
    nudges = 0
    for _ in range(max_steps + 4):
        budget_left = calls_made < max_steps
        schemas = all_schemas if budget_left else [SUBMIT_TOOL]
        try:
            resp = llm.chat_tools(system, messages, schemas, tool_choice="auto", role=agent, context=state)
        except LLMError as e:
            if "tool_use_failed" not in str(e):
                raise
            # Groq rejected the generation (invented tool, or text when a tool was expected).
            # The model's actual output is in failed_generation: try to use it as the answer.
            text = _failed_generation(e)
            state.log(agent, "provider_rejected_call", text[:200])
            findings = _findings_from_text(agent, text, state) if have_evidence() else []
            if findings:
                state.log(agent, "final", [f.text for f in findings])
                return findings
            nudges += 1
            if nudges > 2:
                break
            messages.append({"role": "user", "content": "Call submit_findings with your findings, citing evidence ids."})
            continue

        calls = resp.get("tool_calls") or []
        content = resp.get("content") or ""
        if not calls:
            messages.append({"role": "assistant", "content": content})
            findings = _findings_from_text(agent, content, state) if have_evidence() else []
            if findings:
                state.log(agent, "final", [f.text for f in findings])
                return findings
            state.log(agent, "parse_error", content[:200])
            nudges += 1
            if nudges > 2:
                break
            messages.append({"role": "user", "content": "Call submit_findings with your findings, citing evidence ids."})
            continue

        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        if content.strip():
            state.log(agent, "thought", content[:300])
        submitted = None
        for c in calls:
            if c["name"] == "submit_findings":
                if not have_evidence():
                    state.log(agent, "rejected_final", "no tool evidence yet")
                    obs = "Rejected: you have no evidence yet. Call a data tool first; do not answer from memory."
                else:
                    submitted = c.get("args") or {}
                    obs = "Findings received."
            elif not budget_left:
                obs = "Tool budget used up. Call submit_findings now."
            else:
                calls_made += 1
                obs = _run_tool(agent, tools, state, c)
            messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": obs})

        if submitted is not None:
            findings = _parse_findings(agent, submitted, state)
            state.log(agent, "final", [f.text for f in findings])
            return findings

    state.errors.append(f"{agent}: did not produce a final answer")
    return []


def _run_tool(agent: str, tools: ToolRegistry, state: AnalysisState, call: dict[str, Any], seeded: bool = False) -> str:
    result = tools.call(call["name"], call.get("args") or {})
    state.log(agent, "tool", {"tool": call["name"], "args": call.get("args"), "seeded": seeded,
                              "result": result.get("evidence_id") or result.get("error")})
    if "error" in result:
        return f"Tool {call['name']} failed: {result['error']}"
    return f"Evidence {result['evidence_id']} from {call['name']}: {to_json(result['data'])}"


def _failed_generation(e: LLMError) -> str:
    try:
        return json.loads(e.body).get("error", {}).get("failed_generation", "") or ""
    except (ValueError, AttributeError):
        return ""


_CITES = re.compile(r"[\[(]\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*[\])]")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")


def _findings_from_text(agent: str, text: str, state: AnalysisState) -> list[Finding]:
    """Accept findings as JSON ({"findings": [...]}, {"name": .., "arguments": {...}}) or as a
    markdown bullet list with [S1]-style citations."""
    try:
        obj = extract_json(text)
    except ValueError:
        obj = None
    if obj is not None:
        payload = obj.get("arguments", obj.get("final", obj))
        if isinstance(payload, dict) and "findings" in payload:
            return _parse_findings(agent, payload, state)
    items = []
    for line in text.splitlines():
        m = _BULLET.match(line)
        if not m:
            continue
        body = m.group(1)
        cites = [c.strip() for grp in _CITES.findall(body) for c in re.split(r"[,;]", grp)]
        clean = re.sub(r"\*\*|__", "", _CITES.sub("", body)).strip(" .:-")
        if clean:
            items.append({"text": clean + ".", "citations": cites})
    return _parse_findings(agent, {"findings": items}, state)
