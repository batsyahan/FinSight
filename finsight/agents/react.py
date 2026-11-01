"""A small, provider-agnostic ReAct loop with two transports.

* Native: models that expose ``chat_tools`` (Claude, OpenAI-compatible APIs) use the API's
  own tool calling. The first turn sets ``tool_choice="required"``, so the model cannot
  answer before gathering evidence.
* JSON: other models (the offline mock, plain local models) reply with JSON text, either
      {"thought": "...", "action": "<tool>", "args": {...}}
  or
      {"thought": "...", "final": {"findings": [{"text": "...", "citations": ["S1"]}]}}

Either way the agent ends with findings that cite evidence ids, and uncited findings
are dropped.
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


NATIVE_PROTOCOL = """Use the provided tools to gather evidence. Every tool result is stored as evidence
with an id like S1 (shown in each tool result). You may call at most {max_steps} tools.

When you have enough information, stop calling tools and reply with ONE JSON object only:
{{"findings": [{{"text": "one factual finding", "citations": ["S1"]}}]}}

Rules:
- Only state figures that appear in tool results; never use outside knowledge or memory.
- Every finding must cite at least one evidence id. Write 3 to 6 concise findings."""


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
    if hasattr(llm, "chat_tools"):
        return _run_native(agent, role_prompt, task, llm, tools, state, max_steps)
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
                state: AnalysisState, max_steps: int) -> list[Finding]:
    system = role_prompt.strip() + "\n\n" + NATIVE_PROTOCOL.format(max_steps=max_steps)
    schemas = tools.schemas()
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    state.log(agent, "start", task)
    calls_made = tool_successes = 0
    json_retries = 0

    for _ in range(max_steps + 4):
        budget_left = calls_made < max_steps
        choice = "required" if tool_successes == 0 and budget_left else ("auto" if budget_left else "none")
        if not budget_left and messages[-1]["role"] == "tool":
            messages.append({"role": "user", "content": "Tool budget used up. Reply now with the findings JSON."})
        resp = llm.chat_tools(system, messages, schemas, tool_choice=choice, role=agent, context=state)
        calls = resp.get("tool_calls") or []
        content = resp.get("content") or ""

        if calls and budget_left:
            messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            if content.strip():
                state.log(agent, "thought", content[:300])
            for c in calls:
                calls_made += 1
                result = tools.call(c["name"], c.get("args") or {})
                state.log(agent, "tool", {"tool": c["name"], "args": c.get("args"),
                                          "result": result.get("evidence_id") or result.get("error")})
                if "error" in result:
                    obs = f"Tool {c['name']} failed: {result['error']}"
                else:
                    tool_successes += 1
                    obs = f"Evidence {result['evidence_id']} from {c['name']}: {to_json(result['data'])}"
                messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"], "content": obs})
            continue

        messages.append({"role": "assistant", "content": content})
        if tool_successes == 0 and budget_left:
            state.log(agent, "rejected_final", "no tool evidence yet")
            messages.append({"role": "user", "content": "Call a tool first; you have no evidence yet."})
            continue
        try:
            obj = extract_json(content)
        except ValueError:
            state.log(agent, "parse_error", content[:200])
            json_retries += 1
            if json_retries > 2:
                break
            messages.append({"role": "user", "content": 'Reply with the findings as ONE JSON object: {"findings": [...]}'})
            continue
        findings = _parse_findings(agent, obj.get("final", obj), state)
        state.log(agent, "final", [f.text for f in findings])
        return findings

    state.errors.append(f"{agent}: did not produce a final answer")
    return []
