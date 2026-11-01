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


NATIVE_PROTOCOL = """Use the provided tools to gather evidence. Every tool result is stored as evidence
with an id like S1 (shown in each tool result). You may call at most {max_steps} data tools.

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
    """Native tool calling. Every turn must be a tool call (tool_choice="required"); the answer
    itself is a call to submit_findings, which is refused until real evidence exists."""
    system = role_prompt.strip() + "\n\n" + NATIVE_PROTOCOL.format(max_steps=max_steps)
    all_schemas = tools.schemas() + [SUBMIT_TOOL]
    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    state.log(agent, "start", task)
    calls_made = tool_successes = 0
    provider_errors = 0

    for _ in range(max_steps + 5):
        budget_left = calls_made < max_steps
        schemas = all_schemas if budget_left else [SUBMIT_TOOL]
        try:
            resp = llm.chat_tools(system, messages, schemas, tool_choice="required", role=agent, context=state)
        except LLMError as e:
            # e.g. Groq "tool_use_failed": the model invented a tool name. Nudge and retry.
            if "tool_use_failed" not in str(e) or provider_errors >= 2:
                raise
            provider_errors += 1
            state.log(agent, "provider_rejected_call", str(e)[:200])
            messages.append({"role": "user", "content": (
                "Your last tool call was invalid. Only call the tools provided "
                f"({', '.join(s['name'] for s in schemas)}). Use submit_findings for your final answer.")})
            continue

        calls = resp.get("tool_calls") or []
        content = resp.get("content") or ""
        if not calls:  # some providers ignore "required"; accept a JSON answer as a fallback
            messages.append({"role": "assistant", "content": content})
            try:
                obj = extract_json(content)
            except ValueError:
                obj = None
            if obj is not None and tool_successes > 0:
                findings = _parse_findings(agent, obj.get("final", obj), state)
                state.log(agent, "final", [f.text for f in findings])
                return findings
            state.log(agent, "rejected_final" if obj is not None else "parse_error", content[:200])
            messages.append({"role": "user", "content": "Call one of the provided tools."})
            continue

        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        if content.strip():
            state.log(agent, "thought", content[:300])
        submitted = None
        for c in calls:
            if c["name"] == "submit_findings":
                if tool_successes == 0:
                    state.log(agent, "rejected_final", "no tool evidence yet")
                    obs = "Rejected: you have no evidence yet. Call a data tool first; do not answer from memory."
                else:
                    submitted = c.get("args") or {}
                    obs = "Findings received."
            elif not budget_left:
                obs = "Tool budget used up. Call submit_findings now."
            else:
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

        if submitted is not None:
            findings = _parse_findings(agent, submitted, state)
            state.log(agent, "final", [f.text for f in findings])
            return findings

    state.errors.append(f"{agent}: did not produce a final answer")
    return []
