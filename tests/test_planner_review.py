import json

from finsight.agents.team import keyword_plan
from finsight.config import Settings
from finsight.graph.workflow import run_analysis
from finsight.llm.mock import MockLLM

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)


def test_keyword_plan_picks_analysts_for_the_question():
    assert keyword_plan("Should a long-term investor consider this stock?") == [
        "researcher", "quant", "peers", "sentiment", "filings"]
    assert keyword_plan("How volatile has it been and what was the drawdown?") == ["quant"]
    assert keyword_plan("How does it compare with competitors?") == ["quant", "peers"]
    assert keyword_plan("What does the business do and any recent news?") == ["researcher", "quant", "sentiment"]


def test_planner_drives_which_agents_run():
    steps = []
    state = run_analysis("DEMO", "How volatile is it and how deep was the drawdown?", settings=OFFLINE,
                         engine_name="builtin", on_step=lambda n, s: steps.append(n))
    assert state.plan == ["quant"] and "researcher" not in steps and "peers" not in steps
    assert state.focus and state.approved


def test_planner_falls_back_to_keywords_on_garbage_and_always_keeps_quant():
    class BadPlanner(MockLLM):
        def complete(self, system, messages, *, role="", context=None, max_tokens=1500):
            if role == "planner":
                return "I think you should run everything!!"
            return super().complete(system, messages, role=role, context=context)

    class NoQuantPlanner(MockLLM):
        def complete(self, system, messages, *, role="", context=None, max_tokens=1500):
            if role == "planner":
                return json.dumps({"analysts": ["researcher", "astrologer"], "focus": "x"})
            return super().complete(system, messages, role=role, context=context)

    s1 = run_analysis("DEMO", "Compare with peers", llm=BadPlanner(), settings=OFFLINE, engine_name="builtin")
    plan_event = next(t for t in s1.trace if t["agent"] == "planner")
    assert plan_event["detail"]["source"] == "keywords" and s1.plan == ["quant", "peers"]
    s2 = run_analysis("DEMO", llm=NoQuantPlanner(), settings=OFFLINE, engine_name="builtin")
    assert s2.plan == ["researcher", "quant"]  # unknown name dropped, quant added for the risk rules


def test_human_can_request_changes_then_approve():
    calls = []

    def reviewer(state):
        calls.append(state.memo)
        if len(calls) == 1:
            return {"action": "revise", "feedback": "Mention the debt-to-equity figure in the summary."}
        return {"action": "approve"}

    steps = []
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin", reviewer=reviewer,
                         on_step=lambda n, s: steps.append(n))
    assert steps[-6:] == ["writer", "critic", "human_review", "writer", "critic", "human_review"]
    assert state.human_approved and state.human_rounds == 2 and state.revisions == 1
    reviews = [t["detail"] for t in state.trace if t["agent"] == "human" and t["event"] == "review"]
    assert reviews[0]["action"] == "revise" and "debt-to-equity" in reviews[0]["feedback"]


def test_human_edits_are_rechecked_by_the_critic():
    def reviewer(state):
        if state.human_rounds == 1:  # first review: slip an unsupported figure into the memo
            return {"action": "edit", "memo": state.memo + "\nAnalysts see 88.8% upside [S1]."}
        return {"action": "approve"}

    steps = []
    state = run_analysis("DEMO", settings=OFFLINE, engine_name="builtin", reviewer=reviewer,
                         on_step=lambda n, s: steps.append(n))
    # edit -> critic rejects the made-up figure -> writer revises -> critic -> human approves
    i = steps.index("human_review")
    assert steps[i + 1] == "critic"
    critic_reviews = [t["detail"] for t in state.trace if t["agent"] == "critic"]
    assert critic_reviews[1]["approved"] is False  # the critic rejected the human's unsupported figure
    assert "88.8" not in state.memo and state.human_approved


def test_no_reviewer_means_fully_automatic():
    steps = []
    run_analysis("DEMO", settings=OFFLINE, engine_name="builtin", on_step=lambda n, s: steps.append(n))
    assert "human_review" not in steps
