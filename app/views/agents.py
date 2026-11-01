"""Multi-Agent Research Workspace: start a run, watch real agent execution, inspect every output."""

import queue
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from finsight.agents.team import ANALYST_INFO
from finsight.graph.workflow import run_analysis
from finsight.guardrails import validate_ticker
from finsight.llm.base import LLMError
from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()
pages = st.session_state["_pages"]

st.html(C.page_header("Agents", "Run the multi-agent research workflow and inspect what each agent actually did"))

# ---------------------------------------------------------------- run form
left, right = st.columns([1, 1.6], gap="medium")
with left:
    with st.container(key="glass_run_form"):
        st.html(C.panel_head("New research run", "play"))
        default_t = st.session_state.get("agents_ticker") or st.session_state.get("ticker") or "INFY.NS"
        options = list(dict.fromkeys([default_t, *store.watchlist(), *D.COMPANIES]))
        ticker = st.selectbox("Ticker", options, index=0, key="ag_ticker", accept_new_options=True)
        question = st.text_input("Investor question", "Should a long-term investor consider this stock?",
                                 key="ag_question", max_chars=500)
        analysts = st.multiselect("Analysts (empty = the Planner decides)", list(ANALYST_INFO), key="ag_analysts")
        peers = st.text_input("Peers (optional, comma-separated)", "", key="ag_peers", placeholder="TCS.NS, WIPRO.NS")
        uploads = st.file_uploader("Filings for RAG (optional)", type=["pdf", "txt", "md", "html"],
                                   accept_multiple_files=True, key="ag_files")
        max_rev = st.slider("Max Writer to Critic revisions", 0, 4, s.max_revisions, key="ag_rev")
        if s.llm == "mock":
            st.html(C.note("Mock LLM mode: agents run real tools on the configured data, but language steps use a "
                           "deterministic offline model, not a real LLM. Configure a provider in Settings."))
        start = st.button("Start research run", type="primary", icon=":material/play_arrow:", key="ag_start",
                          use_container_width=True)
        start = start or st.session_state.pop("agents_autostart", False)

with right:
    live = st.container(key="glass_live")

# ---------------------------------------------------------------- execution
if start:
    try:
        ticker = validate_ticker(str(ticker).split(" · ")[0])
        peer_list = [validate_ticker(p) for p in peers.split(",") if p.strip()]
    except ValueError as e:
        st.error(str(e))
        st.stop()
    filings_dir = None
    if uploads:
        filings_dir = tempfile.mkdtemp(prefix="finsight_filings_")
        for f in uploads:
            (Path(filings_dir) / Path(f.name).name).write_bytes(f.getbuffer())
    from dataclasses import replace

    settings = replace(s, max_revisions=max_rev)
    events: queue.Queue = queue.Queue()
    box: dict = {}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def worker() -> None:
        try:
            box["state"] = run_analysis(ticker, question, settings=settings, analysts=analysts or None,
                                        peers=peer_list or None, filings=filings_dir,
                                        on_step=lambda n, st_: events.put((n, time.time(), list(st_.plan),
                                                                           bool(st_.approved))))
        except Exception as e:  # shown to the user and recorded as a failed run
            box["error"] = e

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    status: dict[str, str] = {"planner": "running"}
    done_at: dict[str, float] = {}
    plan: list[str] = []
    t0 = time.time()
    with live:
        st.html(C.panel_head(f"Running: {ticker}", "bot", right_html=C.pill("Live", "info")))
        ph = st.empty()
        while th.is_alive() or not events.empty():
            while not events.empty():
                node, ts, plan, approved = events.get()
                done_at[node] = ts - t0
                status[node] = "completed"
                if node == "planner":
                    for a in plan:
                        status[a] = "running"
                elif node in ANALYST_INFO and all(status.get(a) == "completed" for a in plan):
                    status["risk"] = "running"
                elif node == "risk":
                    status["writer"] = "running"
                elif node == "writer":
                    status["critic"] = "running"
                elif node == "critic":
                    status["critic"] = "completed" if approved else "revised"
                    if not approved:
                        status["writer"] = "running"  # revision (or budget exhausted; resolved when the run ends)
            order = ["planner", *(plan or [a for a in ANALYST_INFO if a != "filings"]), "risk", "writer", "critic"]
            items = []
            for n in order:
                stt = status.get(n, "queued")
                name, desc = D.AGENT_INFO[n]
                if n in done_at:
                    desc += f" · done at {done_at[n]:.1f}s"
                items.append({"name": name, "desc": desc, "status": stt,
                              "pill": {"completed": "Completed", "running": "Running", "queued": "Queued",
                                       "revised": "Sent back"}.get(stt, stt)})
            ph.html(C.timeline(items) + C.note(f"Elapsed {time.time() - t0:.0f}s"))
            time.sleep(0.35)
    if "error" in box:
        err = box["error"]
        store.save_failed_run(ticker, question, f"{type(err).__name__}: {err}", started)
        hint = ""
        if isinstance(err, LLMError) and "429" in str(err):
            hint = " The LLM provider's rate limit was hit; wait a minute or use fewer analysts."
        elif isinstance(err, LLMError) and ("401" in str(err) or "403" in str(err)):
            hint = " The API key was rejected; check Settings / your .env."
        st.error(f"Run failed: {type(err).__name__}: {str(err)[:300]}.{hint}")
    else:
        rid = store.save_run(box["state"], llm=s.llm, data=s.data, started_at=started)
        st.session_state["agents_run_id"] = rid
        st.session_state["ticker"] = ticker
        st.rerun()

# ---------------------------------------------------------------- inspect a run
runs = store.runs(limit=30)
if not runs:
    with live:
        st.html(C.panel_head("Agent activity", "bot"))
        st.html(C.empty("No runs yet. Start one on the left; each agent's real execution appears here."))
    st.stop()

ids = [r["id"] for r in runs]
sel = st.session_state.get("agents_run_id") if st.session_state.get("agents_run_id") in ids else ids[0]
with live:
    label = {r["id"]: f"{r['ticker']} · {r['created_at'][:16].replace('T', ' ')} UTC · {r['status']}" for r in runs}
    sel = st.selectbox("Run", ids, index=ids.index(sel), format_func=label.get, key="ag_run_pick")
    run = store.run(sel)
    if run["status"] == "failed":
        st.html(C.panel_head(f"{run['ticker']}: failed run", "alert", "coral"))
        st.error("; ".join(run.get("error") or ["unknown error"]))
        st.stop()
    u = (run.get("usage") or {}).get("totals", {})
    st.html(C.panel_head(f"{run['ticker']}: agent activity", "bot",
                         right_html=C.pill("Verified" if run["approved"] else "Unverified",
                                           "ok" if run["approved"] else "warn")))
    st.html(C.timeline(D.pipeline_from_run(run)) +
            C.note(f"LLM: {run.get('llm')} · data: {run.get('data')} · {u.get('llm_calls', 0)} model calls · "
                   f"{'~' if u.get('tokens_estimated') else ''}{u.get('input_tokens', 0) + u.get('output_tokens', 0):,} tokens"))
    c1, c2 = st.columns(2)
    memo = next((m for m in store.memos(run["ticker"], 50) if m["run_id"] == run["id"]), None)
    if memo and c1.button("Open memo", type="primary", icon=":material/description:", key="ag_open_memo",
                          use_container_width=True):
        st.session_state["memo_id"] = memo["id"]
        st.switch_page(pages["Memo Library"])
    if c2.button("Open company research", icon=":material/query_stats:", key="ag_open_research", use_container_width=True):
        shell.open_research(run["ticker"])

t_out, t_critic, t_risk, t_sec = st.tabs(["Agent outputs", "Critic rounds", "Risk flags", "Guardrails"])
with t_out:
    findings = run.get("findings") or []
    trace = run.get("trace") or []
    for node in ["planner", *[a for a in ANALYST_INFO if a in (run.get("plan") or [])], "risk"]:
        name, desc = D.AGENT_INFO[node]
        with st.expander(f"{name}: {desc}", expanded=node == "planner"):
            if node == "planner":
                p = next((t["detail"] for t in trace if t.get("agent") == "planner" and t.get("event") == "plan"), {})
                st.html(C.kv_rows([("Analysts", ", ".join(p.get("analysts", []))), ("Source", p.get("source", "")),
                                   ("Focus", p.get("focus", "") or "n/a"), ("Reason", p.get("reason", "") or "n/a")]))
                continue
            tools = [t["detail"] for t in trace if t.get("agent") == node and t.get("event") == "tool"]
            if tools:
                st.html(C.table([("tool", "Tool call"), ("args", "Arguments"), ("res", "Result")],
                                [{"tool": d.get("tool"), "args": str(d.get("args") or {}),
                                  "res": f"evidence {d.get('result')}" if str(d.get("result", "")).startswith("S")
                                  else f"failed: {d.get('result')}"} for d in tools]))
            mine = [f for f in findings if f["agent"] == node]
            if mine:
                st.html("".join(f'<div class="fs-bullet"><span>{C.esc(f["text"])} <span class="fs-xs fs-muted">'
                                f'[{C.esc(", ".join(f["citations"]))}]</span></span></div>' for f in mine))
            dropped = [t["detail"] for t in trace if t.get("agent") == node and t.get("event") == "dropped_uncited_finding"]
            if dropped:
                st.html(C.note(f"{len(dropped)} uncited finding(s) dropped: " + " | ".join(map(str, dropped))[:500]))
            if not tools and not mine:
                st.html(C.empty("No output recorded."))
with t_critic:
    rounds = D.critic_rounds(run)
    if not rounds:
        st.html(C.empty("No Critic review recorded."))
    for i, r in enumerate(rounds, 1):
        ok = r.get("approved")
        with st.container(key=f"glass_round_{i}"):
            st.html(C.panel_head(f"Round {i}: {'approved' if ok else 'sent back to the Writer'}",
                                 "check" if ok else "alert", "" if ok else "amber",
                                 right_html=f'<span class="fs-small fs-muted">{r.get("claims_verified", "?")} of '
                                            f'{r.get("claims", "?")} claims verified</span>'))
            issues = r.get("issue_list") or []
            st.html("".join(C.check_line(x, False) for x in issues) or C.check_line("No issues", True))
    if not run["approved"] and rounds:
        st.warning("The revision budget ran out before every claim was supported. The memo is shown as UNVERIFIED; "
                   "inspect the remaining issues above.")
with t_risk:
    flags = [f for f in store.flags(ticker=run["ticker"], latest_only=False) if f["run_id"] == run["id"]]
    st.html(C.table([("sev", "Severity"), ("cat", "Category"), ("rule", "Rule"), ("val", "#Value"), ("ev", "Evidence")],
                    [{"sev": C.pill(f["severity"], {"high": "bad", "medium": "warn"}.get(f["severity"], "info")),
                      "cat": f["category"], "rule": f"{f['metric']} {f['threshold']}", "val": C.fmt_num(f["value"]),
                      "ev": f["evidence_id"] or ""} for f in flags], html_cols={"sev"})
            if flags else C.empty("No rule-based flags raised in this run."))
with t_sec:
    sec_events = run.get("security") or []
    st.html("".join(C.check_line(f"Removed possible prompt injection from {e['tool']} output: "
                                 f"{', '.join(e['matches'])[:160]}", False) for e in sec_events)
            or C.check_line("No prompt-injection attempts detected in tool data", True))
