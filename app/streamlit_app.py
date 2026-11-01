"""FinSight web UI.   streamlit run app/streamlit_app.py

Tabs: Analyze (live agent graph, memo with hover citations, evidence, exports), Portfolio,
Evaluation, About. The analysis runs in a background thread and the page polls its progress,
so parallel agents never touch Streamlit from worker threads.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

from finsight.agents.team import ANALYST_INFO  # noqa: E402
from finsight.config import Settings  # noqa: E402
from finsight.data.providers import get_provider  # noqa: E402
from finsight.export import export, memo_to_html_fragment  # noqa: E402
from finsight.graph.workflow import run_analysis  # noqa: E402
from finsight.obs import usage_summary  # noqa: E402

st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide")

LABELS = {"planner": "🗺️ Planner", "researcher": "🔎 Researcher", "quant": "📊 Quant", "peers": "👥 Peers",
          "sentiment": "📰 Sentiment", "filings": "📄 Filings", "risk": "⚠️ Risk", "writer": "✍️ Writer",
          "critic": "🧐 Critic"}


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    base = Settings.from_env()
    backends = ["mock", "openai", "anthropic"]
    llm = st.selectbox("LLM backend", backends, index=backends.index(base.llm) if base.llm in backends else 0,
                       help="mock runs offline with no API key")
    model = st.text_input("Model", base.model, help="e.g. openai/gpt-oss-120b on Groq; blank = default")
    if llm == "openai":
        os.environ["OPENAI_BASE_URL"] = st.text_input("Base URL", base.openai_base_url or "https://api.groq.com/openai/v1")
        if key := st.text_input("API key", type="password", help="Leave blank to use .env"):
            os.environ["OPENAI_API_KEY"] = key
    elif llm == "anthropic" and (key := st.text_input("Anthropic API key", type="password")):
        os.environ["ANTHROPIC_API_KEY"] = key
    data = st.selectbox("Market data", ["sample", "yfinance"], index=1 if base.data == "yfinance" else 0,
                        help="sample = synthetic offline data; yfinance = real Yahoo Finance data")
    max_rev = st.slider("Max Writer ↔ Critic revisions", 0, 4, base.max_revisions)
    st.divider()
    st.caption("Analysts (empty = let the Planner decide)")
    chosen = st.multiselect("Analysts", list(ANALYST_INFO), default=[], label_visibility="collapsed")
    peers_txt = st.text_input("Peers (optional)", "", placeholder="TCS.NS, WIPRO.NS")
    uploads = st.file_uploader("Filings for RAG (optional)", type=["pdf", "txt", "md", "html"],
                               accept_multiple_files=True)

settings = replace(Settings.from_env(), llm=llm, model=model, data=data, max_revisions=max_rev)

st.title("📈 FinSight")
st.caption("A team of AI agents researches a stock, runs quant analysis, compares peers, reads filings, flags "
           "risks and writes a memo where every figure is checked against its cited evidence. "
           "Educational research, not financial advice.")

tab_an, tab_pf, tab_ev, tab_about = st.tabs(["Analyze", "Portfolio", "Evaluation", "About"])


def agent_graph(plan: list[str], status: dict[str, str]) -> str:
    colors = {"pending": "#c9ced6", "done": "#3fb950", "rejected": "#f0883e", "approved": "#3fb950"}
    analysts = plan or list(ANALYST_INFO)
    lines = ['digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" '
             'fontname="Helvetica" fontsize=11 color="#8b949e"]; edge [color="#8b949e"];']
    for n in ["planner", *analysts, "risk", "writer", "critic"]:
        fill = colors.get(status.get(n, "pending"), "#c9ced6")
        lines.append(f'"{n}" [label="{LABELS.get(n, n)}" fillcolor="{fill}"];')
    lines += [f'"planner" -> "{a}"; "{a}" -> "risk";' for a in analysts]
    lines += ['"risk" -> "writer"; "writer" -> "critic"; "critic" -> "writer" [style=dashed label="revise"];', "}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- analyze
with tab_an:
    c1, c2 = st.columns([1, 3])
    ticker = c1.text_input("Ticker", "INFY.NS" if data == "yfinance" else "DEMO", help="NSE: .NS, BSE: .BO")
    question = c2.text_input("Investor question", "Should a long-term investor consider this stock?")
    if st.button("Run analysis", type="primary"):
        filings_dir = None
        if uploads:
            filings_dir = tempfile.mkdtemp(prefix="finsight_filings_")
            for f in uploads:
                (Path(filings_dir) / Path(f.name).name).write_bytes(f.getbuffer())
        events: queue.Queue = queue.Queue()
        box: dict = {}

        def worker() -> None:
            try:
                box["state"] = run_analysis(
                    ticker, question, settings=settings, analysts=chosen or None,
                    peers=[p.strip() for p in peers_txt.split(",") if p.strip()] or None, filings=filings_dir,
                    on_step=lambda n, s: events.put((n, s.approved, list(s.plan), len(s.evidence.items))))
            except Exception as e:  # shown in the UI
                box["error"] = f"{type(e).__name__}: {e}"

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        graph_ph, log_ph = st.empty(), st.empty()
        status, plan, log = {}, [], []
        t0 = time.time()
        while th.is_alive() or not events.empty():
            while not events.empty():
                node, approved, plan, n_ev = events.get()
                status[node] = ("approved" if approved else "rejected") if node == "critic" else "done"
                log.append(f"`{time.time() - t0:5.1f}s` {LABELS.get(node, node)} · {n_ev} evidence items"
                           + (" · ✅ approved" if node == "critic" and approved else
                              " · sent back" if node == "critic" else ""))
            graph_ph.graphviz_chart(agent_graph(plan, status))
            log_ph.markdown("\n\n".join(log[-12:]) or "Starting…")
            time.sleep(0.4)
        if "error" in box:
            st.error(box["error"])
        else:
            st.session_state["result"] = box["state"]

    state = st.session_state.get("result")
    if state is not None:
        u = usage_summary(state)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Critic", "Approved ✅" if state.approved else "Not approved")
        m2.metric("Revisions", state.revisions)
        m3.metric("Evidence items", len(state.evidence.items))
        m4.metric("LLM tokens", f"{u['totals']['input_tokens'] + u['totals']['output_tokens']:,}")
        if state.security_events:
            st.warning(f"🛡️ Guardrails removed {len(state.security_events)} prompt-injection attempt(s) from tool data.")
        if not state.approved and state.critique.get("issues"):
            st.warning("Outstanding Critic issues:\n\n- " + "\n- ".join(state.critique["issues"][:8]))

        t_memo, t_evid, t_peer, t_lab, t_usage, t_trace = st.tabs(
            ["Memo", "Evidence", "Peers", "Strategy lab", "Usage", "Trace"])
        with t_memo:
            st.caption("Hover a citation to see the evidence behind it.")
            st.html("<style>abbr.cite{color:#2f81f7;text-decoration:none;border-bottom:1px dotted;cursor:help}</style>"
                    + memo_to_html_fragment(state.memo, state))
            d1, d2, d3 = st.columns(3)
            stem = state.ticker.replace(".", "_")
            d1.download_button("Markdown", export(state, "md")[0], f"{stem}_memo.md")
            d2.download_button("HTML", export(state, "html")[0], f"{stem}_memo.html")
            try:
                d3.download_button("PDF", export(state, "pdf")[0], f"{stem}_memo.pdf")
            except RuntimeError:
                d3.caption("PDF: `pip install fpdf2`")
            except Exception as e:  # a PDF problem must never break the page
                d3.caption(f"PDF unavailable ({type(e).__name__})")
        with t_evid:
            for ev in state.evidence.items.values():
                with st.expander(f"[{ev.id}] {ev.title}"):
                    st.json(ev.data, expanded=False)
        with t_peer:
            ev = next((e for e in state.evidence.items.values() if e.source == "compare_peers"), None)
            if ev:
                st.dataframe(pd.DataFrame(ev.data["table"]).T, use_container_width=True)
                st.caption("Subject vs peer median (rank 1 = best)")
                st.dataframe(pd.DataFrame(ev.data["vs_peers"]).T, use_container_width=True)
            else:
                st.info("The Peers analyst did not run.")
        with t_lab:
            ev = next((e for e in state.evidence.items.values() if e.source == "run_strategy_lab"), None)
            if ev:
                st.dataframe(pd.DataFrame(ev.data["strategies"]).T.drop(columns=["params"], errors="ignore"),
                             use_container_width=True)
                st.caption("Walk-forward (parameters chosen in-sample, tested out-of-sample)")
                wf = pd.DataFrame(ev.data["walk_forward"]).T.drop(columns=["params_chosen"], errors="ignore")
                st.dataframe(wf, use_container_width=True)
                try:
                    px = get_provider(data).prices(state.ticker, "2y").to_frame()
                    px["SMA 50"], px["SMA 200"] = px["close"].rolling(50).mean(), px["close"].rolling(200).mean()
                    st.line_chart(px)
                except Exception as e:
                    st.info(f"Chart unavailable: {e}")
            else:
                st.info("The Quant analyst did not run.")
        with t_usage:
            per = pd.DataFrame(u["per_agent"]).T
            if not per.empty:
                st.bar_chart(per[["input_tokens", "output_tokens"]])
                st.dataframe(per, use_container_width=True)
            st.caption("Wall time per node (seconds)")
            st.bar_chart(pd.Series(u["node_seconds"], name="seconds"))
        with t_trace:
            st.code(json.dumps(state.trace, indent=2, default=str), language="json")

# --------------------------------------------------------------------------- portfolio
with tab_pf:
    from finsight.portfolio import analyze_portfolio, parse_holdings, portfolio_summary

    spec = st.text_input("Holdings (ticker:weight)",
                         "INFY.NS:40,HDFCBANK.NS:30,ITC.NS:30" if data == "yfinance" else "DEMO:40,ACME:30,GLOBEX:30")
    pc1, pc2 = st.columns(2)
    value = pc1.number_input("Portfolio value", 10_000.0, 1e10, 1_000_000.0, step=100_000.0)
    want_summary = pc2.checkbox("LLM summary (grounding-checked)", value=True)
    if st.button("Analyze portfolio"):
        try:
            with st.spinner("Crunching risk numbers…"):
                a = analyze_portfolio(parse_holdings(spec), get_provider(data), value=value)
                s = portfolio_summary(a, __import__("finsight.llm.providers", fromlist=["get_llm"]).get_llm(settings)) \
                    if want_summary else None
            st.session_state["portfolio"] = (a, s)
        except Exception as e:
            st.error(f"{type(e).__name__}: {e}")
    if "portfolio" in st.session_state:
        a, s = st.session_state["portfolio"]
        m, v = a["metrics"], a["var_1d_95"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Return", f"{m['total_return_pct']}%")
        k2.metric("Volatility", f"{m['ann_volatility_pct']}%")
        k3.metric("1-day VaR 95%", f"{v['historical_var_pct']}%", help=f"≈ {v['historical_var_amount']:,.0f}")
        k4.metric("Avg correlation", m["avg_pairwise_correlation"])
        for f in a["flags"]:
            st.warning(f["message"])
        if s:
            st.markdown(s["summary"])
            st.caption("Grounding: " + ("✅ every figure verified" if s["grounding"]["passed"] else
                                        "❌ " + "; ".join(s["grounding"]["issues"][:3])))
        st.dataframe(pd.DataFrame(a["holdings"]).T, use_container_width=True)
        st.caption("Share of portfolio risk by holding (%)")
        st.bar_chart(pd.Series({t: h["risk_contribution_pct"] for t, h in a["holdings"].items()}))
        st.caption("Correlation of daily returns")
        st.dataframe(pd.DataFrame(a["correlation"]), use_container_width=True)

# --------------------------------------------------------------------------- evaluation
with tab_ev:
    from finsight import eval as ev_mod

    st.write("Benchmark FinSight across a universe of stocks: approval rate, grounded figures, revisions, cost.")
    e1, e2 = st.columns([2, 1])
    universe = e1.selectbox("Universe", list(ev_mod.UNIVERSES), index=0)
    if e2.button("Run eval"):
        with st.spinner(f"Evaluating {len(ev_mod.UNIVERSES[universe])} tickers…"):
            res = ev_mod.run_eval(ev_mod.UNIVERSES[universe], settings=settings)
            ev_mod.save(res, "reports")
        st.session_state["eval"] = res
    if "eval" in st.session_state:
        st.markdown(ev_mod.to_markdown(st.session_state["eval"]))
    saved = sorted(Path("reports").glob("eval_*.json"))
    if saved:
        st.subheader("Saved runs")
        st.markdown(ev_mod.compare_markdown([json.loads(p.read_text(encoding="utf-8")) for p in saved[-8:]]))

# --------------------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**How it works.** A Planner reads your question and picks analysts (Researcher, Quant, Peers, Sentiment,
Filings), which run in parallel. A Risk officer applies transparent rules plus judgement. The Writer drafts a
memo and the Critic verifies it. Every figure must appear in the evidence its sentence cites, to the precision
written. Drafts are rejected until they pass or the revision budget runs out.

**Safety.** Tool data is scanned for prompt-injection attempts and redacted before any model sees it, and
memos with buy/sell instructions or no disclaimer are rejected.

Source: [github.com/batsyahan/FinSight](https://github.com/batsyahan/FinSight)
""")
