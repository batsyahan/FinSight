"""FinSight web UI.  Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a clone without installing

from finsight.config import Settings
from finsight.data.providers import get_provider
from finsight.graph.workflow import run_analysis

st.set_page_config(page_title="FinSight", page_icon="📈", layout="wide")
st.title("📈 FinSight")
st.caption("A team of AI agents researches a stock, runs quant analysis, flags risks and writes a cited, "
           "self-verified memo. Educational research only, not financial advice.")

with st.sidebar:
    st.header("Settings")
    base = Settings.from_env()
    llm = st.selectbox("LLM backend", ["mock", "anthropic", "openai"],
                       index=["mock", "anthropic", "openai"].index(base.llm) if base.llm in ("mock", "anthropic", "openai") else 0,
                       help="mock runs offline with no API key")
    model = st.text_input("Model", base.model, help="Leave blank for the default")
    if llm == "anthropic" and not base.anthropic_api_key:
        key = st.text_input("Anthropic API key", type="password")
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
    if llm == "openai":
        url = st.text_input("Base URL", base.openai_base_url)
        key = st.text_input("API key", type="password", value="")
        os.environ["OPENAI_BASE_URL"] = url
        if key:
            os.environ["OPENAI_API_KEY"] = key
    data = st.selectbox("Market data", ["sample", "yfinance"], index=0 if base.data != "yfinance" else 1,
                        help="sample = synthetic offline data; yfinance = real Yahoo Finance data")
    max_rev = st.slider("Max Writer↔Critic revisions", 0, 4, base.max_revisions)

c1, c2 = st.columns([1, 3])
ticker = c1.text_input("Ticker", "INFY.NS" if data == "yfinance" else "DEMO",
                       help="NSE: add .NS (e.g. TCS.NS), BSE: .BO")
question = c2.text_input("Investor question", "Should a long-term investor consider this stock?")
go = st.button("Run analysis", type="primary")

if go and ticker:
    settings = replace(Settings.from_env(), llm=llm, model=model, data=data, max_revisions=max_rev)
    labels = {"researcher": "🔎 Researcher", "quant": "📊 Quant", "risk": "⚠️ Risk officer",
              "writer": "✍️ Writer", "critic": "🧐 Critic"}
    with st.status("Agents at work…", expanded=True) as status:
        def on_step(name, state):
            if name == "critic":
                msg = "approved the memo ✅" if state.approved else f"sent it back ({len(state.critique['issues'])} issues)"
            elif name == "writer":
                msg = f"wrote draft {state.revisions + 1}"
            else:
                msg = f"done, {len(state.evidence.items)} evidence items so far"
            st.write(f"{labels[name]}: {msg}")

        try:
            state = run_analysis(ticker, question, settings=settings, on_step=on_step)
        except Exception as e:
            status.update(label="Failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()
        status.update(label="Approved by the Critic" if state.approved else "Finished (not approved)",
                      state="complete" if state.approved else "error")

    tab_memo, tab_chart, tab_evidence, tab_trace = st.tabs(["Memo", "Price chart", "Evidence", "Agent trace"])
    with tab_memo:
        if not state.approved:
            st.warning("The Critic did not approve this memo. Outstanding issues:\n\n- "
                       + "\n- ".join(state.critique.get("issues", [])))
        st.markdown(state.memo)
        st.download_button("Download memo (.md)", state.memo, file_name=f"{state.ticker}_memo.md")
    with tab_chart:
        try:
            prices = get_provider(data).prices(state.ticker, "2y").to_frame()
            prices["SMA 50"] = prices["close"].rolling(50).mean()
            prices["SMA 200"] = prices["close"].rolling(200).mean()
            st.line_chart(prices)
        except Exception as e:
            st.info(f"Chart unavailable: {e}")
    with tab_evidence:
        for ev in state.evidence.items.values():
            with st.expander(f"[{ev.id}] {ev.title}"):
                st.json(ev.data)
    with tab_trace:
        st.code(json.dumps(state.trace, indent=2, default=str), language="json")
