"""Data Sources: configuration and verified health of every integration."""

import streamlit as st

from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()

st.html(C.page_header("Data Sources", "Integration status. Nothing is shown as connected until a live check succeeds."))
a, b = st.columns([3, 1], vertical_alignment="bottom")
a.html(C.note("Live checks call external services (market data, your LLM provider, SEC EDGAR) and may use a few "
              "tokens, so they run only when you ask."))
if b.button("Run live checks", type="primary", icon=":material/sync:", key="ds_check", use_container_width=True):
    with st.spinner("Checking services…"):
        D.run_health_checks(s, store)
    st.rerun()

HINT = {
    "Market data": "FINSIGHT_DATA=yfinance for live Yahoo Finance data; sample = synthetic offline data.",
    "SEC EDGAR": "SEC_USER_AGENT=\"Your Name you@example.com\" (required by SEC fair-access rules).",
    "NSE data": "No official public API. Use exchange pages for announcements; upload annual reports for RAG.",
    "Retrieval index": "Optional: FINSIGHT_EMBED_MODEL (+ OPENAI_* or FINSIGHT_EMBED_*) for hybrid search.",
    "LLM provider": "FINSIGHT_LLM=openai|anthropic|mock, FINSIGHT_MODEL, and the provider's API key.",
    "Backtest engine": "Predefined strategies only; nothing to configure.",
}
for h in D.health_static(s, store):
    with st.container(key=f"glass_ds_{h['name'].replace(' ', '_').lower()}"):
        x, y = st.columns([3, 1], vertical_alignment="center")
        x.html(f'<div style="font-weight:600">{C.esc(h["name"])}</div><div class="fs-small fs-muted">{C.esc(h["detail"])}</div>'
               + C.note(("Last checked " + h["checked_at"].replace("T", " ") + " UTC") if h["checked_at"] else
                        "Configuration: " + HINT.get(h["name"], "")))
        y.html(f'<div style="text-align:right">{C.pill(h["status"].title(), D.HEALTH_LABELS.get(h["status"], "neutral"))}</div>')

st.html(C.note("Statuses: Connected = a live check succeeded · Degraded = reachable but stale or partial · "
               "Unavailable = the last check failed · Local = runs inside this app · Not configured = needs setup · "
               "Not checked = configured but not yet verified."))
