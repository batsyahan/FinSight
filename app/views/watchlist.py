"""Watchlist: add/remove companies, see prices, valuation, research and risk status, launch research."""

import streamlit as st

from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()
pages = st.session_state["_pages"]

st.html(C.page_header("Watchlist", "Companies you follow, with live research and risk status"))

with st.container(key="glass_wl_add"):
    a, b = st.columns([3, 1], vertical_alignment="bottom")
    new = a.selectbox("Add a company", [f"{t} · {n}" for t, n in D.COMPANIES.items() if t not in store.watchlist()],
                      index=None, placeholder="Search or type a ticker, e.g. TATAMOTORS.NS", accept_new_options=True,
                      key="wl_new")
    if b.button("Add", type="primary", icon=":material/add:", key="wl_add", use_container_width=True):
        if new:
            try:
                t = store.add_to_watchlist(str(new).split(" · ")[0])
                st.toast(f"Added {t}")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

tickers = tuple(store.watchlist())
with st.container(key="glass_wl_table"):
    st.html(C.panel_head(f"{len(tickers)} companies", "bookmark",
                         right_html=f'<span class="fs-xs fs-muted">{"sample data" if s.data == "sample" else "Yahoo Finance"}</span>'))
    if not tickers:
        st.html(C.empty("Your watchlist is empty."))
    rows = shell.watchlist_rows(s.data, tickers, store.kpis()["runs_total"]) if tickers else []
    widths = [1.3, 1.9, 1.1, 1.0, 0.7, 1, 0.9, 3.2]
    head = st.columns(widths)
    for col, h in zip(head, ["Symbol", "Company", "Price", "1D change", "P/E", "Research", "Risk", ""]):
        col.html(f'<span class="fs-xs fs-muted">{h}</span>')
    tone = {"High": "bad", "Medium": "warn", "Low": "info", "None": "ok", "Unknown": "neutral"}
    for r in rows:
        c = st.columns(widths, vertical_alignment="center")
        c[0].html(f'<b>{C.esc(r["ticker"])}</b>')
        c[1].html(f'<span class="fs-small">{C.esc(r["name"])}</span>')
        c[2].html(C.esc(f'{D.currency_symbol(r.get("currency"))}{C.fmt_num(r["price"])}') if r["price"] is not None
                  else C.pill("No data", "neutral"))
        c[3].html(C.pct_html(r["change_pct"]))
        c[4].html(C.esc(C.fmt_num(r.get("pe"), 1)))
        c[5].html(C.pill(r["research"], {"Verified": "ok", "Unverified": "warn"}.get(r["research"], "neutral")))
        c[6].html(C.pill(r["risk"], tone.get(r["risk"], "neutral")))
        with c[7]:
            x, y, z = st.columns(3)
            if x.button("Open", icon=":material/query_stats:", key=f"wl_open_{r['ticker']}", help="Open research"):
                shell.open_research(r["ticker"])
            if y.button("Run", icon=":material/play_arrow:", key=f"wl_run_{r['ticker']}", help="Run research"):
                st.session_state["agents_ticker"] = r["ticker"]
                st.session_state["agents_autostart"] = True
                st.switch_page(pages["Agents"])
            if z.button("Remove", icon=":material/delete:", key=f"wl_del_{r['ticker']}", help="Remove"):
                store.remove_from_watchlist(r["ticker"])
                st.rerun()
    st.html(C.note("Research and risk status come from each company's latest saved run. A risk label is a "
                   "rule-based signal, not a recommendation."))
