"""Research Overview: KPIs, market overview, latest agent pipeline, watchlist, risk summary, memos, health."""

import streamlit as st

from finsight.ui import charts
from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()
pages = st.session_state["_pages"]
k = store.kpis()

c0, c1, c2, c3, c4 = st.columns([1.35, 1, 1, 1, 1], vertical_alignment="center")
c0.html(C.page_header("Research Overview", "Multi-agent investment research, backtesting and risk intelligence"))
c1.html(C.kpi(k["companies_tracked"], "Companies tracked", "chart"))
c2.html(C.kpi(k["runs_this_month"], "Research runs this month", "layers", "slate"))
c3.html(C.kpi(k["backtests"], "Backtests completed", "play"))
c4.html(C.kpi(k["open_flags"], "Open risk flags", "alert", "coral",
              f'<div class="s fs-down">({k["open_high"]} high)</div>' if k["open_high"] else ""))

left, right = st.columns([1.6, 1], gap="medium")

with left:
    with st.container(key="glass_market"):
        bench_name = store.get_setting("benchmark", "NIFTY 50")
        h1, h2 = st.columns([1.4, 1], vertical_alignment="center")
        h1.html(C.panel_head("Market Overview", "chart"))
        rng = h2.segmented_control("Range", list(D.RANGES), default="1M", key="ov_range",
                                   label_visibility="collapsed") or "1M"
        symbol = D.BENCHMARKS.get(bench_name, "^NSEI")
        try:
            series = shell.prices(s.data, symbol, "5y" if rng == "5Y" else "2y")
            win = D.price_window(series, rng)
            last, first = float(win.iloc[-1]), float(win.iloc[0])
            prev = float(series.iloc[-2])
            st.html(f'<div style="display:flex;gap:18px;align-items:baseline;flex-wrap:wrap">'
                    f'<span class="fs-muted" style="font-weight:600">{C.esc(bench_name)}</span>'
                    f'<span class="fs-big">{last:,.2f}</span>{C.change_html(last - prev, (last / prev - 1) * 100)}'
                    f'<span class="fs-small fs-muted">{rng}: {C.pct_html((last / first - 1) * 100)}</span></div>')
            st.plotly_chart(charts.area(win, bench_name), use_container_width=True, config=charts.CONFIG,
                            key="ov_chart")
            st.html(C.note(f"Source: {'synthetic sample series (offline mode), not real index levels' if s.data == 'sample' else 'Yahoo Finance via yfinance'} · last close {series.index[-1].date()} · daily closes"))
        except Exception as e:
            st.html(C.empty(f"Benchmark data unavailable ({type(e).__name__}). Check Data Sources."))

    wl, rs = st.columns([1.55, 1], gap="medium")
    with wl:
        with st.container(key="glass_watch"):
            st.html(C.panel_head("Watchlist", "bookmark"))
            wtickers = tuple(store.watchlist())
            rows = shell.watchlist_rows(s.data, wtickers, k["runs_total"]) if wtickers else []
            if not rows:
                st.html(C.empty("Your watchlist is empty. Add companies on the Watchlist page."))
            else:
                tone = {"High": "bad", "Medium": "warn", "Low": "info", "None": "ok", "Unknown": "neutral"}
                body = [{
                    "sym": r["ticker"].replace(".NS", "").replace(".BO", ""),
                    "price": C.fmt_num(r["price"]),
                    "chg": C.pct_html(r["change_pct"]),
                    "pe": C.fmt_num(r.get("pe"), 1),
                    "risk": C.pill(r["risk"], tone.get(r["risk"], "neutral")),
                    "res": C.pill(r["research"], {"Verified": "ok", "Unverified": "warn"}.get(r["research"], "neutral")),
                } for r in rows]
                st.html(C.table([("sym", "Symbol"), ("price", "#Price"), ("chg", "#1D change"), ("pe", "#P/E"),
                                 ("res", "Research"), ("risk", "Risk")], body, html_cols={"chg", "risk", "res"}))
                st.page_link(pages["Watchlist"], label="Manage watchlist", icon=":material/arrow_forward:")
    with rs:
        with st.container(key="glass_risk"):
            st.html(C.panel_head("Risk Summary", "alert", "amber"))
            flags = store.flags(status="open")
            sev = {x: sum(f["severity"] == x for f in flags) for x in ("high", "medium", "low")}
            st.plotly_chart(charts.donut([("High", sev["high"], charts.CORAL), ("Medium", sev["medium"], charts.AMBER),
                                          ("Low", sev["low"], charts.SLATE)], str(len(flags)), "Open flags"),
                            use_container_width=True, config=charts.CONFIG, key="ov_donut")
            st.html(C.kv_rows([("High", str(sev["high"])), ("Medium", str(sev["medium"])), ("Low", str(sev["low"]))]))
            if not flags:
                st.html(C.note("Flags appear after a research run raises them."))
            st.page_link(pages["Risk Monitor"], label="Open risk monitor", icon=":material/arrow_forward:")

with right:
    with st.container(key="glass_pipeline"):
        run = store.latest_run()
        sub = f'<span class="fs-small fs-muted">{C.esc(run["ticker"])} · {C.esc(run["created_at"][:16].replace("T", " "))} UTC</span>' if run else ""
        st.html(C.panel_head("Agent Pipeline", "bot", right_html=sub))
        if run:
            st.html(C.timeline(D.pipeline_from_run(run)))
        else:
            st.html(C.empty("No research runs yet. Start one to see each agent's real execution here."))
        if st.button("Start a research run", type="primary", key="ov_run", use_container_width=True):
            st.switch_page(pages["Agents"])

    with st.container(key="glass_memos"):
        st.html(C.panel_head("Recent Investment Memos", "file"))
        memos = store.memos(limit=4)
        if not memos:
            st.html(C.empty("No memos yet."))
        for m in memos:
            a, b = st.columns([3, 1], vertical_alignment="center")
            a.html(f'<div style="font-weight:600;font-size:13.5px">{C.esc(m["ticker"])} '
                   f'<span class="fs-muted fs-small">v{m["version"]} · {C.esc(m["created_at"][:10])}</span></div>'
                   + C.pill("Verified" if m["verified"] else "Unverified", "ok" if m["verified"] else "warn"))
            if b.button("Open", key=f"ov_memo_{m['id']}", use_container_width=True):
                st.session_state["memo_id"] = m["id"]
                st.switch_page(pages["Memo Library"])

    with st.container(key="glass_health"):
        st.html(C.panel_head("Data Sources", "database", "slate"))
        st.html("".join(
            f'<div class="fs-row"><span class="k">{C.esc(h["name"])}</span>'
            f'{C.pill(h["status"].title(), D.HEALTH_LABELS.get(h["status"], "neutral"))}</div>'
            for h in D.health_static(s, store)))
        st.page_link(pages["Data Sources"], label="Details and live checks", icon=":material/arrow_forward:")
