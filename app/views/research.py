"""Company Research workspace: price, ratios, agent verdict, news, takeaways, financials, risks, memo."""

import streamlit as st

from finsight.data import sec
from finsight.export import memo_to_html_fragment
from finsight.tools.quant import FORMULAS
from finsight.ui import charts
from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()
pages = st.session_state["_pages"]
watch = store.watchlist()
ticker = st.session_state.get("ticker") or (watch[0] if watch else "INFY.NS")

# ---------------------------------------------------------------- header
prof = {}
try:
    prof = shell.profile(s.data, ticker)
except Exception:
    pass
name = D.COMPANIES.get(ticker) or prof.get("name") or ticker
h1, h2 = st.columns([3, 1.3], vertical_alignment="bottom")
h1.html(C.page_header(f"{name} ({ticker})", "", back="Research") +
        C.chips([D.exchange(ticker), prof.get("sector"), prof.get("industry"),
                 f"Mkt cap {C.fmt_money(prof.get('market_cap'), prof.get('price_currency'))}" if prof.get("market_cap") else None]))
with h2:
    b1, b2 = st.columns(2)
    if ticker in watch:
        if b1.button("Remove from watchlist", icon=":material/bookmark_remove:", key="rs_unwatch", use_container_width=True):
            store.remove_from_watchlist(ticker)
            st.rerun()
    elif b1.button("Add to watchlist", icon=":material/bookmark_add:", key="rs_watch", use_container_width=True):
        store.add_to_watchlist(ticker)
        st.rerun()
    if b2.button("Run research", icon=":material/play_arrow:", type="primary", key="rs_run", use_container_width=True):
        st.session_state["agents_ticker"] = ticker
        st.session_state["agents_autostart"] = True
        st.switch_page(pages["Agents"])

run = store.latest_run(ticker)
memo = next(iter(store.memos(ticker, limit=1)), None)
t_sum, t_fin, t_rat, t_bt, t_risk, t_news, t_memo = st.tabs(
    ["Summary", "Financials", "Ratios", "Backtests", "Risks", "News & Filings", "Memo"])

# ---------------------------------------------------------------- summary
with t_sum:
    left, mid, right = st.columns([1.7, 0.9, 0.9], gap="medium")
    with left:
        with st.container(key="glass_price"):
            try:
                q = shell.quote(s.data, ticker, "5y")
                sym = D.currency_symbol(q["currency"])
                a, b = st.columns([1.5, 1], vertical_alignment="center")
                a.html(f'<div style="display:flex;gap:14px;align-items:baseline;flex-wrap:wrap">'
                       f'<span class="fs-big">{C.esc(sym)}{q["price"]:,.2f}</span>{C.change_html(q["change"], q["change_pct"])}'
                       f'</div><div class="fs-xs fs-muted">Last close {C.esc(q["as_of"])} · {C.esc(q["source"])}</div>')
                rng = b.segmented_control("Range", list(D.RANGES), default="1M", key="rs_range",
                                          label_visibility="collapsed") or "1M"
                st.plotly_chart(charts.area(D.price_window(q["series"], rng), ticker, prefix=sym),
                                use_container_width=True, config=charts.CONFIG, key="rs_chart")
                if q["sample"]:
                    st.html(C.note("Synthetic sample prices (offline mode). Switch to live data in Settings."))
            except Exception as e:
                st.html(C.empty(f"Price data unavailable for {ticker} ({type(e).__name__})."))
    with mid:
        with st.container(key="glass_ratios"):
            st.html(C.panel_head("Key Ratios", "chart"))
            try:
                r = shell.ratios(s.data, ticker)
                st.html(C.kv_rows([
                    ("P/E (trailing)", C.fmt_num(r.get("pe_ratio"), 1)), ("ROE", C.fmt_num(r.get("roe_pct"), 1, "%")),
                    ("ROCE", C.fmt_num(r.get("roce_pct"), 1, "%")), ("Debt / equity", C.fmt_num(r.get("debt_to_equity"), 2)),
                    ("EV / EBITDA", C.fmt_num(r.get("ev_to_ebitda"), 1)),
                    ("Dividend yield", C.fmt_num(r.get("dividend_yield_pct"), 2, "%"))]))
                st.html(C.note("n/a = not available or not meaningful; see the Ratios tab for reasons and formulas."))
            except Exception as e:
                st.html(C.empty(f"Fundamentals unavailable ({type(e).__name__})."))
    with right:
        with st.container(key="glass_verdict"):
            st.html(C.panel_head("Agent Verdict", "shield"))
            if memo and run:
                ver = memo["verification"]
                total = ver.get("stats", {}).get("claims") or 0
                ok = ver.get("stats", {}).get("claims_verified") or 0
                label = D.outlook(memo["memo"]) or "No label"
                st.html(f'<div style="margin:2px 0 10px 0">{C.pill(label, {"Positive": "ok", "Neutral": "info", "Cautious": "warn"}.get(label, "neutral"))}'
                        f' <span class="fs-xs fs-muted">outlook from memo v{memo["version"]}</span></div>'
                        + C.progress(ok / total if total else 0, "Claims verified")
                        + C.check_line("Critic approved the memo" if run["approved"] else "Critic did NOT approve the memo",
                                       bool(run["approved"]))
                        + C.check_line(f"{ok} of {total} claims verified against cited evidence", total > 0 and ok == total)
                        + C.check_line(f"{len(run.get('evidence') or {})} evidence items from tools", True))
                st.html(C.note("An outlook label, not a buy/sell recommendation."))
            else:
                st.html(C.empty("No research run yet for this company."))
    n1, n2 = st.columns([1, 1.25], gap="medium")
    with n1:
        with st.container(key="glass_news"):
            st.html(C.panel_head("Recent News", "news"))
            try:
                items = shell.news(s.data, ticker, 5)
                st.html("".join(C.news_item(i.get("title", ""), i.get("publisher") or "", str(i.get("date") or "")[:10],
                                            i.get("link") or "") for i in items) or C.empty("No recent headlines."))
            except Exception as e:
                st.html(C.empty(f"News unavailable ({type(e).__name__})."))
    with n2:
        with st.container(key="glass_takeaways"):
            st.html(C.panel_head("Key Takeaways", "spark", right_html='<span class="fs-xs fs-muted">agent findings, cited</span>'))
            findings = (run or {}).get("findings") or []
            if findings:
                order = {"researcher": 0, "quant": 1, "peers": 2, "risk": 3, "sentiment": 4, "filings": 5}
                top = sorted(findings, key=lambda f: order.get(f["agent"], 9))[:6]
                st.html("".join(f'<div class="fs-bullet"><span>{C.esc(f["text"])} '
                                f'<span class="fs-xs fs-muted">[{C.esc(", ".join(f["citations"]))}]</span></span></div>'
                                for f in top))
            else:
                st.html(C.empty("Run research to generate evidence-backed takeaways."))

# ---------------------------------------------------------------- financials
with t_fin:
    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(key="glass_hist"):
            st.html(C.panel_head("Revenue, earnings and cash flow", "chart"))
            try:
                hist = shell.history(s.data, ticker)
                if hist:
                    st.plotly_chart(charts.history_bars(hist), use_container_width=True, config=charts.CONFIG, key="rs_hist")
                    f = shell.ratios(s.data, ticker)["_fundamentals"]
                    st.html(C.note(f"Annual statements · currency {f.get('statement_currency') or 'unknown'} · "
                                   f"source: {f.get('source', 'provider')}"))
                else:
                    st.html(C.empty("No annual statement history from the provider."))
            except Exception as e:
                st.html(C.empty(f"Statement history unavailable ({type(e).__name__})."))
    with b:
        with st.container(key="glass_bench"):
            bench_name = store.get_setting("benchmark", "NIFTY 50")
            st.html(C.panel_head(f"vs {bench_name} (2 years)", "trend"))
            try:
                stock = shell.prices(s.data, ticker, "2y")
                bench = shell.prices(s.data, D.BENCHMARKS.get(bench_name, "^NSEI"), "2y")
                st.plotly_chart(charts.compare(stock, bench.reindex(stock.index).ffill()), use_container_width=True,
                                config=charts.CONFIG, key="rs_bench")
                st.html(C.note("Both rebased to 100; price return only (dividends excluded)."))
            except Exception as e:
                st.html(C.empty(f"Comparison unavailable ({type(e).__name__})."))

# ---------------------------------------------------------------- ratios
with t_rat:
    with st.container(key="glass_ratio_table"):
        st.html(C.panel_head("Ratios, formulas and provenance", "list"))
        try:
            r = shell.ratios(s.data, ticker)
            f = r["_fundamentals"]
            unavailable = r.get("unavailable") or {}
            labels = {"revenue_growth_pct": "Revenue growth %", "gross_margin_pct": "Gross margin %",
                      "operating_margin_pct": "Operating margin %", "net_margin_pct": "Net margin %",
                      "fcf_margin_pct": "FCF margin %", "roe_pct": "ROE %", "roce_pct": "ROCE %",
                      "debt_to_equity": "Debt / equity", "current_ratio": "Current ratio", "pe_ratio": "P/E",
                      "pb_ratio": "P/B", "ev_to_ebitda": "EV / EBITDA", "dividend_yield_pct": "Dividend yield %"}
            rows = [{"m": lab, "v": C.fmt_num(r.get(k), 2), "f": FORMULAS.get(k, ""),
                     "n": unavailable.get(k, "") if r.get(k) is None else ""} for k, lab in labels.items()]
            rows.append({"m": "EPS (trailing)", "v": C.fmt_num(r.get("eps"), 2), "f": "provider-reported, price currency",
                         "n": ""})
            rows.append({"m": "Free cash flow", "v": C.fmt_money(r.get("free_cash_flow"), f.get("statement_currency")),
                         "f": "operating cash flow - capital expenditure", "n": ""})
            st.html(C.table([("m", "Metric"), ("v", "#Value"), ("f", "Formula"), ("n", "If unavailable, why")], rows))
            st.html(C.note(f"Period: {f.get('period', 'n/a')}. Statements in {f.get('statement_currency')}, "
                           f"quotes in {f.get('price_currency')}. Source: {f.get('source', 'provider')}. "
                           "Missing values are never shown as zero."))
        except Exception as e:
            st.html(C.empty(f"Fundamentals unavailable ({type(e).__name__})."))

# ---------------------------------------------------------------- backtests
with t_bt:
    with st.container(key="glass_bt_list"):
        st.html(C.panel_head("Saved backtests", "play"))
        mine = [b for b in store.backtests(200) if b["ticker"] == ticker]
        if mine:
            st.html(C.table([("strategy", "Strategy"), ("ret", "#Return %"), ("sh", "#Sharpe"), ("dd", "#Max DD %"),
                             ("tr", "#Trades"), ("when", "Run at")],
                            [{"strategy": b["strategy"].replace("_", " "),
                              "ret": C.fmt_num(b["metrics"].get("cumulative_return_pct")),
                              "sh": C.fmt_num(b["metrics"].get("sharpe")),
                              "dd": C.fmt_num(b["metrics"].get("max_drawdown_pct")),
                              "tr": str(b["metrics"].get("trades")), "when": b["created_at"][:16].replace("T", " ")}
                             for b in mine]))
        else:
            st.html(C.empty("No backtests saved for this company yet."))
        if st.button("Open the backtest lab", key="rs_bt", icon=":material/candlestick_chart:"):
            st.session_state["bt_ticker"] = ticker
            st.switch_page(pages["Backtests"])

# ---------------------------------------------------------------- risks
with t_risk:
    with st.container(key="glass_rs_risk"):
        st.html(C.panel_head("Risk flags from the latest run", "alert", "amber"))
        flags = store.flags(ticker=ticker)
        if not flags:
            st.html(C.empty("No flags: either no research run yet, or the latest run raised none."))
        for fl in flags:
            tone = {"high": "bad", "medium": "warn", "low": "info"}[fl["severity"]]
            with st.expander(f"{fl['severity'].upper()} · {fl['category']} · {fl['rule_id'].replace('_', ' ')}"):
                st.html(C.kv_rows([("Rule", f"{fl['metric']} {fl['threshold']}"),
                                   ("Observed value", C.fmt_num(fl["value"], 2)),
                                   ("Evidence", fl["evidence_id"] or "n/a"), ("Detected", fl["detected_at"]),
                                   ("Review status", fl["review_status"])]) + f'<p class="fs-small">{C.esc(fl["message"])}</p>'
                        + C.pill(fl["severity"], tone))
        st.page_link(pages["Risk Monitor"], label="Review all flags", icon=":material/arrow_forward:")

# ---------------------------------------------------------------- news & filings
with t_news:
    a, b = st.columns(2, gap="medium")
    with a:
        with st.container(key="glass_news_all"):
            st.html(C.panel_head("Headlines", "news"))
            try:
                items = shell.news(s.data, ticker, 10)
                st.html("".join(C.news_item(i.get("title", ""), i.get("publisher") or "", str(i.get("date") or "")[:10],
                                            i.get("link") or "") for i in items) or C.empty("No headlines."))
                st.html(C.note("Headlines are untrusted text: prompt-injection attempts are removed before display "
                               "and before any agent sees them."))
            except Exception as e:
                st.html(C.empty(f"News unavailable ({type(e).__name__})."))
    with b:
        with st.container(key="glass_filings"):
            st.html(C.panel_head("Regulatory filings", "file"))
            if sec.is_us_ticker(ticker):
                if sec.is_configured():
                    try:
                        fl = sec.recent_filings(ticker)
                        st.html("".join(C.news_item(f"{x['form']} filed {x['date']}", "SEC EDGAR", x["accession"], x["url"])
                                        for x in fl) or C.empty("No recent 10-K/10-Q/8-K filings found."))
                    except Exception as e:
                        st.html(C.empty(f"SEC EDGAR unavailable ({type(e).__name__})."))
                else:
                    st.html(C.empty("Set SEC_USER_AGENT to list SEC EDGAR filings for US companies."))
            elif ticker.endswith((".NS", ".BO")):
                symbol = ticker.rsplit(".", 1)[0]
                st.html(C.news_item(f"{symbol} on NSE: announcements and annual reports", "nseindia.com", "",
                                    f"https://www.nseindia.com/get-quotes/equity?symbol={symbol}")
                        + C.note("Opens the exchange's own page. Finsight does not scrape NSE. To analyse an annual "
                                 "report, upload it on the Agents page (Filings RAG)."))

# ---------------------------------------------------------------- memo
with t_memo:
    if memo and run:
        state = D.state_from_run(run, memo)
        with st.container(key="glass_rs_memo"):
            st.html(f'<div class="fs-memo">{memo_to_html_fragment(memo["memo"], state)}</div>')
        if st.button("Open in memo library", key="rs_memo_lib", icon=":material/description:"):
            st.session_state["memo_id"] = memo["id"]
            st.switch_page(pages["Memo Library"])
    else:
        st.html(C.empty("No memo yet. Use Run research."))
