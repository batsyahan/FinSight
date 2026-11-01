"""Memo Library: read memos with hover citations, the Critic's claim-by-claim report, sources and exports."""

import streamlit as st

from finsight.export import export, memo_to_html_fragment
from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui import shell

store, s = shell.store(), shell.settings()
pages = st.session_state["_pages"]

memos = store.memos(limit=200)
st.html(C.page_header("Memo Library", "Investment memos with their sources and verification reports"))
if not memos:
    st.html(C.empty("No memos yet. Start a research run on the Agents page."))
    st.stop()

tickers = sorted({m["ticker"] for m in memos})
f1, f2 = st.columns([1, 3])
tk = f1.selectbox("Company", ["all", *tickers], key="ml_ticker")
shown = [m for m in memos if tk == "all" or m["ticker"] == tk]
ids = [m["id"] for m in shown]
want = st.session_state.get("memo_id")
idx = ids.index(want) if want in ids else 0
label = {m["id"]: f"{m['ticker']} · v{m['version']} · {m['created_at'][:16].replace('T', ' ')} UTC · "
                  f"{'verified' if m['verified'] else 'unverified'}" for m in shown}
mid = f2.selectbox("Memo", ids, index=idx, format_func=label.get, key="ml_pick")
st.session_state["memo_id"] = mid
memo = store.memo(mid)
run = store.run(memo["run_id"]) or {"ticker": memo["ticker"], "evidence": {}, "approved": memo["verified"]}
state = D.state_from_run(run, memo)
ver = memo["verification"]
stats = ver.get("stats", {})

ev = run.get("evidence") or {}
price_ev = next((e for e in ev.values() if e["source"] == "get_price_stats"), None)
fund_ev = next((e for e in ev.values() if e["source"] == "get_fundamentals"), None)
cutoff = price_ev["data"].get("end_date") if price_ev else "n/a"

left, right = st.columns([2.1, 1], gap="medium")
with left:
    with st.container(key="glass_memo_doc"):
        a, b = st.columns([3, 1], vertical_alignment="center")
        a.html(C.brand())
        b.html(f'<div class="fs-small fs-muted" style="text-align:right">{C.esc(memo["created_at"][:10])}</div>')
        st.html(C.chips([memo["ticker"], f"v{memo['version']}", f"Data cutoff {cutoff}",
                         f"Statements FY{fund_ev['data'].get('fiscal_year')}" if fund_ev else None,
                         "Sample data" if run.get("data") == "sample" else (run.get("data") or None),
                         f"LLM: {run.get('llm')}" if run.get("llm") else None]))
        st.html(f'<div class="fs-memo">{memo_to_html_fragment(memo["memo"], state)}</div>')
        st.html(C.note("Hover a citation to see its evidence. Figures are copied from tool outputs and checked "
                       "against the cited evidence; interpretations are the agents' views, not facts."))
    with st.container(key="glass_method"):
        st.html(C.panel_head("Methodology and limitations", "list", "slate"))
        st.html(C.bullets([
            "Calculations (returns, ratios, backtests, risk rules) are deterministic Python on provider data; the "
            "language model never does the arithmetic.",
            "Each figure must match the evidence its sentence cites, to the precision written; the Critic sends "
            "drafts back until they pass or the revision budget runs out.",
            "Qualitative statements are reviewed by the LLM Critic only and are interpretations, not verified facts.",
            "Data comes from " + ("synthetic sample data (offline)." if run.get("data") == "sample"
                                  else "Yahoo Finance via yfinance, a third-party source; coverage and timeliness vary."),
            "No price targets or return forecasts are produced. This is research, not investment advice.",
        ]))

with right:
    with st.container(key="glass_verif"):
        st.html(C.panel_head("Verification (Critic Agent)", "shield"))
        total, ok = stats.get("claims") or 0, stats.get("claims_verified") or 0
        rounds = D.critic_rounds(run)
        rejected = sum(1 for r in rounds if not r.get("approved"))
        st.html(C.check_line("Critic approved the memo" if run.get("approved") else "Critic did NOT approve the memo",
                             bool(run.get("approved")))
                + C.check_line(f"{ok} of {total} claims verified against cited evidence", total > 0 and ok == total)
                + C.check_line(f"{stats.get('unsupported_numbers', 0)} unsupported figure(s) in the final memo",
                               not stats.get("unsupported_numbers"))
                + C.check_line(f"{rejected} draft(s) rejected before approval" if run.get("approved")
                               else f"{rejected} draft(s) rejected", True)
                + C.check_line("Memo revised and approved" if run.get("approved") and rejected
                               else ("Approved on first draft" if run.get("approved") else "Unresolved issues remain"),
                               bool(run.get("approved"))))
        if not memo["verified"]:
            for issue in ver.get("issues", [])[:6]:
                st.html(C.check_line(issue, False))
    with st.container(key="glass_sources"):
        st.html(C.panel_head(f"Sources ({len(ev)})", "list"))
        rows = []
        for k, e in ev.items():
            rows.append(f'<div class="fs-row"><span><b>{C.esc(k)}</b> <span class="fs-small">{C.esc(e["title"])}</span>'
                        f'</span><span class="fs-xs fs-muted">{C.esc(e["source"])}</span></div>')
            for item in (e["data"].get("items") or [])[:5] if e["source"] == "get_news" else []:
                if str(item.get("link", "")).startswith("http"):
                    rows.append(C.news_item(item.get("title", ""), item.get("publisher") or "", "", item["link"]))
            for res in (e["data"].get("results") or [])[:2] if e["source"] == "search_filings" else []:
                rows.append(f'<div class="fs-note">{C.esc(res["doc"])}, page {res["page"]}</div>')
        st.html("".join(rows) or C.empty("No evidence stored for this run."))
    with st.container(key="glass_exports"):
        st.html(C.panel_head("Export", "file"))
        stem = f"{memo['ticker'].replace('.', '_')}_memo_v{memo['version']}"
        st.download_button("Markdown", export(state, "md")[0], f"{stem}.md", key="ml_md", use_container_width=True)
        st.download_button("HTML (hover citations)", export(state, "html")[0], f"{stem}.html", key="ml_html",
                           use_container_width=True)
        try:
            from finsight.export import to_pdf

            st.download_button("PDF with verification report", to_pdf(state, ver), f"{stem}.pdf", key="ml_pdf",
                               type="primary", use_container_width=True)
        except Exception as e:
            st.html(C.note(f"PDF unavailable: {type(e).__name__}"))
        if st.button("Regenerate (new research run)", icon=":material/refresh:", key="ml_regen", use_container_width=True):
            st.session_state["agents_ticker"] = memo["ticker"]
            st.session_state["agents_autostart"] = True
            st.switch_page(pages["Agents"])

with st.expander(f"Claim-by-claim verification report ({total} claims)"):
    claims = ver.get("claims", [])
    tone = {"verified": "ok", "qualitative": "info", "unsupported": "bad", "uncited": "bad", "bad_citation": "bad"}
    st.html(C.table([("id", "Claim"), ("text", "Text"), ("type", "Type"), ("cites", "Cites"), ("out", "Outcome"),
                     ("why", "Explanation")],
                    [{"id": c["id"], "text": c["text"][:220], "type": c["type"], "cites": ", ".join(c["citations"]),
                      "out": C.pill(c["outcome"], tone.get(c["outcome"], "neutral")), "why": c["explanation"]}
                     for c in claims], html_cols={"out"}) if claims else C.empty("No claims recorded."))

with st.expander("Version history"):
    hist = store.memos(memo["ticker"], 50)
    st.html(C.table([("v", "Version"), ("d", "Created"), ("st", "Status"), ("c", "Claims verified")],
                    [{"v": f"v{m['version']}", "d": m["created_at"][:16].replace("T", " "),
                      "st": C.pill("verified" if m["verified"] else "unverified", "ok" if m["verified"] else "warn"),
                      "c": f"{m['verification'].get('stats', {}).get('claims_verified', 0)} / "
                           f"{m['verification'].get('stats', {}).get('claims', 0)}"} for m in hist], html_cols={"st"}))
