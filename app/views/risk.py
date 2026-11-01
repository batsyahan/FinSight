"""Risk Monitor: evidence-backed, rule-based flags with filters and a review workflow."""

import streamlit as st

from finsight.ui import components as C
from finsight.ui import shell

store, s = shell.store(), shell.settings()

WHY = {
    "Excessive leverage": "High debt relative to equity raises refinancing risk and amplifies losses in a downturn.",
    "Weak cash-flow conversion": "Negative free cash flow means the business consumes cash; growth then depends on "
                                 "borrowing or new equity.",
    "Financial deterioration": "Shrinking revenue, losses or tight liquidity can signal a weakening business.",
    "Unusual historical drawdown": "Deep past declines show how much an investor could have lost from a peak.",
    "Unusual price behaviour": "High volatility or returns below the risk-free rate mean poor risk-adjusted outcomes.",
    "Valuation concern": "A very high multiple prices in a lot of future growth, leaving little room for disappointment.",
    "News and disclosures": "A clearly negative news flow can precede fundamental or regulatory problems.",
}

st.html(C.page_header("Risk Monitor", "Rule-based warning signals from research runs, each with its evidence"))
all_flags = store.flags(latest_only=True)
k1, k2, k3, k4 = st.columns(4)
k1.html(C.kpi(sum(f["review_status"] == "open" for f in all_flags), "Open flags", "alert", "coral"))
k2.html(C.kpi(sum(f["severity"] == "high" and f["review_status"] == "open" for f in all_flags), "High severity, open",
              "alert", "coral"))
k3.html(C.kpi(sum(f["severity"] == "medium" and f["review_status"] == "open" for f in all_flags), "Medium, open",
              "alert", "amber"))
k4.html(C.kpi(sum(f["review_status"] != "open" for f in all_flags), "Reviewed or dismissed", "check"))

with st.container(key="glass_risk_filters"):
    a, b, c, d, e = st.columns([1, 1, 1.2, 1.4, 1])
    status = a.selectbox("Status", ["open", "reviewed", "dismissed", "all"], key="rk_status")
    severity = b.selectbox("Severity", ["all", "high", "medium", "low"], key="rk_sev")
    tickers = sorted({f["ticker"] for f in all_flags})
    ticker = c.selectbox("Company", ["all", *tickers], key="rk_ticker")
    cats = sorted({f["category"] for f in all_flags})
    category = d.selectbox("Category", ["all", *cats], key="rk_cat")
    sort = e.selectbox("Sort by", ["severity", "newest", "company"], key="rk_sort")

flags = [f for f in all_flags
         if (status == "all" or f["review_status"] == status) and (severity == "all" or f["severity"] == severity)
         and (ticker == "all" or f["ticker"] == ticker) and (category == "all" or f["category"] == category)]
rank = {"high": 0, "medium": 1, "low": 2}
flags.sort(key={"severity": lambda f: (rank[f["severity"]], f["ticker"]),
                "newest": lambda f: f["detected_at"], "company": lambda f: (f["ticker"], rank[f["severity"]])}[sort],
           reverse=sort == "newest")

st.html(C.note("Flags are automated signals from transparent rules. They do not imply misconduct or wrongdoing, "
               "and reviewing a flag never deletes its evidence."))
if not flags:
    st.html(C.empty("No flags match. Flags appear after research runs raise them."))
runs_cache: dict = {}
for f in flags:
    tone = {"high": "bad", "medium": "warn", "low": "info"}[f["severity"]]
    with st.expander(f"{f['ticker']} · {f['severity'].upper()} · {f['category']} · {f['rule_id'].replace('_', ' ')}"
                     f"{'' if f['review_status'] == 'open' else ' · ' + f['review_status']}"):
        run = runs_cache.setdefault(f["run_id"], store.run(f["run_id"]))
        left, right = st.columns([1.3, 1], gap="medium")
        with left:
            st.html(C.pill(f["severity"], tone) + " " + C.pill(f["review_status"], "ok" if f["review_status"] != "open"
                                                                 else "neutral")
                    + f'<p style="margin-top:8px">{C.esc(f["message"])}</p>'
                    + f'<p class="fs-small fs-muted"><b>Why it matters.</b> {C.esc(WHY.get(f["category"], ""))}</p>')
            st.html(C.kv_rows([("Rule", f"{f['metric']} {f['threshold']}"), ("Observed value", C.fmt_num(f["value"], 2)),
                               ("Evidence id", f["evidence_id"] or "n/a"), ("Detected", f["detected_at"]),
                               ("Data", (run or {}).get("data") or "unknown")]))
            if (run or {}).get("data") == "sample":
                st.html(C.note("Raised on synthetic sample data: illustrates the rule, says nothing about the real company."))
        with right:
            ev = ((run or {}).get("evidence") or {})
            metric_src = next((e for e in ev.values() if f["metric"] in (e.get("data") or {})), None)
            if metric_src:
                st.caption(f"Supporting calculation: {metric_src['title']} ({metric_src['source']})")
                st.json({k: v for k, v in metric_src["data"].items() if not isinstance(v, (dict, list))}, expanded=False)
            b1, b2, b3 = st.columns(3)
            if f["review_status"] != "reviewed" and b1.button("Mark reviewed", key=f"rk_rev_{f['id']}"):
                store.set_flag_status(f["id"], "reviewed")
                st.rerun()
            if f["review_status"] != "dismissed" and b2.button("Dismiss", key=f"rk_dis_{f['id']}"):
                store.set_flag_status(f["id"], "dismissed")
                st.rerun()
            if f["review_status"] != "open" and b3.button("Reopen", key=f"rk_open_{f['id']}"):
                store.set_flag_status(f["id"], "open")
                st.rerun()
