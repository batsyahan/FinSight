"""App shell: global CSS, cached services, the left icon rail and the top navigation bar.

Navigation uses st.navigation / st.page_link (official APIs). The current page is drawn as a static
"active" pill instead of a link, so no Streamlit-internal attributes are needed for highlighting.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from finsight.store import Store
from finsight.ui import components as C
from finsight.ui import data as D
from finsight.ui.theme import css

TOP_PAGES = ["Overview", "Research", "Agents", "Backtests", "Watchlist", "Memos"]


@st.cache_resource
def store() -> Store:
    return Store()


def settings():
    return D.effective_settings(store())


def provider():
    return D.provider_for(settings())


def mode() -> str:
    return settings().data


@st.cache_data(ttl=900, show_spinner=False)
def prices(data_mode: str, ticker: str, period: str = "2y"):
    return D.get_provider(data_mode).prices(ticker, period)


@st.cache_data(ttl=900, show_spinner=False)
def quote(data_mode: str, ticker: str, period: str = "1y") -> dict[str, Any]:
    q = D.quote(D.get_provider(data_mode), ticker, period)
    return q


@st.cache_data(ttl=3600, show_spinner=False)
def ratios(data_mode: str, ticker: str) -> dict[str, Any]:
    return D.ratios(D.get_provider(data_mode), ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def profile(data_mode: str, ticker: str) -> dict[str, Any]:
    return D.get_provider(data_mode).profile(ticker)


@st.cache_data(ttl=1800, show_spinner=False)
def news(data_mode: str, ticker: str, limit: int = 6) -> list[dict[str, Any]]:
    return D.news(D.get_provider(data_mode), ticker, limit)


@st.cache_data(ttl=3600, show_spinner=False)
def history(data_mode: str, ticker: str) -> list[dict[str, Any]]:
    return D.get_provider(data_mode).financial_history(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def watchlist_rows(data_mode: str, tickers: tuple[str, ...], stamp: int) -> list[dict[str, Any]]:
    return D.watchlist_rows(store(), D.get_provider(data_mode))


def html(markup: str) -> None:
    st.html(markup)


def open_research(ticker: str) -> None:
    st.session_state["ticker"] = ticker
    st.switch_page(st.session_state["_pages"]["Research"])


def _on_search() -> None:
    """Callback: resolve the typed or picked value to a ticker, then navigate after the rerun starts."""
    raw = st.session_state.get("global_search") or ""
    st.session_state["global_search"] = None
    hits = D.search(raw.split(" · ")[0])
    if hits:
        st.session_state["ticker"] = hits[0][0]
        st.session_state["_goto"] = "Research"
    else:
        st.session_state["_search_error"] = f"'{raw}' is not a known company or valid ticker"


def render_shell(pages: dict[str, Any], current: str) -> None:
    st.html(css())
    st.session_state["_pages"] = pages
    s = settings()
    # ---- left rail
    with st.sidebar:
        with st.container(key="rail"):
            for title, page in pages.items():
                if title == current:
                    st.html(f'<div class="fs-rail-active">{C.esc(title)}</div>')
                else:
                    st.page_link(page, label=title, icon=page.icon)
        st.divider()
        st.caption("Mode")
        st.html(C.pill("Sample data" if s.data == "sample" else "Live data (Yahoo Finance)",
                       "warn" if s.data == "sample" else "ok") + " " +
                C.pill("Mock LLM" if s.llm == "mock" else f"LLM: {s.llm}", "warn" if s.llm == "mock" else "ok"))
        st.caption("Research, not financial advice.")
    # ---- top bar
    with st.container(key="topbar"):
        c_brand, c_nav, c_search, c_icons = st.columns([1.1, 4.6, 2.2, 1.1], vertical_alignment="center")
        c_brand.html(C.brand())
        with c_nav:
            with st.container(key="topnav"):
                cols = st.columns(len(TOP_PAGES))
                for col, title in zip(cols, TOP_PAGES):
                    key = "Memo Library" if title == "Memos" else title
                    with col:
                        if key == current:
                            st.html(f'<span class="fs-pill-active">{C.esc(title)}</span>')
                        else:
                            st.page_link(pages[key], label=title)
        with c_search:
            options = [f"{t} · {n}" for t, n in D.COMPANIES.items()]
            st.selectbox("Search", options, index=None, placeholder="Search company or ticker",
                         label_visibility="collapsed", key="global_search", accept_new_options=True,
                         on_change=_on_search)
        with c_icons:
            k = store().kpis()
            a, b = st.columns(2, vertical_alignment="center")
            if a.button(f"{k['open_high']}" if k["open_high"] else "0", icon=":material/notifications:",
                        key="top_flags", help="Open high-severity risk flags"):
                st.switch_page(pages["Risk Monitor"])
            b.html(C.avatar(store().get_setting("user_initials", "FS")))
    if err := st.session_state.pop("_search_error", None):
        st.toast(err)
    if st.session_state.pop("_goto", None) == "Research" and current != "Research":
        st.switch_page(pages["Research"])
