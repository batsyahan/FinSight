"""finsight web app.   streamlit run app/streamlit_app.py

Multipage app on st.navigation. The shell (finsight.ui.shell) draws the icon rail and the top bar;
each page lives in app/views/ and uses shared components, charts and data services.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # run from a clone without installing

import streamlit as st  # noqa: E402

st.set_page_config(page_title="finsight", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="expanded")

from finsight.ui import shell  # noqa: E402

VIEWS = Path(__file__).parent / "views"
PAGES = {
    "Overview": st.Page(VIEWS / "overview.py", title="Overview", icon=":material/space_dashboard:", default=True),
    "Research": st.Page(VIEWS / "research.py", title="Research", icon=":material/query_stats:"),
    "Agents": st.Page(VIEWS / "agents.py", title="Agents", icon=":material/hub:"),
    "Backtests": st.Page(VIEWS / "backtests.py", title="Backtests", icon=":material/candlestick_chart:"),
    "Watchlist": st.Page(VIEWS / "watchlist.py", title="Watchlist", icon=":material/bookmark:"),
    "Risk Monitor": st.Page(VIEWS / "risk.py", title="Risk Monitor", icon=":material/warning:"),
    "Memo Library": st.Page(VIEWS / "memos.py", title="Memo Library", icon=":material/description:"),
    "Data Sources": st.Page(VIEWS / "sources.py", title="Data Sources", icon=":material/database:"),
    "Settings": st.Page(VIEWS / "settings.py", title="Settings", icon=":material/settings:"),
}

page = st.navigation(list(PAGES.values()), position="hidden")
shell.render_shell(PAGES, page.title)
page.run()
