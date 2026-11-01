"""Headless tests of the real multipage web app (Streamlit AppTest). Run where streamlit + plotly exist (CI)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
VIEWS = ["overview", "research", "agents", "backtests", "watchlist", "risk", "memos", "sources", "settings"]


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("FINSIGHT_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("FINSIGHT_LLM", "mock")
    monkeypatch.setenv("FINSIGHT_DATA", "sample")
    from finsight.ui import shell

    shell.store.clear()  # fresh database per test
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception, at.exception
    return at


def test_every_page_renders_on_an_empty_database(app):
    for view in VIEWS:
        app.switch_page(f"views/{view}.py").run()
        assert not app.exception, (view, app.exception)


def test_research_run_end_to_end(app):
    app.switch_page("views/agents.py").run()
    app.button(key="ag_start").click().run()
    assert not app.exception, app.exception
    from finsight.ui import shell

    store = shell.store()
    runs = store.runs()
    assert runs and runs[0]["status"] == "completed" and runs[0]["approved"] == 1
    assert store.memos()[0]["verified"] == 1
    for view in VIEWS:  # every page renders with real data present
        app.switch_page(f"views/{view}.py").run()
        assert not app.exception, (view, app.exception)


def test_backtest_and_watchlist_interactions(app):
    app.switch_page("views/backtests.py").run()
    app.button(key="bt_run").click().run()
    assert not app.exception, app.exception
    from finsight.ui import shell

    assert len(shell.store().backtests()) == 1
    app.switch_page("views/watchlist.py").run()
    app.button(key="wl_del_TCS.NS").click().run()
    assert not app.exception and "TCS.NS" not in shell.store().watchlist()
