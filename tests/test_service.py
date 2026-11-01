import json
import time

import pandas as pd
import pytest

from finsight.config import Settings
from finsight.data.cache import CachedProvider, FileBackend
from finsight.data.providers import SampleProvider
from finsight.graph.workflow import run_analysis
from finsight.jobs import JobManager
from finsight.watch import diff, run_watch

OFFLINE = Settings(llm="mock", data="sample", max_revisions=2)


def offline_runner(ticker, question=None, **kw):
    return run_analysis(ticker, question, settings=OFFLINE, engine_name="builtin", **kw)


# --------------------------------------------------------------------------- cache
class CountingProvider(SampleProvider):
    name = "yfinance"

    def __init__(self):
        self.calls = 0

    def prices(self, ticker, period="2y"):
        self.calls += 1
        return super().prices(ticker, period)


def test_cache_round_trips_series_and_avoids_refetching(tmp_path):
    inner = CountingProvider()
    cp = CachedProvider(inner, FileBackend(tmp_path))
    a = cp.prices("DEMO", "1y")
    b = CachedProvider(CountingProvider(), FileBackend(tmp_path)).prices("DEMO", "1y")  # fresh process, same disk
    pd.testing.assert_series_equal(a, b, check_freq=False)
    cp.prices("DEMO", "1y")
    assert inner.calls == 1 and cp.hits == 1 and cp.misses == 1
    assert cp.fundamentals("DEMO") == SampleProvider().fundamentals("DEMO")


def test_cache_entries_expire(tmp_path, monkeypatch):
    backend = FileBackend(tmp_path)
    backend.set("k", json.dumps(1), ttl=10)
    assert backend.get("k") == "1"
    monkeypatch.setattr(time, "time", lambda: 10**12)
    assert backend.get("k") is None


# --------------------------------------------------------------------------- jobs
def test_job_runs_and_reports_progress():
    jm = JobManager(offline_runner, workers=1)
    job = jm.submit("analyze", {"ticker": "DEMO"})
    done = jm.wait(job.id)
    assert done.status == "done" and done.result["approved"]
    assert [p["node"] for p in done.progress][0] == "planner" and done.progress[-1]["node"] == "critic"
    assert jm.list()[0]["id"] == job.id


def test_job_human_review_over_the_api_flow():
    jm = JobManager(offline_runner, workers=1)
    job = jm.submit("analyze", {"ticker": "DEMO", "review": True})
    jm.wait(job.id, until=("awaiting_review",))
    assert "## Summary" in job.public()["review_draft"]["memo"]
    with pytest.raises(ValueError):
        jm.submit_review(job.id, {"action": "publish"})
    jm.submit_review(job.id, {"action": "revise", "feedback": "Shorter, please."})
    jm.wait(job.id, until=("awaiting_review",))
    assert job.public()["review_draft"]["round"] == 2
    jm.submit_review(job.id, {"action": "approve"})
    done = jm.wait(job.id)
    assert done.result["human_approved"] and done.result["revisions"] == 1
    with pytest.raises(ValueError):
        jm.submit_review(job.id, {"action": "approve"})  # no longer waiting


def test_failed_jobs_report_errors_and_queue_is_bounded():
    jm = JobManager(offline_runner, workers=1, max_jobs=1)
    bad = jm.submit("analyze", {"ticker": "DEMO", "analysts": ["astrologer"]})
    assert jm.wait(bad.id).status == "failed" and "astrologer" in bad.error
    slow = jm.submit("analyze", {"ticker": "DEMO", "review": True})
    jm.wait(slow.id, until=("awaiting_review",))
    with pytest.raises(OverflowError):
        jm.submit("analyze", {"ticker": "ACME"})
    jm.submit_review(slow.id, {"action": "approve"})


# --------------------------------------------------------------------------- watchlist
def test_watch_diff_alerts():
    prev = {"ticker": "X", "price": 100.0, "net_sentiment": 0.3, "flags": ["deep_drawdown"], "risk_score": 15}
    cur = {"ticker": "X", "price": 92.0, "net_sentiment": -0.2, "flags": ["high_leverage"], "risk_score": 30}
    kinds = sorted(a["kind"] for a in diff(prev, cur))
    assert kinds == ["flag_cleared", "flag_raised", "price_move", "sentiment_swing"]
    assert diff(None, cur)[0]["kind"] == "baseline"
    assert diff(cur, cur) == []


def test_run_watch_persists_state_and_survives_bad_tickers(tmp_path):
    class Flaky(SampleProvider):
        def prices(self, ticker, period="2y"):
            if ticker == "BROKEN":
                raise ValueError("delisted")
            return super().prices(ticker, period)

    state = tmp_path / "state.json"
    first = run_watch(["DEMO", "BROKEN"], Flaky(), state)
    assert "BROKEN" in first["errors"] and first["alerts"][0]["kind"] == "baseline"
    second = run_watch(["DEMO"], Flaky(), state)
    assert second["alerts"] == [] and json.loads(state.read_text())["DEMO"]["ticker"] == "DEMO"


# --------------------------------------------------------------------------- REST API
def test_rest_api_end_to_end(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    import finsight.api as api

    monkeypatch.setattr(api, "manager", JobManager(offline_runner, workers=1))
    monkeypatch.setenv("FINSIGHT_API_KEY", "s3cret")
    c = TestClient(api.app)
    assert c.get("/healthz").json()["ok"]
    assert c.post("/analyze", json={"ticker": "DEMO"}).status_code == 401
    h = {"X-API-Key": "s3cret"}
    assert c.post("/analyze", json={"ticker": "../etc"}, headers=h).status_code == 422
    r = c.post("/analyze", json={"ticker": "demo", "review": True}, headers=h)
    assert r.status_code == 202
    jid = r.json()["job_id"]
    api.manager.wait(jid, until=("awaiting_review",))
    assert c.post(f"/jobs/{jid}/review", json={"action": "approve"}, headers=h).json()["accepted"] == "approve"
    api.manager.wait(jid)
    body = c.get(f"/jobs/{jid}", headers=h).json()
    assert body["status"] == "done" and body["result"]["approved"] and body["result"]["ticker"] == "DEMO"
    assert c.get("/jobs/nope", headers=h).status_code == 404
    pf = c.post("/portfolio", json={"holdings": "DEMO:50,ACME:50"}, headers=h).json()
    assert pf["analysis"]["var_1d_95"]["historical_var_pct"] > 0
