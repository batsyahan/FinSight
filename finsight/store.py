"""Persistence (SQLite) for the web app: watchlist, research runs, memo versions with verification
reports, risk flags with review status, backtests and settings.

The interface is small and dict-based so a production database can replace SQLite later.
Path: FINSIGHT_DB (default .finsight/finsight.db).
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from finsight.guardrails import validate_ticker
from finsight.obs import usage_summary
from finsight.state import AnalysisState
from finsight.verify import verify_memo

DEFAULT_WATCHLIST = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watchlist (ticker TEXT PRIMARY KEY, added_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, ticker TEXT NOT NULL, question TEXT, created_at TEXT NOT NULL, finished_at TEXT,
    status TEXT NOT NULL, approved INTEGER, revisions INTEGER, llm TEXT, data TEXT, plan TEXT, usage TEXT,
    critique TEXT, trace TEXT, evidence TEXT, findings TEXT, security TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS memos (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ticker TEXT NOT NULL, version INTEGER NOT NULL,
    created_at TEXT NOT NULL, memo TEXT NOT NULL, verified INTEGER NOT NULL, verification TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk_flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ticker TEXT NOT NULL, rule_id TEXT NOT NULL,
    severity TEXT NOT NULL, metric TEXT, value REAL, threshold TEXT, message TEXT, evidence_id TEXT,
    category TEXT, detected_at TEXT NOT NULL, review_status TEXT NOT NULL DEFAULT 'open', reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS backtests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT NOT NULL, strategy TEXT NOT NULL, params TEXT,
    config TEXT, metrics TEXT, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_runs_ticker ON runs(ticker);
CREATE INDEX IF NOT EXISTS ix_memos_ticker ON memos(ticker);
CREATE INDEX IF NOT EXISTS ix_flags_ticker ON risk_flags(ticker);
"""

RISK_CATEGORIES = {
    "high_leverage": "Excessive leverage", "elevated_leverage": "Excessive leverage",
    "negative_fcf": "Weak cash-flow conversion", "loss_making": "Financial deterioration",
    "revenue_decline": "Financial deterioration", "weak_liquidity": "Financial deterioration",
    "high_volatility": "Unusual price behaviour", "deep_drawdown": "Unusual historical drawdown",
    "rich_valuation": "Valuation concern", "poor_risk_adjusted_return": "Unusual price behaviour",
    "negative_news_sentiment": "News and disclosures",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("FINSIGHT_DB", ".finsight/finsight.db"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=10)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------ settings
    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._conn() as c:
            row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                      (key, json.dumps(value)))

    # ------------------------------------------------------------------ watchlist
    def watchlist(self) -> list[str]:
        if not self.get_setting("watchlist_initialised", False):
            for t in DEFAULT_WATCHLIST:
                self.add_to_watchlist(t)
            self.set_setting("watchlist_initialised", True)
        with self._conn() as c:
            return [r["ticker"] for r in c.execute("SELECT ticker FROM watchlist ORDER BY rowid")]

    def add_to_watchlist(self, ticker: str) -> str:
        t = validate_ticker(ticker)
        with self._conn() as c:
            c.execute("INSERT OR IGNORE INTO watchlist(ticker, added_at) VALUES(?, ?)", (t, now()))
        return t

    def remove_from_watchlist(self, ticker: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM watchlist WHERE ticker=?", (ticker.upper(),))

    # ------------------------------------------------------------------ runs + memos + flags
    def save_run(self, state: AnalysisState, *, llm: str = "", data: str = "", started_at: str | None = None,
                 run_id: str | None = None) -> str:
        rid = run_id or uuid.uuid4().hex[:12]
        verification = verify_memo(state.memo, state.evidence) if state.memo else {"passed": False, "claims": []}
        evidence = {k: {"source": e.source, "title": e.title, "data": e.data} for k, e in state.evidence.items.items()}
        findings = [{"agent": f.agent, "text": f.text, "citations": f.citations} for f in state.findings]
        risk_ev = next((e.id for e in state.evidence.items.values() if e.source == "risk_rules"), None)
        ts = now()
        with self._conn() as c:
            c.execute("""INSERT INTO runs(id, ticker, question, created_at, finished_at, status, approved, revisions,
                         llm, data, plan, usage, critique, trace, evidence, findings, security, error)
                         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (rid, state.ticker, state.question, started_at or ts, ts, "completed", int(state.approved),
                       state.revisions, llm, data, json.dumps(state.plan), json.dumps(usage_summary(state)),
                       json.dumps(state.critique, default=str), json.dumps(state.trace, default=str),
                       json.dumps(evidence, default=str), json.dumps(findings), json.dumps(state.security_events),
                       json.dumps(state.errors)))
            if state.memo:
                version = c.execute("SELECT COUNT(*) FROM memos WHERE ticker=?", (state.ticker,)).fetchone()[0] + 1
                c.execute("""INSERT INTO memos(run_id, ticker, version, created_at, memo, verified, verification)
                             VALUES(?,?,?,?,?,?,?)""",
                          (rid, state.ticker, version, ts, state.memo, int(state.approved and verification["passed"]),
                           json.dumps(verification, default=str)))
            for f in state.risk_flags:
                c.execute("""INSERT INTO risk_flags(run_id, ticker, rule_id, severity, metric, value, threshold,
                             message, evidence_id, category, detected_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                          (rid, state.ticker, f["id"], f["severity"], f.get("metric"), _num(f.get("value")),
                           f.get("threshold"), f.get("message"), risk_ev, RISK_CATEGORIES.get(f["id"], "Other"), ts))
        return rid

    def save_failed_run(self, ticker: str, question: str, error: str, started_at: str | None = None) -> str:
        rid = uuid.uuid4().hex[:12]
        with self._conn() as c:
            c.execute("INSERT INTO runs(id, ticker, question, created_at, finished_at, status, error) VALUES(?,?,?,?,?,?,?)",
                      (rid, ticker, question, started_at or now(), now(), "failed", json.dumps([error])))
        return rid

    def runs(self, ticker: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM runs", []
        if ticker:
            q, args = q + " WHERE ticker=?", [ticker.upper()]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY created_at DESC LIMIT ?", (*args, limit)).fetchall()
        return [_decode(dict(r)) for r in rows]

    def run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return _decode(dict(row)) if row else None

    def latest_run(self, ticker: str | None = None) -> dict[str, Any] | None:
        rows = [r for r in self.runs(ticker, limit=10) if r["status"] == "completed"]
        return rows[0] if rows else None

    def memos(self, ticker: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        q, args = "SELECT * FROM memos", []
        if ticker:
            q, args = q + " WHERE ticker=?", [ticker.upper()]
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY created_at DESC, id DESC LIMIT ?", (*args, limit)).fetchall()
        return [_decode(dict(r)) for r in rows]

    def memo(self, memo_id: int) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM memos WHERE id=?", (memo_id,)).fetchone()
        return _decode(dict(row)) if row else None

    def flags(self, status: str | None = None, severity: str | None = None, ticker: str | None = None,
              latest_only: bool = True) -> list[dict[str, Any]]:
        """Risk flags. latest_only keeps each ticker's flags from its most recent run only."""
        q = "SELECT * FROM risk_flags WHERE 1=1"
        args: list[Any] = []
        for col, val in (("review_status", status), ("severity", severity), ("ticker", ticker)):
            if val:
                q += f" AND {col}=?"
                args.append(val)
        if latest_only:
            q += " AND run_id IN (SELECT id FROM runs r WHERE created_at = (SELECT MAX(created_at) FROM runs " \
                 "WHERE ticker=r.ticker AND status='completed'))"
        with self._conn() as c:
            rows = c.execute(q + " ORDER BY detected_at DESC, id", args).fetchall()
        return [dict(r) for r in rows]

    def set_flag_status(self, flag_id: int, status: str) -> None:
        if status not in ("open", "reviewed", "dismissed"):
            raise ValueError("status must be open, reviewed or dismissed")
        with self._conn() as c:  # evidence is never deleted; only the review status changes
            c.execute("UPDATE risk_flags SET review_status=?, reviewed_at=? WHERE id=?",
                      (status, now() if status != "open" else None, flag_id))

    # ------------------------------------------------------------------ backtests
    def save_backtest(self, ticker: str, strategy: str, params: dict[str, Any], config: dict[str, Any],
                      metrics: dict[str, Any]) -> int:
        with self._conn() as c:
            cur = c.execute("INSERT INTO backtests(ticker, strategy, params, config, metrics, created_at) "
                            "VALUES(?,?,?,?,?,?)", (ticker, strategy, json.dumps(params), json.dumps(config),
                                                    json.dumps(metrics, default=str), now()))
            return int(cur.lastrowid)

    def backtests(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM backtests ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [_decode(dict(r)) for r in rows]

    # ------------------------------------------------------------------ dashboard
    def kpis(self) -> dict[str, Any]:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        with self._conn() as c:
            runs_month = c.execute("SELECT COUNT(*) FROM runs WHERE substr(created_at,1,7)=?", (month,)).fetchone()[0]
            runs_total = c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            bt = c.execute("SELECT COUNT(*) FROM backtests").fetchone()[0]
        open_flags = self.flags(status="open")
        return {"companies_tracked": len(self.watchlist()), "runs_this_month": runs_month, "runs_total": runs_total,
                "backtests": bt, "open_flags": len(open_flags),
                "open_high": sum(f["severity"] == "high" for f in open_flags)}


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_JSON_COLS = {"plan", "usage", "critique", "trace", "evidence", "findings", "security", "error", "verification",
              "params", "config", "metrics"}


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    for k in _JSON_COLS & set(row):
        if isinstance(row[k], str):
            try:
                row[k] = json.loads(row[k])
            except json.JSONDecodeError:
                pass
    return row
