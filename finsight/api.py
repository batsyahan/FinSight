"""FinSight REST API.

    uvicorn finsight.api:app --port 8000        # or: docker compose up

Endpoints
    GET  /healthz                    liveness + configuration
    POST /analyze                    start an analysis job -> 202 {job_id}
    GET  /jobs                       recent jobs
    GET  /jobs/{id}                  status, progress, result
    POST /jobs/{id}/review           human-in-the-loop decision (approve | revise | edit)
    POST /portfolio                  portfolio risk report (synchronous, no LLM unless summary=true)
    GET  /universes                  stock universes available to the eval harness

Set FINSIGHT_API_KEY to require an ``X-API-Key`` header on every request except /healthz.
Install with: pip install "finsight[api]"
"""

from __future__ import annotations

import os
import secrets
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from finsight import __version__
from finsight.config import Settings
from finsight.graph.workflow import run_analysis
from finsight.guardrails import validate_ticker
from finsight.jobs import JobManager

app = FastAPI(title="FinSight API", version=__version__,
              description="Multi-agent AI equity research with cited, self-verified memos.")
manager = JobManager(run_analysis, workers=int(os.getenv("FINSIGHT_WORKERS", "2")))


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("FINSIGHT_API_KEY", "")
    if expected and not (x_api_key and secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")


class AnalyzeRequest(BaseModel):
    ticker: str = Field(examples=["INFY.NS"])
    question: str | None = Field(default=None, max_length=500)
    analysts: list[str] | None = None
    peers: list[str] | None = None
    review: bool = Field(default=False, description="pause for human review before finishing")

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        return validate_ticker(v)

    @field_validator("peers")
    @classmethod
    def _peers(cls, v: list[str] | None) -> list[str] | None:
        return [validate_ticker(p) for p in v] if v else v


class ReviewRequest(BaseModel):
    action: Literal["approve", "revise", "edit"]
    feedback: str | None = Field(default=None, max_length=2000)
    memo: str | None = Field(default=None, max_length=50_000)


class PortfolioRequest(BaseModel):
    holdings: str = Field(examples=["INFY.NS:40,HDFCBANK.NS:30,ITC.NS:30"])
    value: float = 1_000_000
    period: str = "2y"
    summary: bool = False


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    s = Settings.from_env()
    return {"ok": True, "version": __version__, "llm": s.llm, "model": s.model or "(default)", "data": s.data,
            "auth": bool(os.getenv("FINSIGHT_API_KEY"))}


@app.post("/analyze", status_code=202, dependencies=[Depends(require_key)])
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    try:
        job = manager.submit("analyze", req.model_dump())
    except OverflowError as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    return {"job_id": job.id, "status": job.status, "status_url": f"/jobs/{job.id}"}


@app.get("/jobs", dependencies=[Depends(require_key)])
def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    return manager.list(min(limit, 100))


@app.get("/jobs/{job_id}", dependencies=[Depends(require_key)])
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return manager.get(job_id).public()
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/jobs/{job_id}/review", dependencies=[Depends(require_key)])
def review(job_id: str, req: ReviewRequest) -> dict[str, Any]:
    try:
        job = manager.submit_review(job_id, req.model_dump())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return {"job_id": job.id, "accepted": req.action}


@app.post("/portfolio", dependencies=[Depends(require_key)])
def portfolio(req: PortfolioRequest) -> JSONResponse:
    from finsight.data.providers import get_provider
    from finsight.llm.providers import get_llm
    from finsight.portfolio import analyze_portfolio, parse_holdings, portfolio_summary

    s = Settings.from_env()
    try:
        analysis = analyze_portfolio(parse_holdings(req.holdings), get_provider(s.data), period=req.period,
                                     value=req.value)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    body: dict[str, Any] = {"analysis": analysis}
    if req.summary:
        body["summary"] = portfolio_summary(analysis, get_llm(s))
    return JSONResponse(body)


@app.get("/universes", dependencies=[Depends(require_key)])
def universes() -> dict[str, list[str]]:
    from finsight.eval import UNIVERSES

    return UNIVERSES
