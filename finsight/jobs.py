"""Background job manager used by the REST API.

Analyses run on a thread pool. Each job records its progress (one entry per finished graph
node) so clients can poll it. Jobs that ask for human review pause in ``awaiting_review``
until a reviewer answers through ``submit_review`` (POST /jobs/{id}/review).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from finsight.obs import usage_summary
from finsight.state import AnalysisState


@dataclass
class Job:
    id: str
    kind: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | awaiting_review | done | failed
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    progress: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    review_draft: dict[str, Any] | None = None
    _decision: dict[str, Any] | None = None
    _event: threading.Event = field(default_factory=threading.Event)

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "status": self.status, "params": self.params,
                "created": self.created, "started": self.started, "finished": self.finished,
                "progress": self.progress, "review_draft": self.review_draft, "result": self.result,
                "error": self.error}


def state_result(state: AnalysisState) -> dict[str, Any]:
    return {
        "ticker": state.ticker,
        "approved": state.approved,
        "human_approved": state.human_approved,
        "revisions": state.revisions,
        "plan": state.plan,
        "memo": state.memo,
        "critique": state.critique,
        "risk_flags": state.risk_flags,
        "evidence": {k: {"source": v.source, "title": v.title} for k, v in state.evidence.items.items()},
        "usage": usage_summary(state),
        "security_events": state.security_events,
        "errors": state.errors,
    }


class JobManager:
    def __init__(self, runner: Callable[..., AnalysisState], workers: int = 2, max_jobs: int = 200,
                 review_timeout: float = 3600) -> None:
        self.runner = runner
        # daemon threads + a semaphore (not a ThreadPoolExecutor): a job paused for human review
        # must never stop the server process from shutting down
        self._slots = threading.BoundedSemaphore(workers)
        self.jobs: dict[str, Job] = {}
        self.max_jobs, self.review_timeout = max_jobs, review_timeout
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ submit
    def submit(self, kind: str, params: dict[str, Any], fn: Callable[[Job], dict[str, Any]] | None = None) -> Job:
        with self._lock:
            active = sum(1 for j in self.jobs.values() if j.status in ("queued", "running", "awaiting_review"))
            if active >= self.max_jobs:
                raise OverflowError("too many active jobs; try again later")
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, params=params)
            self.jobs[job.id] = job
            self._trim()
        threading.Thread(target=self._run, args=(job, fn or self._analyze), daemon=True,
                         name=f"finsight-job-{job.id}").start()
        return job

    def _trim(self) -> None:
        done = sorted((j for j in self.jobs.values() if j.status in ("done", "failed")), key=lambda j: j.created)
        for j in done[: max(0, len(self.jobs) - self.max_jobs)]:
            self.jobs.pop(j.id, None)

    def _run(self, job: Job, fn: Callable[[Job], dict[str, Any]]) -> None:
        with self._slots:  # at most `workers` analyses run at once; the rest stay queued
            job.status, job.started = "running", time.time()
            try:
                job.result = fn(job)
                job.status = "done"
            except Exception as e:  # report, never crash the worker
                job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                job.finished = time.time()

    # ------------------------------------------------------------------ analysis jobs
    def _analyze(self, job: Job) -> dict[str, Any]:
        p = job.params

        def on_step(node: str, state: AnalysisState) -> None:
            job.progress.append({"node": node, "t": round(time.time() - (job.started or job.created), 2),
                                 "evidence": len(state.evidence.items), "approved": state.approved})

        reviewer = self._reviewer(job) if p.get("review") else None
        state = self.runner(p["ticker"], p.get("question"), analysts=p.get("analysts"), peers=p.get("peers"),
                            reviewer=reviewer, on_step=on_step, **p.get("runner_kwargs", {}))
        return state_result(state)

    def _reviewer(self, job: Job) -> Callable[[AnalysisState], dict[str, Any]]:
        def review(state: AnalysisState) -> dict[str, Any]:
            job.review_draft = {"round": state.human_rounds, "memo": state.memo, "critic_approved": state.approved,
                                "critic_issues": state.critique.get("issues", [])}
            job._decision = None
            job._event.clear()
            job.status = "awaiting_review"
            if not job._event.wait(self.review_timeout):
                job.status = "running"
                return {"action": "approve"}  # reviewer never answered: keep the critic-approved draft
            job.status = "running"
            job.review_draft = None
            return job._decision or {"action": "approve"}

        return review

    def submit_review(self, job_id: str, decision: dict[str, Any]) -> Job:
        job = self.get(job_id)
        if job.status != "awaiting_review":
            raise ValueError(f"job {job_id} is not awaiting review (status: {job.status})")
        if decision.get("action") not in ("approve", "revise", "edit"):
            raise ValueError("action must be approve, revise or edit")
        job._decision = decision
        job.status = "running"  # set before waking the worker, so pollers never see a stale state
        job.review_draft = None
        job._event.set()
        return job

    # ------------------------------------------------------------------ queries
    def get(self, job_id: str) -> Job:
        try:
            return self.jobs[job_id]
        except KeyError:
            raise KeyError(f"no job {job_id}") from None

    def list(self, limit: int = 20) -> list[dict[str, Any]]:
        jobs = sorted(self.jobs.values(), key=lambda j: j.created, reverse=True)[:limit]
        return [{"id": j.id, "kind": j.kind, "status": j.status, "ticker": j.params.get("ticker"),
                 "created": j.created} for j in jobs]

    def wait(self, job_id: str, timeout: float = 30, until: tuple[str, ...] = ("done", "failed")) -> Job:
        end = time.time() + timeout
        while time.time() < end:
            j = self.get(job_id)
            if j.status in until:
                return j
            time.sleep(0.02)
        raise TimeoutError(f"job {job_id} still {self.get(job_id).status}")
