# FinSight 📈

![CI](https://github.com/batsyahan/FinSight/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**A multi-agent AI equity analyst that has to show its work.** You give it a stock and a question. A Planner assembles a team of agents (Researcher, Quant, Peers, Sentiment, Filings), which run in parallel and gather evidence through tools. A Writer drafts an investment memo, and a Critic **verifies every figure against the evidence its sentence cites** before the memo is approved.

> Educational project: FinSight produces research, not financial advice.

| | |
|---|---|
| 🧠 **Agentic** | Planner → parallel analysts → Risk → Writer ⇄ Critic loop, with optional human-in-the-loop review. Runs on LangGraph. |
| 🔎 **Grounded** | Every figure must match the evidence its sentence cites, to the precision written. Made-up figures get rejected. |
| 📊 **Fintech depth** | Strategy lab with walk-forward validation, peer ranking, news sentiment, portfolio VaR/CVaR, filings RAG. |
| 🛡️ **Safe** | Prompt-injection redaction on all tool data, no personal buy/sell advice, input validation. |
| 📏 **Measured** | Evaluation harness (approval rate, % grounded figures) plus per-agent tokens, latency and cost. |
| 🚀 **Shippable** | CLI, Streamlit UI, REST API with job queue, Docker Compose with Redis, scheduled watchlist alerts, CI. |

**Example memo from a live run:** [Infosys (NSE)](examples/INFY_NS_memo.md), using Groq `gpt-oss-120b` and Yahoo Finance data. The Critic rejected the first draft and approved the second.

---

## Architecture

```
                         ┌─────────────┐
                    ┌──▶ │ Researcher  │  profile, financial statements, news
                    │    ├─────────────┤
                    ├──▶ │    Quant    │  returns, risk, ratios, strategy lab (walk-forward)
 question ─▶ Planner┤    ├─────────────┤
  + ticker          ├──▶ │    Peers    │  sector peer group, rank vs peer median       (parallel)
                    │    ├─────────────┤
                    ├──▶ │  Sentiment  │  headline tone and trend
                    │    ├─────────────┤
                    └──▶ │   Filings   │  RAG over annual reports / transcripts
                         └──────┬──────┘
                                ▼ fan-in
                         ┌─────────────┐
                         │    Risk     │  transparent rule flags + LLM judgement
                         └──────┬──────┘
                                ▼
                         ┌─────────────┐  revise with exact issues   ┌─────────────┐
                         │   Writer    │ ◀───────────────────────── │   Critic    │ grounding checker
                         └──────┬──────┘ ─────────────────────────▶ └──────┬──────┘ + LLM review
                                                                           │
                                               optional human review ◀─────┘ (approve / revise / edit)
                                                                           ▼
                                                     memo (md / html / pdf) + trace.json
```

### What makes it different

1. **Tools do the maths, agents do the reasoning.** Returns, Sharpe, drawdown, backtests, ratios, VaR and peer ranks come from tested Python functions, never from the model.
2. **Every tool result is citable evidence** (`S1`, `S2`, …). A finding without a valid citation is dropped.
3. **Citation-scoped grounding.** A figure has to appear in the evidence *its own sentence cites*, to the precision written ("37.7" matches 37.6–37.8). It only rescales a figure when a unit word follows it ("20.16 billion", "4,117 crore"). When a figure sits in a different evidence item, the Critic says so: "it appears in S5; fix the citation".
4. **Robust agent loop.** API-native tool calling with a `submit_findings` tool. Core tools are run up front, so weaker models can't skip them. Malformed provider output (Groq `tool_use_failed`) is recovered instead of crashing.
5. **Provider-agnostic.** Works with Claude, any OpenAI-compatible API (OpenAI, Groq, Ollama, OpenRouter), or a deterministic offline mode that needs no API key.

## Quick start

```bash
git clone https://github.com/batsyahan/FinSight.git && cd FinSight
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -e ".[all]"
finsight analyze DEMO --print                            # offline: synthetic data, no API key
```

Real data and a real model: `cp .env.example .env`, then set for example:

```bash
FINSIGHT_LLM=openai
OPENAI_BASE_URL=https://api.groq.com/openai/v1           # free tier; or OpenAI, Ollama (localhost:11434/v1)...
OPENAI_API_KEY=gsk_...
FINSIGHT_MODEL=openai/gpt-oss-120b
FINSIGHT_DATA=yfinance
```

## Usage

```bash
finsight analyze INFY.NS -q "Is the valuation justified by growth?"
finsight analyze INFY.NS --peers TCS.NS,WIPRO.NS,HCLTECH.NS             # choose the peer group
finsight analyze INFY.NS --analysts quant,peers                         # skip the Planner
finsight analyze INFY.NS -q "What did management say about AI deals?" --filings reports/infosys_ar.pdf
finsight analyze INFY.NS --review                                       # approve / request changes / edit
finsight analyze INFY.NS --export html,pdf                              # hover-citation HTML, PDF
finsight portfolio "INFY.NS:40,HDFCBANK.NS:30,ITC.NS:30" --memos        # portfolio risk + memo per holding
finsight watch --file watchlist.txt --webhook https://hooks.slack.com/... # daily alerts
finsight eval --universe nifty-it --pause 20                            # benchmark
finsight eval-compare reports/eval_*.json                               # compare models
streamlit run app/streamlit_app.py                                      # web UI
uvicorn finsight.api:app --port 8000                                    # REST API (docs at /docs)
docker compose up --build                                               # API + UI + Redis
```

NSE tickers end in `.NS`, BSE in `.BO`. Every run writes `reports/<TICKER>_memo.md` and a `_trace.json` with every agent step, tool call, evidence item, token count and guardrail event.

## Features

### Agents
| Agent | What it does |
|---|---|
| **Planner** | Reads the question and picks analysts and a memo focus. A keyword planner is the fallback; Filings is offered only when documents are given. |
| **Researcher** | Company profile, financial statements, recent news. |
| **Quant** | Returns, volatility, Sharpe, drawdown, fundamental ratios, **strategy lab**: SMA crossover, momentum, RSI and Bollinger strategies net of costs, with **walk-forward** in-sample vs out-of-sample results and an overfitting gap. |
| **Peers** | Curated NSE sector groups (IT, banks, FMCG, autos, pharma, …). Ranks the company against the peer median on 9 metrics. |
| **Sentiment** | An explainable finance word list with negation handling. Net sentiment and trend. |
| **Filings (RAG)** | PDF, text, Markdown or HTML. Chunked with page numbers and searched with BM25, or hybrid BM25 + embeddings with reciprocal rank fusion. The agent writes its own queries. |
| **Risk** | Transparent threshold rules (leverage, cash burn, drawdown, valuation, sentiment, …) plus LLM judgement. |
| **Writer / Critic** | Draft → verify → revise loop with a revision budget. |
| **Human review** | Approve, request changes or edit, in the CLI (`--review`) or over HTTP. Human edits go back through the Critic too. |

### Portfolio mode
Historical and parametric **VaR** (95/99%), **Expected Shortfall**, correlation matrix, diversification ratio, each holding's **share of portfolio risk**, sector **HHI** and concentration flags. The LLM summary goes through the same grounding check.

### Guardrails
- **Prompt injection.** Every tool result is scanned for instructions aimed at the AI ("ignore previous instructions", "rate this a strong buy", chat-template tokens …). Matches are redacted *before* any model sees them and logged as security events. The tests run real attacks through a poisoned news feed and a poisoned filing.
- **Advice.** Memos with direct buy/sell instructions, guaranteed-return claims or no disclaimer are rejected by the verifier.
- **Input validation.** Tickers are validated everywhere, because they also become file names.

### Service
- **REST API** (FastAPI): async jobs with per-agent progress, human review over HTTP, a bounded queue, and an optional API key.
- **Caching**: market data is cached on disk, or in Redis when `FINSIGHT_REDIS_URL` is set.
- **Docker**: a non-root image with a health check. CI builds it and runs a full analysis through the API inside the container.
- **Watchlist**: daily snapshot diffs (risk flags raised or cleared, moves of 5% or more, sentiment swings) to Slack or Discord via a scheduled GitHub Action.

| Endpoint | |
|---|---|
| `POST /analyze` | `{"ticker": "INFY.NS", "question": "...", "review": true}` → `202 {job_id}` |
| `GET /jobs/{id}` | `queued` / `running` / `awaiting_review` / `done` / `failed`, progress, result |
| `POST /jobs/{id}/review` | `{"action": "approve" \| "revise" \| "edit", "feedback": "...", "memo": "..."}` |
| `POST /portfolio` | Portfolio risk report |
| `GET /healthz` | Health and configuration |

## Evaluation and observability

`finsight eval` runs FinSight over a stock universe (`demo`, `nifty-it`, `nifty-top10`, `banks` or `--tickers`). It reports Critic approval rate, first-draft approval, **% of figures grounded in cited evidence**, average revisions, failed tools, uncited findings dropped, and tokens, latency and cost per run. Use `eval-compare` to put several models side by side. Every LLM call is recorded per agent; set `FINSIGHT_PRICE_IN` and `FINSIGHT_PRICE_OUT` for cost estimates.

## Lessons from live runs

Running on a real model and on the eval harness exposed failure modes that unit tests didn't. Each now has a regression test:

| Observed | Fix |
|---|---|
| `gpt-oss-120b` answered from memory with 0 tool calls | Core tools run first; uncited findings dropped |
| Model called a non-existent `JSON` tool (HTTP 400) | `submit_findings` tool, and the model's text is recovered from `failed_generation` |
| Model wrote text when a tool call was required | Findings accepted as a tool call, JSON or a cited bullet list |
| Reasoning model used its whole output budget | `reasoning_effort=low` and a token floor for reasoning models |
| Free-tier 429 rate limits | Waits for the provider's "try again in Xs" |
| `52‑week` with a Unicode hyphen failed the checker | Dashes normalised before verification |
| "$" and INR mixed for the same figure | Separate statement and price currencies; currency check |
| Eval: made-up figures matched other figures by coincidence | Precision-based matching, then **citation-scoped** grounding |
| CI: LangGraph couldn't resolve a type declared inside a function | Graph state type moved to module level |
| CI: API sent `"feedback": null` on approve and crashed the job | Null-safe review handling plus a test with the exact API payload |

## Project layout

| Path | Contents |
|---|---|
| `finsight/agents/` | Planner, analysts, Risk, Writer, Critic, human review; ReAct loop (native tool calling and JSON fallback); prompts |
| `finsight/tools/` | Quant, strategy lab, peers, sentiment, risk rules; tool registry (evidence + guardrails) |
| `finsight/graph/` | Workflow wiring; LangGraph backend; built-in engine with parallel fan-out |
| `finsight/verify.py` | Citation-scoped grounding checker |
| `finsight/rag/` | Filings loaders, chunking, BM25 / hybrid retrieval, cache |
| `finsight/llm/` | Anthropic and OpenAI-compatible clients over HTTP, offline mock |
| `finsight/data/` | Yahoo Finance and synthetic providers, cache (file / Redis) |
| `finsight/portfolio.py`, `watch.py`, `eval.py`, `obs.py`, `guardrails.py`, `export.py` | Portfolio risk, alerts, benchmark, telemetry, safety, HTML/PDF |
| `finsight/api.py`, `jobs.py` | REST API and job manager |
| `app/streamlit_app.py` | Web UI: live agent graph, hover citations, peers, strategy lab, portfolio, eval |

## Tests and CI

There are 80+ tests: quant maths (with a no-look-ahead check), grounding rules, ReAct recovery, parallel execution timing, planner and human review flows, RAG (including a hand-built PDF), injection attacks, the job manager, the REST API, export escaping (XSS), and the Streamlit app run headless. CI runs them on Python 3.10 and 3.12, with and without optional dependencies. It also builds the Docker image and smoke-tests the API inside the container.

## Deploy

See [docs/DEPLOY.md](docs/DEPLOY.md) for Streamlit Community Cloud (free), Docker Compose, Hugging Face Spaces, or any container host.

## License

MIT
