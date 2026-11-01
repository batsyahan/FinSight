# FinSight 📈

![CI](https://github.com/batsyahan/FinSight/actions/workflows/ci.yml/badge.svg)

**A multi-agent AI financial analyst.** Give it a stock ticker and a team of AI agents researches the company, runs quantitative analysis, flags risks, and writes an investment memo. Every figure in the memo is cited, and a Critic agent checks each one against the evidence before the memo is approved.

> Educational project. FinSight's output is research, not financial advice.

## Why it's interesting

Most "AI analyst" demos let an LLM write numbers from memory. FinSight doesn't:

- **Tools do the maths, agents do the reasoning.** Returns, volatility, Sharpe, drawdown, backtests and ratios come from tested Python functions, never from the model.
- **Every tool result becomes citable evidence** (`S1`, `S2`, …). The Writer must cite them.
- **The Critic verifies, not just reviews.** A deterministic checker extracts every figure in the draft and matches it against the evidence (handling rounding, signs and crore/billion scaling). It also rejects unknown citations and numeric sentences with no citation. Then an LLM review covers balance and tone. Rejected drafts go back to the Writer with the exact issues.
- **Native tool calling, with tools required first.** For Claude and OpenAI-compatible APIs the agents use the API's own tool calling, and the first turn sets `tool_choice=required`. The model cannot answer from memory before it has evidence. Models without tool calling fall back to a JSON ReAct protocol.
- **Provider-agnostic.** It works with Claude, any OpenAI-compatible API (OpenAI, Groq, Ollama, OpenRouter), or a deterministic offline mode that needs no API key.
- **Planner agent.** It reads the investor's question and picks which analysts to run and what the memo should focus on. A keyword planner is the fallback.
- **Human in the loop.** With `--review` you approve, request changes or edit the memo. Edits go back through the Critic, so a human can't slip in an unsupported figure either.
- **Runs on LangGraph** when installed, or on a small built-in state-graph engine with the same node functions.

## Architecture

```
                        ┌────────────┐
                   ┌──▶ │ Researcher │  profile, financial statements, news
                   │    └────────────┘
 ticker ─▶ Planner ┼──▶ │   Quant    │  price stats, SMA backtest, ratios        (parallel)
                   │    └────────────┘
                   └──▶ │   Peers    │  sector peer group, side-by-side ranking
                        └─────┬──────┘
                              ▼  fan-in
            ┌────────────┐
            │    Risk    │  rule-based flags (transparent thresholds) + LLM judgement
            └─────┬──────┘
                  ▼
            ┌────────────┐   revise (with issues)   ┌────────────┐
            │   Writer   │ ◀─────────────────────── │   Critic   │  grounding checker
            └─────┬──────┘ ───────────────────────▶ └─────┬──────┘  + LLM review
                                                          │ approved / budget used
                                                          ▼
                                         memo.md (with [S#] citations) + trace.json
```

| Layer | Files |
|---|---|
| Shared state and evidence store | `finsight/state.py` |
| Market data (yfinance, offline sample) | `finsight/data/providers.py` |
| Quant and risk tools, tool registry | `finsight/tools/` |
| LLM clients (Anthropic, OpenAI-compatible, mock) | `finsight/llm/` |
| ReAct loop, prompts, the five agents | `finsight/agents/` |
| Grounding checker | `finsight/verify.py` |
| Graph (LangGraph or built-in) | `finsight/graph/` |
| CLI and Streamlit UI | `finsight/cli.py`, `app/streamlit_app.py` |

## Quick start

```bash
git clone https://github.com/batsyahan/FinSight.git
cd FinSight
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"

# 1. Offline demo: no API key, synthetic data
finsight analyze DEMO --print
```

### Real data and a real model

```bash
cp .env.example .env   # then edit it
```

**Claude**
```bash
FINSIGHT_LLM=anthropic  ANTHROPIC_API_KEY=sk-ant-...  FINSIGHT_DATA=yfinance
```

**Groq (free tier) or any OpenAI-compatible API**
```bash
FINSIGHT_LLM=openai  OPENAI_BASE_URL=https://api.groq.com/openai/v1  OPENAI_API_KEY=gsk_...  FINSIGHT_MODEL=<model name>
```

**Ollama (fully local)**
```bash
FINSIGHT_LLM=openai  OPENAI_BASE_URL=http://localhost:11434/v1  FINSIGHT_MODEL=llama3.1
```

```bash
finsight analyze INFY.NS -q "Is the valuation justified by growth?"
finsight analyze RELIANCE.NS --data yfinance --llm anthropic
finsight analyze INFY.NS --peers TCS.NS,WIPRO.NS,HCLTECH.NS     # choose peers
finsight analyze INFY.NS --analysts quant,peers                # run only some analysts
finsight analyze INFY.NS --review                              # human-in-the-loop approval
```

NSE tickers use `.NS` and BSE tickers use `.BO`. Reports are saved to `reports/<TICKER>_memo.md` and `reports/<TICKER>_trace.json`. The trace contains every agent thought, tool call and evidence item.

### Web UI

```bash
streamlit run app/streamlit_app.py
```

It shows live agent progress, the memo, a price chart with the 50/200 SMAs, the evidence behind each citation, and the full agent trace.

## Evaluation and observability

Every LLM call is recorded per agent: tokens, latency and estimated cost. A summary is printed after each run and saved in the trace.

```bash
finsight eval --universe nifty-it --data yfinance --pause 20   # benchmark across stocks
finsight eval-compare reports/eval_*.json                      # compare models side by side
```

Universes: `demo` (offline), `nifty-it`, `nifty-top10`, `banks`, or `--tickers A,B,C`. Each eval writes a JSON file and a Markdown report covering:
- approval rate and first-draft approval;
- **% of figures grounded in evidence**;
- average revisions, failed tools, uncited findings dropped;
- tokens, latency and cost per run.

The grounding checker matches each figure to the **precision it is written with** ("37.7" must match 37.6–37.8). It only rescales a figure when a unit follows it ("20.16 billion", "4,117 crore"). The eval harness itself found that looser rules let made-up figures pass by coincidence.

## Example output

- [`examples/INFY_NS_memo.md`](examples/INFY_NS_memo.md): **live run** on Infosys (NSE) with Groq `openai/gpt-oss-120b` and Yahoo Finance data. The Critic rejected the first draft with 2 issues; the Writer fixed them and the second draft was approved (about 2 minutes on Groq's free tier).
- [`examples/DEMO_memo.md`](examples/DEMO_memo.md): offline run with synthetic data and the mock model.

## Lessons from live runs

Running on a real model exposed failure modes that the offline tests didn't. Each now has a regression test:

| Seen on `gpt-oss-120b` (Groq) | Fix |
|---|---|
| Answered from memory, 0 tool calls | Core tools run first; uncited findings dropped |
| Called a non-existent `JSON` tool (HTTP 400) | `submit_findings` tool, and the model's text is recovered from `failed_generation` |
| Wrote text when a tool call was required | Findings accepted as a tool call, JSON or a cited bullet list |
| Spent its output budget reasoning | `reasoning_effort=low` and a token floor for reasoning models |
| Free-tier 429 rate limits | Waits for the provider's "try again in Xs" |
| `52‑week` with a Unicode hyphen failed the checker | Dashes normalised before verification |
| Mixed "$" and INR for one figure | Separate statement and price currencies; currency check |

## Tests

```bash
pytest
```

The tests cover the quant maths, risk rules, the evidence registry, the grounding checker, the ReAct loop's recovery from bad JSON and unknown tools, and the full graph. The revision-loop test uses a model that makes up a figure in its first draft. The Critic must reject that draft and the Writer must fix it. CI runs the suite with and without LangGraph on Python 3.10 and 3.12.

## Roadmap

- [x] Scaffold, config, shared state and evidence store
- [x] Market data layer and quant/risk tools
- [x] Agents, ReAct loop, Writer↔Critic graph with grounding checks
- [x] CLI, Streamlit UI, CI
- [ ] Peer comparison agent (compare against sector peers)
- [ ] Parallel Researcher/Quant branches in LangGraph
- [ ] Filings RAG (annual reports, concall transcripts) as an evidence source
- [ ] Evaluation set: score memos for groundedness across many tickers

## License

MIT
