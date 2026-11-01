# FinSight 📈

![CI](https://github.com/batsyahan/FinSight/actions/workflows/ci.yml/badge.svg)

**A multi-agent AI financial analyst.** Give it a stock ticker and a team of AI agents researches the company, runs quantitative analysis, flags risks, and writes an investment memo. Every figure in the memo is cited, and a Critic agent checks each one against the evidence before the memo is approved.

> Educational project. FinSight's output is research, not financial advice.

## Why it's interesting

Most "AI analyst" demos let an LLM write numbers from memory. FinSight doesn't:

- **Tools do the maths, agents do the reasoning.** Returns, volatility, Sharpe, drawdown, backtests and ratios come from tested Python functions, never from the model.
- **Every tool result becomes citable evidence** (`S1`, `S2`, …). The Writer must cite them.
- **The Critic verifies, not just reviews.** A deterministic checker extracts every figure in the draft and matches it against the evidence (handling rounding, signs and crore/billion scaling). It also rejects unknown citations and numeric sentences with no citation. Then an LLM review covers balance and tone. Rejected drafts go back to the Writer with the exact issues.
- **Provider-agnostic.** It works with Claude, any OpenAI-compatible API (OpenAI, Groq, Ollama, OpenRouter), or a deterministic offline mode that needs no API key.
- **Runs on LangGraph** when installed, or on a small built-in state-graph engine with the same node functions.

## Architecture

```
            ┌────────────┐
 ticker ──▶ │ Researcher │  ReAct tool loop: profile, financial statements, news
            └─────┬──────┘
                  ▼
            ┌────────────┐
            │   Quant    │  ReAct tool loop: price stats, SMA backtest, ratios
            └─────┬──────┘
                  ▼
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
```

NSE tickers use `.NS` and BSE tickers use `.BO`. Reports are saved to `reports/<TICKER>_memo.md` and `reports/<TICKER>_trace.json`. The trace contains every agent thought, tool call and evidence item.

### Web UI

```bash
streamlit run app/streamlit_app.py
```

It shows live agent progress, the memo, a price chart with the 50/200 SMAs, the evidence behind each citation, and the full agent trace.

## Example output

See [`examples/DEMO_memo.md`](examples/DEMO_memo.md) for a memo produced offline (synthetic data, mock model).

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
