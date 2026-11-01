# FinSight 🦇📈

**A multi-agent AI financial analyst.** Give it a stock ticker and a team of AI agents researches the company, runs quantitative analysis, flags risks, and writes an investment memo. Every claim in the memo is cited, and a Critic agent checks the citations before the memo is approved.

> 🚧 Work in progress. Being built in milestones; see the roadmap below.

## Architecture

```
            ┌────────────┐
 ticker ──▶ │ Researcher │  ReAct tool loop: profile, fundamentals, news
            └─────┬──────┘
                  ▼
            ┌────────────┐
            │   Quant    │  returns, volatility, Sharpe, drawdown, SMA backtest, ratios
            └─────┬──────┘
                  ▼
            ┌────────────┐
            │    Risk    │  rule-based red flags + LLM judgement
            └─────┬──────┘
                  ▼
            ┌────────────┐   needs revision   ┌────────────┐
            │   Writer   │ ◀───────────────── │   Critic   │
            └─────┬──────┘ ─────────────────▶ └─────┬──────┘
                  │                                 │ approved
                  ▼                                 ▼
                                 memo.md (with [S#] citations)
```

## Roadmap

- [x] Milestone 1: project scaffold, config, shared state and evidence store
- [x] Milestone 2: market data layer and quant/risk tools, with tests
- [x] Milestone 3: agents, ReAct loop, Writer↔Critic graph
- [ ] Milestone 4: CLI, Streamlit UI, CI

## License

MIT
